import os
import pandas as pd
from typing import Any, Dict, List
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.adam import Adam
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader
import torch
import time
import gc


def build_model(config: Dict[str,Any], train_dataloader: Any) -> StandardModel:
    
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"]
    )
    
    task = DirectionReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=VonMisesFisher3DLoss(),
    )
    prediction_columns = [config["target"] + "_x", 
                            config["target"] + "_y", 
                            config["target"] + "_z", 
                            config["target"] + "_kappa" ]
                            
    additional_attributes = ['zenith', 'azimuth', 'event_id']

    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(train_dataloader) / 2,
                len(train_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-02, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

def make_dataloaders(config: Dict[str, Any]) -> List[Any]:
    
    train_dataloader = make_dataloader(db = config['path'],
                                            selection = pd.read_pickle(config['train_selection'])[config['index_column']].ravel().tolist(),
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = True,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            )
    
    validate_dataloader = make_dataloader(db = config['path'],
                                            selection = pd.read_pickle(config['validate_selection'])[config['index_column']].ravel().tolist(),
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            )
                                            
    return train_dataloader, validate_dataloader

def train_dynedge_from_scratch(config: Dict[str, Any]) -> StandardModel:
    idx = 3

    train_dataloader, validate_dataloader = make_dataloaders(config = config)

    model = build_model(config, train_dataloader)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"]
        ),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(config["base_dir"], config["run_name_tag"]),
            filename=f"M{idx}",
            save_top_k=1,
            mode="min",
            save_weights_only = True
        ),
        ModelCheckpoint(
            monitor= None,
            mode = 'min', 
            every_n_train_steps = 0, 
            every_n_epochs = 1, 
            train_time_interval = None
        )

    ]
    
    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks=callbacks,
        **config["fit"],
    )
    return model

# Constants
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE

# Configuration
idx = 3
config = {
        "path": f'data/extra_big_batch_{idx}.db',
        "inference_database_path": '',
        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": f'batch_{idx}',
        "batch_size": 512,
        "num_workers": 32,
        "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
                "max_epochs": 5,
                "gpus": [0],
                "distribution_strategy": None,
                "ckpt_path": None
                },
        'train_selection': f'./data/train_selection_max_500_pulses_{idx}.pkl',
        'validate_selection': f'./data/validate_selection_max_500_pulses_{idx}.pkl',
        'test_selection': None,
        'base_dir': 'training'
}

while True:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


        files = os.listdir('checkpoints/')
        if files:
            config['fit']['ckpt_path'] = 'checkpoints/' + files[0]
        else:
            config['fit']['ckpt_path'] = None
        model = train_dynedge_from_scratch(config=config)
        torch.save(model.state_dict(), f'M{idx}.pth')

        break
    except Exception as e:
        with open('error.txt', 'a') as f:
            f.write('\n' + str(e) + '\n')
            f.write('-'*50)

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
        
        time.sleep(10)
        