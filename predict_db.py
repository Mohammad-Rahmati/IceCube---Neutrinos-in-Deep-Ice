import time
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.adam import Adam
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa, ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher3DLoss, VonMisesFisher2DLoss
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader
from graphnet.utilities.logging import get_logger
from pytorch_lightning import Trainer
import pandas as pd
import torch
from typing import Any, Dict, List, Optional
import gc


def build_model(config: Dict[str,Any], train_dataloader: Any, pooling_list: List) -> StandardModel:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=pooling_list,
    )

    if config["target"] == 'direction':
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
    """Constructs training and validation dataloaders for training with early stopping."""
    
    train_dataloader = make_dataloader(db = config['path'],
                                            selection = pd.read_csv(config['train_selection'])[config['index_column']].ravel().tolist() if config['train_selection'] else None,
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
                                            selection = pd.read_csv(config['validate_selection'])[config['index_column']].ravel().tolist() if config['validate_selection'] else None,
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

def load_pretrained_model(config: Dict[str,Any], state_dict_path: str = '', pooling_list: List = []) -> StandardModel:
    train_dataloader, _ = make_dataloaders(config = config)
    model = build_model(config = config, 
                        train_dataloader = train_dataloader,
                        pooling_list=pooling_list)
    
    model.load_state_dict(state_dict_path)
    model.prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
    model.additional_attributes = ['event_id']
    return model


def inference(model, config: Dict[str, Any]) -> pd.DataFrame:
    """Applies model to the database specified in config['inference_database_path'] and saves results to disk."""
    # Make Dataloader
    test_dataloader = make_dataloader(db = config['inference_database_path'],
                                            selection = None, # Entire database
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False,
                                            labels = None, # Cannot make labels in test data
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            )
    
    # Get predictions
    results = model.predict_as_dataframe(
        gpus = [0],
        dataloader = test_dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=['event_id']
    )
    return results


def prepare_dataframe(df, angle_post_fix = '_reco', vec_post_fix = '') -> pd.DataFrame:
    r = np.sqrt(df['direction_x'+ vec_post_fix]**2 + df['direction_y'+ vec_post_fix]**2 + df['direction_z' + vec_post_fix]**2)
    df['zenith' + angle_post_fix] = np.arccos(df['direction_z'+ vec_post_fix]/r)
    df['azimuth'+ angle_post_fix] = np.arctan2(df['direction_y'+ vec_post_fix],df['direction_x' + vec_post_fix])
    df['azimuth'+ angle_post_fix][df['azimuth'  + angle_post_fix]<0] = df['azimuth'  + angle_post_fix][df['azimuth'  +  angle_post_fix]<0] + 2*np.pi 

    return df[['event_id', 'azimuth', 'zenith']].set_index('event_id')

# Constants
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE

# Configuration
config = {
        "path": f'data/F4/focus_batch_4.db',
        "inference_database_path": None,
        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'submission',
        "batch_size": 100,
        "num_workers": 32,
        "target": 'direction',
        "early_stopping_patience": 5,
        "fit": {
                "max_epochs": 50,
                "gpus": [0],
                "distribution_strategy": None,
                },
        'train_selection': None,
        'validate_selection':  None,
        'test_selection': None,
        'base_dir': 'training'
}

for b_id in [4]:
    config['inference_database_path'] = f'data/F{b_id}/focus_batch_{b_id}.db'
    test_dataloader = make_dataloader(db = config['inference_database_path'],
                                                selection = None, # Entire database
                                                pulsemaps = config['pulsemap'],
                                                features = features,
                                                truth = truth,
                                                batch_size = config['batch_size'],
                                                num_workers = config['num_workers'],
                                                shuffle = False,
                                                labels = None, # Cannot make labels in test data
                                                index_column = config['index_column'],
                                                truth_table = config['truth_table'],
                                                )

    m_list = [3]
    # m_list.remove(b_id)
    for m_id in m_list:
        counter = 0
        while True:
            try:
                with open('log_prediction.txt', 'a') as p:
                    current_time = time.time()
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

                    p.write("\ntime:" +  str(formatted_time))
                    p.write('\n' + str(m_id) + '_' + str(b_id) + '\n')
                    p.write('-'*50)

                gc.collect()
                torch.cuda.empty_cache()
                checkpoint = torch.load(f'training/batch_{m_id}/F{m_id}.ckpt')
                model = load_pretrained_model(config = config, state_dict_path = checkpoint['state_dict'], pooling_list=["min", "max", "mean", "sum"])
                pred = model.predict_as_dataframe(
                        gpus = [0],
                        dataloader = test_dataloader,
                        prediction_columns=model.prediction_columns,
                        additional_attributes=['event_id']
                    )
                print('start preparing dataframe')
                pred.set_index('event_id').to_pickle(f'./inference/pred_M{m_id}_F{b_id}.pkl')

                break
            except Exception as e:
                with open('error_prediction.txt', 'a') as f:
                    current_time = time.time()
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

                    f.write("\ntime:" + str(formatted_time))
                    f.write('\n' + str(e) + '\n')
                    f.write('-'*50)

                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
                gc.collect()
                
                time.sleep(10)
                if counter > 5:
                    break
                counter += 1
