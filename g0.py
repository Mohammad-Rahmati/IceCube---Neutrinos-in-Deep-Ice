#!/usr/bin/env python3

import subprocess
import sys
import os

# get the command-line arguments
args = sys.argv

# Move software to working disk
subprocess.run(['rm', '-r', 'software'])
subprocess.run(['scp', '-r', '/kaggle/input/graphnet-and-dependencies/software', '.'])

# Install dependencies
subprocess.run(['pip', 'install', '/kaggle/working/software/dependencies/torch-1.11.0+cu115-cp37-cp37m-linux_x86_64.whl'])
subprocess.run(['pip', 'install', '/kaggle/working/software/dependencies/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl'])
subprocess.run(['pip', 'install', '/kaggle/working/software/dependencies/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl'])
subprocess.run(['pip', 'install', '/kaggle/working/software/dependencies/torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl'])
subprocess.run(['pip', 'install', '/kaggle/working/software/dependencies/torch_geometric-2.0.4.tar.gz'])

# define the command to install GraphNeT
command = 'cd software/graphnet;pip install --no-index --find-links="/kaggle/working/software/dependencies" -e .[torch]'

# run the command in a subprocess
subprocess.run(command, shell=True, check=True)

# Append to PATH
sys.path.append('/kaggle/working/software/graphnet/src')

import pyarrow.parquet as pq
import sqlite3
import pandas as pd
import sqlalchemy
from tqdm import tqdm
import os
from typing import Any, Dict, List, Optional
import numpy as np

from graphnet.data.sqlite.sqlite_utilities import create_table

def load_input(meta_batch: pd.DataFrame, input_data_folder: str) -> pd.DataFrame:
        """
        Will load the corresponding detector readings associated with the meta data batch.
        """
        batch_id = pd.unique(meta_batch['batch_id'])

        assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"
        
        detector_readings = pd.read_parquet(path = f'{input_data_folder}/batch_{batch_id[0]}.parquet')
        sensor_positions = geometry_table.loc[detector_readings['sensor_id'], ['x', 'y', 'z']]
        sensor_positions.index = detector_readings.index

        for column in sensor_positions.columns:
            if column not in detector_readings.columns:
                detector_readings[column] = sensor_positions[column]

        detector_readings['auxiliary'] = detector_readings['auxiliary'].replace({True: 1, False: 0})
        return detector_readings.reset_index()

def add_to_table(database_path: str,
                      df: pd.DataFrame,
                      table_name:  str,
                      is_primary_key: bool,
                      ) -> None:
    """Writes meta data to sqlite table. 

    Args:
        database_path (str): the path to the database file.
        df (pd.DataFrame): the dataframe that is being written to table.
        table_name (str, optional): The name of the meta table. Defaults to 'meta_table'.
        is_primary_key(bool): Must be True if each row of df corresponds to a unique event_id. Defaults to False.
    """
    try:
        print(database_path)
        create_table(   columns=  df.columns,
                        database_path = database_path, 
                        table_name = table_name,
                        integer_primary_key= is_primary_key,
                        index_column = 'event_id')
    except sqlite3.OperationalError as e:
        if 'already exists' in str(e):
            pass
        else:
            raise e
    engine = sqlalchemy.create_engine("sqlite:///" + database_path)
    df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize = 200000)
    engine.dispose()
    return

def convert_to_sqlite(meta_data_path: str,
                      database_path: str,
                      input_data_folder: str,
                      batch_size: int = 200000) -> None:
    """Converts a selection of the Competition's parquet files to a single sqlite database.

    Args:
        meta_data_path (str): Path to the meta data file.
        batch_size (int): the number of rows extracted from meta data file at a time. Keep low for memory efficiency.
        database_path (str): path to database. E.g. '/my_folder/data/my_new_database.db'
        input_data_folder (str): folder containing the parquet input files.
        accepted_batch_ids (List[int]): The batch_ids you want converted. Defaults to None (all batches will be converted)
    """
    meta_data_iter = pq.ParquetFile(meta_data_path).iter_batches(batch_size = batch_size)
    
    if not database_path.endswith('.db'):
        database_path = database_path+'.db'
        
    converted_batches = [] 
    progress_bar = tqdm(total = None)
    for meta_data_batch in meta_data_iter:
        unique_batch_ids = pd.unique(meta_data_batch['event_id']).tolist()
        meta_data_batch  = meta_data_batch.to_pandas()
        add_to_table(database_path = database_path,
                    df = meta_data_batch,
                    table_name='meta_table',
                    is_primary_key= True)
        pulses = load_input(meta_batch=meta_data_batch, input_data_folder= input_data_folder)
        del meta_data_batch # memory
        add_to_table(database_path = database_path,
                    df = pulses,
                    table_name='pulse_table',
                    is_primary_key= False)
        del pulses # memory
        progress_bar.update(1)
    progress_bar.close()
    del meta_data_iter # memory
    print(f'Conversion Complete!. Database available at\n {database_path}')


subprocess.run(['rm', '/kaggle/working/test_database.db'])

input_data_folder = '/kaggle/input/icecube-neutrinos-in-deep-ice/test'
geometry_table = pd.read_csv('/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv')
meta_data_path = '/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet'

database_path = '/kaggle/working/test_database.db'
convert_to_sqlite(meta_data_path,
                  database_path=database_path,
                  input_data_folder=input_data_folder)

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
from pytorch_lightning import Trainer
import pandas as pd
import gc
import torch

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


def prepare_dataframe(df, angle_post_fix = '_reco', vec_post_fix = '') -> pd.DataFrame:
    r = np.sqrt(df['direction_x'+ vec_post_fix]**2 + df['direction_y'+ vec_post_fix]**2 + df['direction_z' + vec_post_fix]**2)
    df['zenith' + angle_post_fix] = np.arccos(df['direction_z'+ vec_post_fix]/r)
    df['azimuth'+ angle_post_fix] = np.arctan2(df['direction_y'+ vec_post_fix],df['direction_x' + vec_post_fix]) #np.sign(results['true_y'])*np.arccos((results['true_x'])/(np.sqrt(results['true_x']**2 + results['true_y']**2)))
    df['azimuth'+ angle_post_fix][df['azimuth'  + angle_post_fix]<0] = df['azimuth'  + angle_post_fix][df['azimuth'  +  angle_post_fix]<0] + 2*np.pi 

    return df[['event_id', 'azimuth', 'zenith']].set_index('event_id')

# Constants
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE

# Configuration
config = {
        "path": '/kaggle/working/test_database.db',
        "inference_database_path": '/kaggle/working/test_database.db',
        "pulsemap": 'pulse_table',
        "truth_table": 'meta_table',
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'submission',
        "batch_size": 100,
        "num_workers": 2,
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

# conn = sqlite3.connect(config['inference_database_path'])
# c = conn.cursor()
# c.execute("SELECT DISTINCT(event_id) FROM pulse_table LIMIT 200;")
# event_ids_list = [i[0] for i in c.fetchall()]

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

model   = load_pretrained_model(config = config, state_dict_path = '/kaggle/input/model-1/m0.pth', pooling_list=["min", "max", "mean", "sum"])
pred = model.predict_as_dataframe(
        gpus = [0],
        dataloader = test_dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=['event_id']
    )

pred.to_csv('/kaggle/working/output.csv', index=False)