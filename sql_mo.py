import os, sys
from graphnet.data.sqlite.sqlite_utilities import create_table
import pandas as pd
from sklearn.model_selection import train_test_split
import sqlite3
import pyarrow.parquet as pq
import sqlalchemy
from tqdm import tqdm
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import gc

input_data_folder = './data/train'
meta_data_path = './data/train_meta.parquet'
geometry_table = pd.read_csv('./data/sensor_geometry.csv')

def load_input(batch_id: int, input_data_folder: str, event_ids: []) -> pd.DataFrame:
        
        detector_readings = pd.read_parquet(path = f'{input_data_folder}/batch_{batch_id}.parquet')
        detector_readings = detector_readings.loc[detector_readings.index.isin(event_ids)]
        sensor_positions = geometry_table.loc[detector_readings['sensor_id'], ['x', 'y', 'z']]
        sensor_positions.index = detector_readings.index
        detector_readings_copy = detector_readings.copy()
        detector_readings_copy.loc[:, 'x'] = sensor_positions['x']
        detector_readings_copy.loc[:, 'y'] = sensor_positions['y']
        detector_readings_copy.loc[:, 'z'] = sensor_positions['z']
        detector_readings = detector_readings_copy
        del detector_readings_copy
        detector_readings['auxiliary'].replace({True: 1, False: 0}, inplace=True)
        
        return detector_readings.reset_index()

def add_to_table(database_path: str,
                      df: pd.DataFrame,
                      table_name:  str,
                      is_primary_key: bool,
                      engine: sqlalchemy.engine.base.Engine) -> None:
                      
    try:
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
   
    df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize = 200000)
    engine.dispose()
    return

def convert_to_sqlite(meta_data_path: str,
                      database_path: str,
                      input_data_folder: str,
                      batch_size: int = 200000,
                      batch_ids: list = [],
                      event_ids: list = [],
                      engine: sqlalchemy.engine.base.Engine = None
                      ) -> None:
    
    meta_data_iter = pq.ParquetFile(meta_data_path).iter_batches(batch_size = batch_size)
    batch_id = 1
    for meta_data_batch_raw in tqdm(meta_data_iter):
        while True:
            try:
                if batch_id in batch_ids:
                    meta_data_batch  = meta_data_batch_raw.to_pandas()
                    meta_data_batch.drop(columns=['first_pulse_index', 'last_pulse_index'], inplace=True)
                    meta_data_batch = meta_data_batch.loc[meta_data_batch['event_id'].isin(event_ids)].reset_index(drop=True)
                            
                    if meta_data_batch.shape[0] > 0:

                        pulses = load_input(batch_id = batch_id, input_data_folder= input_data_folder, event_ids = event_ids)       
                        pulses = pulses.groupby('event_id').head(600).reset_index(drop=True)
                        
                        first_event_id_meta = meta_data_batch.iloc[0].event_id
                        last_event_id_meta = meta_data_batch.iloc[-1].event_id
                        if batch_id == 1:
                            add_to_table(database_path = database_path,
                                            df = meta_data_batch,
                                            table_name='meta_table',
                                            is_primary_key= True,
                                            engine = engine)
                        else:
                            with sqlite3.connect(database_path) as con:
                                query = f'select event_id from meta_table where event_id in ({first_event_id_meta}, {last_event_id_meta})'
                                events_meta_df = pd.read_sql(query,con)
                            if events_meta_df.shape[0] == 0:
                                add_to_table(database_path = database_path,
                                                df = meta_data_batch,
                                                table_name='meta_table',
                                                is_primary_key= True,
                                                engine = engine)

                        first_event_id_pulses = pulses.iloc[0].event_id
                        last_event_id_pulses = pulses.iloc[-1].event_id
                        if batch_id == 1:
                            add_to_table(database_path = database_path,
                                            df = pulses,
                                            table_name='pulse_table',
                                            is_primary_key= False,
                                            engine = engine)
                        else:
                            with sqlite3.connect(database_path) as con:
                                    query = f'select event_id from pulse_table where event_id in ({first_event_id_pulses}, {last_event_id_pulses})'
                                    events_pulse_df = pd.read_sql(query,con)
                            if events_pulse_df.shape[0] == 0:                                       
                                add_to_table(database_path = database_path,
                                                df = pulses,
                                                table_name='pulse_table',
                                                is_primary_key= False,
                                                engine = engine)

                        gc.collect()
                batch_id +=1
                break

            except Exception as e:
                with open('error_sql.txt', 'a') as f:
                    current_time = time.time()
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                    f.write("\ntime:" + str(formatted_time))
                    f.write('\n' + str(e) + '\n')
                    f.write('-'*50)
                    
                time.sleep(2)

    del meta_data_iter 

with open('uniform_dict.pkl', 'rb') as f:
    focus_dict = pickle.load(f)

idx = 6
event_id_list = focus_dict[f'f{idx}']
database_path = f'./data/F{idx}/focus_batch_{idx}.db'
engine = sqlalchemy.create_engine("sqlite:///" + database_path)
convert_to_sqlite(meta_data_path,
                database_path=database_path,
                input_data_folder=input_data_folder,
                batch_size=200000,
                batch_ids=list(range(1,661,1)),
                event_ids=event_id_list,
                engine=engine)

with sqlite3.connect(database_path) as con:
        query = 'select event_id from meta_table'
        events_df = pd.read_sql(query,con) 

train_selection, validate_selection = train_test_split(np.arange(0, events_df.shape[0], 1), 
                                                        shuffle=True, 
                                                        random_state = 42, 
                                                        test_size=0.02)

train_selection_events = events_df[events_df.index.isin(train_selection)]['event_id'].to_list()
validate_selection_events = events_df[events_df.index.isin(validate_selection)]['event_id'].to_list()
event_dict = {'train': train_selection_events, 'validate': validate_selection_events}
with open(f'data/F{idx}/event_dict.pkl', 'wb') as f:
    pickle.dump(event_dict, f)