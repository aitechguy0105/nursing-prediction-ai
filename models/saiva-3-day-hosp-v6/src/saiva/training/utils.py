import numpy as np
import pandas as pd
import os
import pickle
import mlflow
from pathlib import Path
import subprocess
from omegaconf import OmegaConf
import sys


def get_facilities_from_train_data(df):
    return list(df.facilityid.unique())

def get_date_diff(start_date, end_date):
    diff = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    return f'{diff}'

def load_x_y_idens(processed_path, model_type=None, data_split=None, x_filename=None, y_filename=None, idens_filename=None):
    
    """ Load x, y and idens dataframes from the local disk, the filenames can be passed explicitly through
    `x_filepath`, `y_filepath` and `idens_filepath` arguments or thtough `model_type` ('model_upt', 'model_fall') and
    `data_split` ('train', 'test', 'valid')
    
    Parameters:
    -----------
    processed_path : str or pathlib.Path
        Path to the location where the files are stored.
    model_type : str or None
        Common values are 'model_upt', 'model_fall', can be any string.
        If None, filepaths should be set explicitly.
    data_split : str
        Data split type, e.g. 'train', 'valid', 'test'
        If None, filepaths should be set explicitly.
    x_filepath, y_filepath, idens_filepath : str or None
        If string, should link to the file in `processed_path`
        If None, file name to load will be generated using `model_type` and `data_split`
        
    Returns:
    --------
    x : Any, usually pandas.DataFrame
    y : Any, usually numpy.ndarray
    idens : Any, usually pandas.DataFrame
        Data to train the model
    
    """
    
    def generate_path(filename, processed_path, model_type, data_split, table):
        if filename is None:
            return processed_path/f'final-{data_split}_{table}_{model_type}.pickle'
        else:
            return processed_path/filename
    
    if (model_type is None or data_split is None) and (x_filepath is None or y_filepath is None or idens_filepath is None):
        raise ValueError("Either both `model_type` and `data_split` should be passed or all `x_filepath`, `y_filepath`, `idens_filepath`")
        
    processed_path = Path(processed_path)                     
    x_filepath = generate_path(x_filename, processed_path, model_type, data_split, 'x')
    y_filepath = generate_path(y_filename, processed_path, model_type, data_split, 'target_3_day')
    idens_filepath = generate_path(idens_filename, processed_path, model_type, data_split, 'idens')
    
    with open(x_filepath,'rb') as f: x = pickle.load(f)
    with open(y_filepath,'rb') as f: y = pickle.load(f)
    with open(idens_filepath,'rb') as f: idens = pickle.load(f)
        
    return x, y, idens





def load_lgb_model(modelid=None, model_path=None):
    """ Load a pickled model wrap and extract LightGBM model from there
    
    Parameters:
    -----------
    modelid : str
        Model's uuid
    model_path : str or pathlib.Path or None
        The location of the model. If not specified explicitly the model with the passed `modelid`
        from src/notebooks will be loaded
        
    Returns:
    --------
    model : lightgbm.Booster or any other model type.
        
    """
    from .core import BaseModel
    if model_path is None:
        model_path = Path(f'/src/saiva/notebooks/{modelid}.pickle')
    else:
        model_path = Path(model_path)
    with open(model_path, 'rb') as f:
        try:
            model = pickle.load(f).model
        except ModuleNotFoundError:
            import saiva.training as training_module

            sys.modules['training'] = training_module
            model = pickle.load(f).model
            
    return model

def download_model_from_mlflow(modelid, local_path='/src/saiva/notebooks'):
    """ Find the model by `modelid` in MLflow, load and save it in `local_path`
    """
    mlflow.set_tracking_uri('http://mlflow.saiva-dev')
    local_path = '/src/saiva/notebooks'
    return mlflow.artifacts.download_artifacts(run_id=modelid, artifact_path=f'{modelid}.pickle', dst_path=local_path)

def load_config(path):
    path = Path(path)
    assert Path.exists(path/'defaults.yaml'), f"Default configuration file doesn't exist in {path}"
    conf = OmegaConf.load(path/'defaults.yaml')
    if Path.exists(path/'generated/'):
        generated_files = [fname for fname in os.listdir(path/'generated/') if fname.endswith('yaml')]
        for fname in generated_files:
            conf = OmegaConf.merge(conf, OmegaConf.load(path/'generated'/fname))
    return conf