import re
import sys
import os
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import pandas as pd
from eliot import log_message
from omegaconf import OmegaConf

from clients.base import Base

sys.path.insert(0, '/src')


def pascal_case(chars):
    word_regex_pattern = re.compile('[^A-Za-z]+')
    words = word_regex_pattern.split(chars)
    return ''.join(w.title() for i, w in enumerate(words))


def get_client_class(client):
    try:
        module = import_module(f'clients.{client}')
    except ModuleNotFoundError:
        log_message(f'Client specific file not found for {client}')
        return Base
    return getattr(module, pascal_case(client))


def get_memory_usage(df):
    BYTES_TO_MB_DIV = 0.000001
    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3)
    return (str(mem) + ' MB')


def print_dtypes(df):
    # Get all different datatypes used and their column count
    result = defaultdict(lambda: [])
    for col in df.columns:
        result[df[col].dtype].append(col)
    print(dict(result))


def clean_multi_columns(cols):
    """
    input: list
    output: list
    desc: 1. Join sub-column names into a single column structure
            after pivoting dataframes
    """
    new_cols = []
    for col in cols:
        if col[1] == '':
            new_cols.append(col[0])
        else:
            new_cols.append('_'.join(col))
    return new_cols


def clean_feature_names(cols):
    """
    - Used on alertdescription values when converted to feature names
    - Remove special characters from feature names
    - replace space with underscore
    - LGBM does not support special non ascii characters
    """
    new_cols = []
    for col in cols:
        col = col.strip()
        col = re.sub('[^A-Za-z0-9_ ]+', '', col)
        new_cols.append(col.replace(" ", "_"))
    return new_cols


def load_config(path):
    path = Path(path)
    assert Path.exists(path/'defaults.yaml'), f"Default configuration file doesn't exist in {path}"
    conf = OmegaConf.load(path/'defaults.yaml')
    if Path.exists(path/'generated/'):
        generated_files = [fname for fname in os.listdir(path/'generated/') if fname.endswith('yaml')]
        for fname in generated_files:
            conf = OmegaConf.merge(conf, OmegaConf.load(path/'generated'/fname))
    return conf
