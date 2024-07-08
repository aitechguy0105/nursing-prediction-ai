import re
import sys
from collections import defaultdict
from importlib import import_module

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')


def pascal_case(chars):
    word_regex_pattern = re.compile('[^A-Za-z]+')
    words = word_regex_pattern.split(chars)
    return ''.join(w.title() for i, w in enumerate(words))


def get_client_class(client):
    module = import_module(f'clients.{client}')
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
