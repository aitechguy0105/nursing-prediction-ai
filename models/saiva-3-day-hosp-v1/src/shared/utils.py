import re
import sys
from collections import defaultdict
from importlib import import_module

import pandas as pd

sys.path.insert(0, '/src')


def pascal_case(chars):
    word_regex_pattern = re.compile('[^A-Za-z]+')
    words = word_regex_pattern.split(chars)
    return ''.join(w.title() for i, w in enumerate(words))


def get_client_class(client):
    module = import_module(f'clients.{client}')
    return getattr(module, pascal_case(client))


def downcast_dtype(df):
    # convert float64 and int64 32 bit verison to save on memory
    df.loc[:, df.select_dtypes(include=['int64']).columns] = df.select_dtypes(
        include=['int64']).apply(
        pd.to_numeric, downcast='unsigned'
    )
    # Convert all float64 columns to float32
    df.loc[:, df.select_dtypes(include=['float64']).columns] = df.select_dtypes(
        include=['float64']
    ).astype('float32')

    return df


def convert_to_int16(df):
    """
    Convert entire dataframe to int16 columns
    """
    for col in df:
        df[col] = df[col].astype('int16')
    return df


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
