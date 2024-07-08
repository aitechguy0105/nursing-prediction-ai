import typing
import re

from collections import defaultdict
from eliot import log_message
from importlib import import_module
import urllib.parse

from saiva.model.clients.base import Base


def url_decode_cols(cols):
    decoded_cols = []
    for col in cols:
        decoded_cols.append(urllib.parse.unquote(col))

    return decoded_cols


def url_encode_cols(df):
    """ Latest version of LGBM library does not allow feature names 
    to have special characters, hence doing a URL-encode on all features names.
    """
    new_cols = []
    for col in df.columns:
        new_cols.append(urllib.parse.quote(col))
    df.columns = new_cols
    return df


def pascal_case(chars):
    word_regex_pattern = re.compile('[^A-Za-z0-9]+')
    words = word_regex_pattern.split(chars)
    return ''.join(w.title() for i, w in enumerate(words))


def get_client_class(client) -> typing.Type[Base]:
    try:
        module = import_module(f'saiva.model.clients.{client}')
        return getattr(module, pascal_case(client))
    except ModuleNotFoundError:
        log_message(
            message_type='info',
            message=f'Client specific file not found for {client}'
        )
        return Base


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
