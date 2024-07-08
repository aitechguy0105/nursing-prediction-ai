"""
Functions used for Training & realtime Predictions.
Fetch the data from Client SQL db and cache them in S3 while running realtime Predictions.
Fetch the data from Client SQL db and cache them in local instance.
NOTE: Trio has schema difference when compared to other clients
ie. `view_ods_physician_order_list_v2` table has `PhysiciansOrderID` rather than
`PhysicianOrderID`.
"""

import os
import sys
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
from eliot import start_action, log_message

sys.path.insert(0, '/src')
from shared.utils import get_client_class
from shared.database import DbEngine
from shared.patient_stays import get_patient_stays
from shared.constants import ONLY_USE_CACHE, IGNORE_CACHE
from shared.constants import S3


def validate_dataset(name, data):
    """
    :param name: name of the dataframe
    :param data: Panda dataframe
    """
    # Check whether dataset is empty
    assert len(data) != 0, f'''Empty {name} table. Cant proceed further'''


def validate_vitals(vitals_pd, prediction_date):
    yesterday = pd.to_datetime(prediction_date) - timedelta(days=1)
    max_date = vitals_pd['date'].max()
    log_message(
        message_type='info',
        message=f'Max date for Vitals data is {max_date}.',
    )
    # Make sure there are latest vitals data for the given prediction_date
    assert max_date >= yesterday


def check_file_exists(path, cache_strategy):
    """
    :param path:
    :return:
    """
    # log_message(message_type='info', path=path, cache_strategy=cache_strategy)
    if cache_strategy == IGNORE_CACHE:
        return None
    else:
        # cache_strategy is either ONLY_USE_CACHE or CHECK_CACHE_FIRST
        if S3.exists(path):
            log_message(message_type='info',
                        message='Found existing s3 file for this day - using that instead of pulling new data',
                        path=path)
            return pd.read_parquet(path)
        elif cache_strategy == ONLY_USE_CACHE:
            assert False, f'Abort: cache_strategy is {cache_strategy}, but cache does not exist: {path}'
        else:
            return None


def fetch_prediction_data(client_sql_engine, prediction_date, facilityid, train_start_date, client,
                          cache_strategy, s3_location_path_prefix,
                          local_folder=None, save_outputs_in_local=False):
    """
    Used only while predicting
    :param client_sql_engine: SQL engine connected to client db
    :param prediction_date:
    :param facilityid:
    :param train_start_date:
    :param client:
    :param cache_strategy:
    :param s3_location_path_prefix: s3 prefix of where the generated parquet file (query results) would be stored
    :param local_folder: location of where to save sql query text
    :param save_outputs_in_local: should you save sql query text in local dir or not
    :return: Client SQLDB dataset which needs prediction for the given date
    """
    # Dynamic import of client Class
    clientClass = get_client_class(client)
    queries = getattr(clientClass(), 'get_prediction_queries')(prediction_date, facilityid, train_start_date)

    result_dict = {}
    for name, query in queries.items():
        # path is where the output of the sql query is stored
        path = f'{s3_location_path_prefix}/{name}.parquet'

        if save_outputs_in_local and (local_folder is not None):
            # save the actual text of the sql query (for testing purposes)
            save_string_to_local_file(name, query, local_folder)

        with start_action(action_type=f'pull_{name}', facilityid=facilityid):
            # Check whether the file is already present in the S3 folder. If found use the same file
            result_dict[name] = check_file_exists(path, cache_strategy)

            # If file not found in S3
            if result_dict[name] is None:
                # Execute the query using pandas
                log_message(message_type='info', name=name, query=query)
                result_dict[name] = pd.read_sql(query, con=client_sql_engine)

                assert len(result_dict[name]) != 0, f'''{name} , Empty Dictionary!'''

                if name == 'patient_vitals':
                    validate_vitals(result_dict[name], prediction_date)

                # Save the query results to S3
                log_message(message_type='info', message=f'saving query results to S3', path=path)
                result_dict[name].to_parquet(path, index=False)

                # Save the actual query to S3 as well (for testing purposes)
                save_string_to_s3_location(name, query, s3_location_path_prefix)
            log_message(message_type='info', name=name, dataframe_shape=len(result_dict[name]))

    return result_dict


def save_string_to_s3_location(query_name, query_str, s3_location_path_prefix):
    s3_client = boto3.client('s3')
    # s3_query_loc is where the actual text of the sql query is stored (for testing purposes)
    s3_query_loc = f'{s3_location_path_prefix}/{query_name}.txt'
    # get the bucket and key from the path
    o = urlparse(s3_query_loc)
    bucket = o.netloc
    key = o.path.lstrip('/')
    s3_client.put_object(Body=query_str, Bucket=bucket, Key=key)


def save_string_to_local_file(query_name, query_str, local_folder):
    local_filename = f'{local_folder}/{query_name}.txt'
    file = open(local_filename, 'w')
    file.write(query_str)
    file.close()


def fetch_training_data(client_list, train_start_date, train_end_date):
    """
    :param client_list: client names in a list
    :param train_start_date:
    :param test_end_date:
    When multiple client names are passed, we combine the data
    """
    result_dict = {}
    # Create directory if it does not exist
    data_path = Path('/data/raw')
    data_path.mkdir(parents=True, exist_ok=True)
    engine = DbEngine()

    for client in client_list:
        # Dynamic import of client Class
        clientClass = get_client_class(client)
        queries = getattr(clientClass(), 'get_training_queries')(train_start_date, train_end_date)
        client_sql_engine = engine.get_sqldb_engine(clientdb_name=client)

        for name, query in queries.items():
            with start_action(
                    action_type=f'pull_{name}'
            ):
                # Execute the query using pandas
                data = pd.read_sql(
                    query,
                    con=client_sql_engine
                )
                # Add client column & add client name into masterpatientid
                data['masterpatientid'] = data['masterpatientid'].apply(
                    lambda x: client + '_' + str(x)
                )
                data['client'] = client
                # First time check
                if name not in result_dict:
                    result_dict[name] = pd.DataFrame()

                result_dict[name] = result_dict[name].append(data, ignore_index=True)

                log_message(message_type='info', client=client, total_records=len(result_dict[name]))
                # Check whether table is empty before saving in local folder
                validate_dataset(name, result_dict[name])

    # Calculate the stay duration
    result_dict['stays'] = get_patient_stays(
        result_dict['patient_census'],
        result_dict['patient_rehosps']
    )
    # for training purpose - we are generating parquet after join_tables function
    for key in result_dict.keys():
        path = data_path / f'{key}.parquet'
        result_dict[key].to_parquet(path)

    return result_dict


def load_raw_data_from_files(dirname, prefix='', categories=None):
    if categories is None:
        categories = [
            'patient_demographics',
            'patient_census',
            'patient_rehosps',
            'patient_diagnosis',
            'patient_meds',
            'patient_vitals',
            'patient_progress_notes',
            'patient_lab_results',
            'patient_orders',
            'patient_alerts',
            'stays',
        ]
    data_dict = {}
    for category_name in categories:
        if prefix is not '':
            filename = f'{prefix}{category_name}.parquet'
        else:
            filename = f'{category_name}.parquet'
        file_path = Path(os.path.join(dirname, filename))

        if file_path.is_file():
            print(f'Loading file {file_path}')
            # data_dict[category_name] = pd.read_parquet(os.path.join(dirname, filename))
            data_dict[category_name] = pd.read_parquet(file_path)
        else:
            print(f'Skipping {file_path} - no such file')

    return data_dict
