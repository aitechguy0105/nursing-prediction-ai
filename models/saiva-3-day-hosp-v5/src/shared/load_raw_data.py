"""
Functions used for Training & realtime Predictions.
Fetch the data from Client SQL db and cache them in S3 while running realtime Predictions.
Fetch the data from Client SQL db and cache them in local instance.
NOTE: Trio has schema difference when compared to other clients
ie. `view_ods_physician_order_list_v2` table has `PhysiciansOrderID` rather than
`PhysicianOrderID`.
"""

import datetime
import re
import sys
from datetime import timedelta
from pathlib import Path
import typing
from urllib.parse import urlparse

import boto3
import pandas as pd
from eliot import start_action, log_message
import sqlalchemy

sys.path.insert(0, '/src')
from shared.constants import ONLY_USE_CACHE, IGNORE_CACHE
from shared.constants import S3
from shared.utils import get_client_class


def join_tables(result_dict):
    """
    :param result_dict:
    :return: Combine master_patient_lookup dataset with few other dataset
    """
    result_dict['patient_census'] = result_dict['patient_census'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )
    result_dict['patient_rehosps'] = result_dict['patient_rehosps'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )
    result_dict['patient_admissions'] = result_dict['patient_admissions'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )
    result_dict['patient_diagnosis'] = result_dict['patient_diagnosis'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )
    result_dict['patient_vitals'] = result_dict['patient_vitals'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )
#     result_dict['patient_meds'] = result_dict['patient_meds'].merge(
#         result_dict['master_patient_lookup'],
#         on=["patientid", "facilityid"]
#     )
#     result_dict['patient_orders'] = result_dict['patient_orders'].merge(
#         result_dict['master_patient_lookup'],
#         on=["patientid", "facilityid"]
#     )
#     result_dict['patient_alerts'] = result_dict['patient_alerts'].merge(
#         result_dict['master_patient_lookup'],
#         on=["patientid", "facilityid"]
#     )
    # Progress notes are not mandotary features
    if not result_dict.get('patient_progress_notes', pd.DataFrame()).empty:
        result_dict['patient_progress_notes'] = result_dict['patient_progress_notes'].merge(
            result_dict['master_patient_lookup'],
            on=["patientid", "facilityid"]
        )
    # Labs are not mandotary features
#     if not result_dict.get('patient_lab_results', pd.DataFrame()).empty:
#         result_dict['patient_lab_results'] = result_dict['patient_lab_results'].merge(
#             result_dict['master_patient_lookup'],
#             on=["patientid", "facilityid"]
#         )

    return result_dict


def validate_dataset(name, data):
    """
    :param name: name of the dataframe
    :param data: Panda dataframe
    """
    # Check whether dataset is empty
    if name not in ['patient_orders', 'patient_meds']:
        assert len(data) != 0, f'''Empty {name} table. Cant proceed further'''

def remove_invalid_datevalues(df):
    # Below code converts invalid dates values to nan values.
    for col in df.columns:
        if 'date' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def isLatestDataAvailable(data_pd, name, prediction_date, colname):
    yesterday = pd.to_datetime(prediction_date) - timedelta(days=1)
    output = False
    # Make sure there are latest data for the given prediction_date for the respective medical data.
    if name in data_pd.keys() and colname in data_pd[name]: 
        max_date = data_pd[name][colname].max()
        log_message(
            message_type="info",
            message=f"Max date for {name} data is {max_date}.",
        )
        output = max_date >= yesterday
        
    else:
        log_message(
        message_type="info",
        message=f"{name} does not exist in prediction data or {colname} column doesnot exist in {name} data.",
        )
    return output

def fetch_prediction_data(
    *,
    client_sql_engine: sqlalchemy.engine.Engine,
    prediction_date: datetime.date,
    facilityid: str,
    train_start_date: datetime.date,
    client: str,
    cache_strategy: str,
    s3_location_path_prefix: str,
    local_folder: str = None,
    save_outputs_in_local: bool = False,
    save_outputs_in_s3: bool = False,
    model_missing_datasets: typing.Optional[typing.List[str]] = [],
    facility_missing_datasets: typing.Optional[typing.List[str]] = []
):
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
    queries = getattr(clientClass(), 'get_prediction_queries')(
        prediction_date=prediction_date,
        facilityid=facilityid,
        train_start_date=train_start_date,
    )

    result_dict = {}
    for name, query in queries.items():
        if name in model_missing_datasets:
            log_message(message_type="info", name=name, skip=f"Not fetching {name} becuase the model was not trained on it")
            continue
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
                log_message(message_type="info", name=name, query=query)
                result_dict[name] = pd.read_sql(query, con=client_sql_engine)

                if name not in facility_missing_datasets:
                    # There should be data in the dataset
                    assert len(result_dict[name]) != 0, f'''{name} , Empty Dataset!'''

                result_dict[name] = remove_invalid_datevalues(result_dict[name])

                # save the query results to S3 always so we have a copy of data at prediciton time
                # if we want to rerun the prediction with another model in the future
                log_message(message_type="info", message=f"saving query results to S3", path=path)
                result_dict[name].to_parquet(path, index=False)

                # Save the actual query to S3 as well (for testing purposes)
                if save_outputs_in_s3:
                    save_string_to_s3_location(name, query, s3_location_path_prefix)
            log_message(message_type="info", name=name, dataframe_shape=len(result_dict[name]))

    assert(
        isLatestDataAvailable(result_dict, 'patient_vitals', prediction_date,'date') or 
        isLatestDataAvailable(result_dict, 'patient_progress_notes', prediction_date,'createddate')
        )
    result_dict = join_tables(result_dict)
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
    file = open(local_filename, "w")
    file.write(query_str)
    file.close()


def fetch_training_data(
    *,
    client: str,
    client_sql_engine: sqlalchemy.engine.Engine,
    train_start_date: datetime.date,
    test_end_date: datetime.date,
):
    # Dynamic import of client Class
    clientClass = get_client_class(client)
    queries = getattr(clientClass(), 'get_training_queries')(test_end_date=test_end_date, train_start_date=train_start_date)

    result_dict = {}
    # Create directory if it does not exist
    data_path = Path('/data/raw')
    data_path.mkdir(parents=True, exist_ok=True)

    for name, query in queries.items():
        with start_action(
                action_type=f'pull_{name}'
        ):
            # Execute the query using pandas
            result_dict[name] = pd.read_sql(
                query,
                con=client_sql_engine
            )
            log_message(message_type="info", total_records=len(result_dict[name]))
            # Check whether table is empty before saving in local folder
            validate_dataset(name, result_dict[name])
            result_dict[name] = remove_invalid_datevalues(result_dict[name])
            
    # Do client specific Transformation on the data
    result_dict = getattr(clientClass(), 'client_specific_transformations')(result_dict)

    result_dict = join_tables(result_dict)

    # for training purpose - we are generating parquet after join_tables function
    for key in result_dict.keys():
        path = data_path / f'{client + "_" + key}.parquet'
        result_dict[key].to_parquet(path)

    return result_dict


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


def get_genric_file_names(data_path, client):
    """
    Return all the filenames present in the cache
    :param client: client name
    :param generic: Indicates whether to return client specific names or generic names
    """

    cached_files = data_path.glob(f'{client}*')
    file_names = list()
    for gs_file in cached_files:
        t = re.findall(r'_(.*)\.parquet', gs_file.name)
        if t:
            file_names.append(t[0])
    return file_names


def fetch_training_cache_data(client, generic=False):
    """
    During training or testing used to load the cached data 
    rather than fetching from remote SQL db.
    :param client: client name
    :param generic: Indicates whether to load client specific files or generic files
    """
    result_dict = {}
    data_path = Path('/data/raw')

    for name in get_genric_file_names(data_path, client):
        if generic:
            path = data_path / f'{name}.parquet'
        else:
            path = data_path / f'{client + "_" + name}.parquet'

        result_dict[name] = pd.read_parquet(path)

    return result_dict


# ******************************* BackFill related functions *********************************

def get_backfill_patient_ids(patient_census_df, prediction_start_date, facilityid):
    return list(set(patient_census_df.query(
        '(censusdate >= @prediction_start_date) & (facilityid == @facilityid)'
    )['patientid']))

def filter_patientid(patientid_list, facilityid, df):
    return df.query('(patientid.isin(@patientid_list)) & (facilityid == @facilityid)')

def fetch_backfill_data(
    *,
    client: str,
    client_sql_engine: sqlalchemy.engine.Engine,
    train_start_date: datetime.date,
    prediction_start_date: datetime.date,
    prediction_end_date: datetime.date,
    facilityid: str,
    model_missing_datasets: typing.Optional[typing.List[str]] = [],
    facility_missing_datasets: typing.Optional[typing.List[str]] = []
):    
    # Dynamic import of client Class
    clientClass = get_client_class(client)
    clientClass = get_client_class(client)
    queries = getattr(clientClass(), 'get_training_queries')(
        train_start_date=train_start_date,
        test_end_date=prediction_end_date,
    )

    result_dict = {}
    # Create directory if it does not exist

    for name, query in queries.items():
        with start_action(
                action_type=f'pull_{name}'
        ):
            # Execute the query using pandas
            result_dict[name] = pd.read_sql(
                query,
                con=client_sql_engine
            )
            log_message(message_type="info", total_records=len(result_dict[name]))


    result_dict = join_tables(result_dict)
    
    # Get all the patientID's who have census_date between prediction_start_date and prediction_end_date 
    patientid_list = get_backfill_patient_ids(result_dict['patient_census'], prediction_start_date, facilityid)
    
    for key in result_dict.keys():
        if key != 'patient_demographics':
            result_dict[key] = filter_patientid(patientid_list, facilityid, result_dict[key])
        if key not in facility_missing_datasets:
            # There should be data in the dataset
            assert len(result_dict[key]) != 0, f'''{key} , Empty Dataset!'''

    return result_dict
