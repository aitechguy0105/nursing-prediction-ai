"""
Functions used for Training & realtime Predictions.
Fetch the data from Client SQL db and cache them in S3 while running realtime Predictions.
Fetch the data from Client SQL db and cache them in local instance.
NOTE: Trio has schema difference when compared to other clients
ie. `view_ods_physician_order_list_v2` table has `PhysiciansOrderID` rather than
`PhysicianOrderID`.
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
from eliot import start_action, log_message

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
    result_dict['patient_admissions'] = result_dict['patient_admissions'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )
    result_dict['patient_diagnosis'] = result_dict['patient_diagnosis'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )
    result_dict['patient_alerts'] = result_dict['patient_alerts'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )
    result_dict['patient_rehosps'] = result_dict['patient_rehosps'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )

    return result_dict


def fetch_prediction_data(client, client_sql_engine, start_date, end_date, facilityid):
    """
    Used only while predicting
    :param client_sql_engine: SQL engine connected to client db
    :param prediction_date:
    :param facilityid:
    :param train_start_date:
    :param client:
    :return: Client SQLDB dataset which needs prediction for the given date
    """
    # Dynamic import of client Class
    clientClass = get_client_class(client)
    queries = getattr(clientClass(), 'get_prediction_queries')(start_date, end_date, facilityid)

    result_dict = {}
    # path is where the output of the sql query is stored
    data_path = Path('/data/raw')
    data_path.mkdir(parents=True, exist_ok=True)
    
    for name, query in queries.items():
        with start_action(action_type=f'pull_{name}', facilityid=facilityid):

            # Execute the query using pandas
            log_message(message_type="info", name=name, query=query)
            result_dict[name] = pd.read_sql(query, con=client_sql_engine)
            
            # call client specific validate function that makes sure data was present
            getattr(clientClass(), 'validate_dataset')(facilityid, name, len(result_dict[name]))

            log_message(message_type="info", name=name, dataframe_shape=len(result_dict[name]))

    result_dict = join_tables(result_dict)
    # for training purpose - we are generating parquet after join_tables function
    for key in result_dict.keys():
        path = data_path / f'{key}.parquet'
        result_dict[key].to_parquet(path)

    return result_dict


def fetch_cache_data(client, generic=False):
    """
    During training or testing used to load the cached data 
    rather than fetching from remote SQL db.
    :param client: client name
    :param generic: Indicates whether to load client specific files or generic files
    """
    result_dict = {}
    data_path = Path('/data/raw')

    for name in ['patient_admissions','patient_rehosps','patient_alerts','patient_diagnosis','patient_census','master_patient_lookup']:
        path = data_path / f'{name}.parquet'

        result_dict[name] = pd.read_parquet(path)

    return result_dict
