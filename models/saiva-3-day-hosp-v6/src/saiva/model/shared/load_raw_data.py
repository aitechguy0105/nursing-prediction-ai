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
from datetime import timedelta
from pathlib import Path
import typing
from urllib.parse import urlparse
import time
import boto3
import pandas as pd
from eliot import start_action, log_message
import sqlalchemy
from omegaconf import OmegaConf

from .constants import ONLY_USE_CACHE, IGNORE_CACHE, INVALID_ACTIONTYPE, REQUIRED_DATAFRAMES
from .constants import S3
from .utils import get_client_class


def join_tables(result_dict):
    """
    :param result_dict:
    :return: Combine master_patient_lookup dataset with few other dataset
    """
    for table_name in result_dict.keys():
        if "patientid" in result_dict[table_name].columns:
            result_dict[table_name].dropna(subset=["patientid"], axis=0, inplace=True)

    result_dict['patient_census'] = result_dict['patient_census'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid"]
    )

    exclude_tables = {'patient_census', 'master_patient_lookup', 'patient_demographics'}
    for table_name in result_dict.keys():
        if table_name not in exclude_tables:
            log_message(
                message_type="info",
                message=f"merging {table_name} with master_patient_lookup"
            )
            if not result_dict.get(table_name, pd.DataFrame()).empty:
                result_dict[table_name] = result_dict[table_name].merge(
                    result_dict['master_patient_lookup'][['facilityid', 'patientid', 'masterpatientid']],
                    on=["patientid", "facilityid"]
                    )

    return result_dict


def validate_dataset(name, data):
    """
    :param name: name of the dataframe
    :param data: Panda dataframe
    """
    # Check whether dataset is empty
    if len(data) == 0:
        if name in REQUIRED_DATAFRAMES:
            raise Exception(f'''Empty {name} table. Cant proceed further''')
        else:
            log_message(
                message_type="warning",
                message=f'''Empty {name} table. Will proceed ignoring this table'''
            )


def postprocess_data(df, name):
    # Below code converts invalid dates values to nan values.
    for col in df.columns:
        if 'date' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if name == 'patient_vitals':
        # get all collumns except warnings
        cols = [col for col in df.columns if 'warning' not in col]

        # concatinate warnings for each unique row
        def null_safe_join(group):
            return None if group.isnull().any() else ''.join(group)

        df = df.groupby(cols, dropna=False)['warnings'].apply(null_safe_join).reset_index()
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
            message_type="warning",
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
    config: typing.Optional[OmegaConf] = None,
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

    configured_facilities = None

    if config:
        if config.client_configuration.multiple_clients:
            configured_facilities = config.client_configuration.facilities[re.sub('_*v6.*', '', client)]
        else:
            configured_facilities = config.client_configuration.facilities

    # Dynamic import of client Class
    client_obj = get_client_class(client)(
        engine=client_sql_engine,
        facilities=configured_facilities,
    )
    queries = getattr(client_obj, 'get_prediction_queries')(
        prediction_date=prediction_date,
        facilityid=facilityid,
        train_start_date=train_start_date,
    )

    result_dict = {}
    for name, query in queries.items():
        if query is None:
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
                start_time = time.time()
                result_dict[name] = pd.read_sql(query, con=client_sql_engine)
                end_time = time.time()
                result_dict[name] = postprocess_data(df=result_dict[name], name=name)

                # save the query results to S3 always so we have a copy of data at prediciton time
                # if we want to rerun the prediction with another model in the future
                log_message(
                    message_type="info",
                    message="saving query results to S3",
                    path=path,
                    time_taken=round(end_time - start_time, 2)
                )
                result_dict[name].to_parquet(path, index=False)

                # Save the actual query to S3 as well (for testing purposes)
                if save_outputs_in_s3:
                    save_string_to_s3_location(name, query, s3_location_path_prefix)
            log_message(message_type="info", name=name, dataframe_shape=result_dict[name].shape)

    assert (
        isLatestDataAvailable(result_dict, 'patient_vitals', prediction_date, 'date') or
        isLatestDataAvailable(result_dict, 'patient_progress_notes', prediction_date, 'createddate')
    )

    if 'begineffectivedate' in result_dict['patient_census']:
        result_dict['patient_census'] = unroll_patient_census(
            result_dict['patient_census'],
            train_start_date,
            prediction_date,
            training=False
        )

    result_dict['patient_census'] = result_dict['patient_census'].merge(
        result_dict['master_patient_lookup'],
        on=["patientid", "facilityid", "masterpatientid"]
    )

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
    conditional_census: bool = False,
):
    # Dynamic import of client Class
    client_obj = get_client_class(client)(
        engine=client_sql_engine,
    )
    queries = client_obj.get_training_queries(
        test_end_date=test_end_date,
        train_start_date=train_start_date,
    )

    result_dict = {}
    # Create directory if it does not exist
    data_path = Path('/data/raw')
    data_path.mkdir(parents=True, exist_ok=True)

    for name, query in queries.items():
        if query is None:
            continue

        with start_action(
                action_type=f'pull_{name}'
        ):
            # Execute the query using pandas
            result_dict[name] = pd.read_sql(
                query,
                con=client_sql_engine
            )
            log_message(
                message_type="info", 
                message=f'fetching training data for - {name}', 
                total_records=result_dict[name].shape
            )
            # Check whether table is empty before saving in local folder
            validate_dataset(name, result_dict[name])
            
            result_dict[name] = postprocess_data(result_dict[name], name)
            
    if 'begineffectivedate' in result_dict['patient_census']:
        log_message(
                message_type="info", 
                message=f'unrolling patient census', 
            )
        result_dict['patient_census'] = unroll_patient_census(
            result_dict['patient_census'],
            train_start_date,
            test_end_date,
            training=True,
            conditional_census=conditional_census
        )


    # Do client specific Transformation on the data
    result_dict = client_obj.client_specific_transformations(result_dict)

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
    if cache_strategy == IGNORE_CACHE:
        return None
    else:
        # cache_strategy is either ONLY_USE_CACHE or CHECK_CACHE_FIRST
        if S3.exists(path):
            df = pd.read_parquet(path)
            log_message(message_type='info',
                        message='Found existing s3 file for this day - using that instead of pulling new data',
                        path=path, dataframe_shape = df.shape)
            return df
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
        log_message(
                message_type='info', 
                message=f'fetching training cache data.', 
                dataframe_shape = result_dict[name].shape, 
        )
    return result_dict

def get_last_operational_month(census):
    """ Counts number of the census records (do not confuse with patient days)
        and identifies the last operational month for each facility as the last one
        when there were more than 3 census records.
    """
    census = census.copy()
    census['year'] = census['begineffectivedate'].dt.year
    census['month'] = census['begineffectivedate'].dt.month
    counts = census.groupby(['facilityid', 'year', 'month'])['begineffectivedate'].count().reset_index()
    last_seen = counts.groupby('facilityid').apply(lambda x: x.nlargest(1, ['year', 'month'])).reset_index(drop=True)
    last_seen['date'] = last_seen.apply(lambda row: pd.Timestamp(row['year'], row['month'], 1), axis=1)
    return last_seen.set_index('facilityid')['date'].to_dict()

def cutoff_dead_facilities(census, last_date):
    
    # The last month of the give dataset
    last_date = pd.to_datetime(last_date)
    last_month = last_date.replace(day=1)
    
    # When the facility last time was alive
    last_seen = get_last_operational_month(census)

    # Cutting off the census records after the facility is considered dead
    census['cutoff_date'] = census['facilityid'].map(last_seen)
    census.loc[census['cutoff_date']==last_month, 'cutoff_date'] = last_date + pd.DateOffset(days=1)

    # This is an exception for the case when the last date of the dataset is the 1st, 2nd or 3rd of a month
    # and the last qualified month is the previous one. In this case we believe the facility is still alive
    # till the last date of the facility, just doesn't have enough records to be qualified as alive
    # in the last month since the month just started.
    if last_date.day <= 3:
        census.loc[census['cutoff_date']+pd.DateOffset(months=1)==last_month, 'cutoff_date'] = last_date + pd.DateOffset(days=1)

    census = census.loc[census['begineffectivedate']<census['cutoff_date']]
    census['endeffectivedate'].fillna(census['cutoff_date'], inplace=True)
    census['endeffectivedate'].clip(upper=census['cutoff_date'], inplace=True)

    census.drop(['cutoff_date'], axis=1, inplace=True)

    return census

def unroll_patient_census(census, first_date, last_date, training, invalid_action_types=None, conditional_census=False, experiment_dates_facility_wise_overrides=None):
    """
    The function is indroduced as a part of switching from 4.10.7. view_ods_daily_census_v2
    to 4.10.3. view_ods_patient_census (see https://saivahc.atlassian.net/wiki/spaces/MDS/pages/1748369419/The+different+ways+to+create+almost+the+same+list+of+the+patient-days+to+be+ranked)

    Parameters:
    -----------
        census : pandas.DataFrame
            Raw census data fetched from the 4.10.3 view_ods_patient_census

        first_date : datetime.datetime or str in format 'YYYY-MM-DD'
            The first date that has to be included into the census, usually `train_start_date`

        last_date : datetime.datetime or str in format 'YYYY-MM-DD'
            The last date that has to be included into the census, usually `test_end_date` or
            `prediction_date`

        training :
            If True (i.e. training) we also cut-off the dead facilities
            If False (i.e. prediction), we believe the facility we are making prediction for is active no matter what

    Return:
    -------
        census : pandas.DataFrame
            Unrolled census where every line represents one patient-day

    """
    if training:
        census = cutoff_dead_facilities(census, last_date)

    if invalid_action_types is None:
        invalid_action_types = INVALID_ACTIONTYPE

    # We don't want to include the endeffectivedate into the range:
    census['endeffectivedate'] -= pd.DateOffset(1)

    # If the endefeectivedate is null (action is still effective)
    # fill it with the last date:
    census['endeffectivedate'].fillna(last_date, inplace=True) 
    census.sort_values(['facilityid','patientid', 'begineffectivedate'], inplace=True)

    # set createddate to createddate if exists else None
    census['createddate'] = census.get('createddate', None)

    # cast createddate as date type
    census['createddate'] = pd.to_datetime(census['createddate']).dt.normalize()

    # store temp_createddate as copy of current createddate
    census['temp_createddate'] = census['createddate'].copy()

    # Replace the Internal Transfer actiontype and createddate by preceding actiontype, createddate (if any)
    census.loc[census.actiontype=='Internal Transfer', ['actiontype', 'createddate']] = None

    # ffill actiontypes 
    census['actiontype'] = census.groupby(
        ['facilityid', 'patientid']
    )['actiontype'].fillna(method='ffill').fillna('Internal Transfer')

    # ffill createddate. If createddate is null, use temp_createddate.
    census['createddate'] = census.groupby(
        ['facilityid', 'patientid']
    )['createddate'].fillna(method='ffill').fillna(census['temp_createddate'])

    census = census.loc[~census['actiontype'].isin(invalid_action_types)]

    # Create the range (kind of list) of days from begineffectivedate to endeffectivedate
    census['censusdate'] = census.apply(
        lambda row: pd.date_range(
            row.begineffectivedate.date(),
            row.endeffectivedate.date(),
            freq='D'
        ),
        axis=1
    )
    census = census.explode('censusdate') # Each date in that range now has its own row 

    # when conditional_census is True, exclude patient days prior to createddate of admission or return from leave record
    if conditional_census:
        # drop rows where censusdate < createddate of the most recent admission or return from leave record
        census = census.loc[census['censusdate']>=census['createddate']]

    # drop unwanted columns
    census.drop(columns=['actiontype', 'begineffectivedate', 'endeffectivedate', 'createddate', 'temp_createddate'], inplace=True)

    # Finally drop the dates out of our range
    # since SQL query allows some of them to be presented
    experiment_dates_facility_wise_overrides = (experiment_dates_facility_wise_overrides or {})
    facility_wise_start_dates = {facilityid: experiment_dates.train_start_date for facilityid, experiment_dates in experiment_dates_facility_wise_overrides.items()}
    facility_wise_end_dates = {facilityid: experiment_dates.test_end_date for facilityid, experiment_dates in experiment_dates_facility_wise_overrides.items()}

    census['train_start_date'] = census['facilityid'].map(facility_wise_start_dates)
    census['test_end_date'] = census['facilityid'].map(facility_wise_end_dates)

    census['train_start_date'].fillna(first_date, inplace=True)
    census['test_end_date'].fillna(last_date, inplace=True)

    census['train_start_date'] = pd.to_datetime(census['train_start_date'])
    census['test_end_date'] = pd.to_datetime(census['test_end_date'])

    census = census.loc[
        (census.censusdate >= census.train_start_date) &
        (census.censusdate <= census.test_end_date)
    ]

    census.drop(columns=['train_start_date', 'test_end_date'], inplace=True)

    return census


# ******************************* BackFill related functions *********************************

def get_backfill_patient_ids(patient_census_df, prediction_start_date, facilityid):
    return list(set(patient_census_df.query(
        '(censusdate >= @prediction_start_date) & (facilityid == @facilityid)'
    )['patientid']))

def filter_patientid(patientid_list, facilityid, df):
    return df.query('(patientid.isin(@patientid_list)) & (facilityid == @facilityid)')

def fetch_backfill_data(client, client_sql_engine, train_start_date, prediction_start_date, prediction_end_date, facilityid, config=None):
    
    configured_facilities = None

    if config:
        if config.client_configuration.multiple_clients:
            configured_facilities = config.client_configuration.facilities[client]
        else:
            configured_facilities = config.client_configuration.facilities
    
    # Dynamic import of client Class
    clientClass = get_client_class(client)
    queries = getattr(clientClass(
        facilities=configured_facilities
    ), 'get_training_queries')(test_end_date=prediction_end_date, train_start_date=train_start_date)

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
            log_message(
                message_type="info", 
                message= f'backfilling data for - {name}', 
                total_records=result_dict[name].shape
            )


    result_dict = join_tables(result_dict)
    
    # Get all the patientID's who have census_date between prediction_start_date and prediction_end_date 
    patientid_list = get_backfill_patient_ids(result_dict['patient_census'], prediction_start_date, facilityid)
    
    for key in result_dict.keys():
        if key != 'patient_demographics':
            result_dict[key] = filter_patientid(patientid_list, facilityid, result_dict[key])
        # call client specific validate function that makes sure data was present
        getattr(clientClass(), 'validate_dataset')(facilityid, key, result_dict[key].shape)

    return result_dict
