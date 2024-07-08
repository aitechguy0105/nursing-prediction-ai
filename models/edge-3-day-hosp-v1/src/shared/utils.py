import boto3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import os 
import sys
import datetime

from collections import defaultdict, namedtuple 
from multiprocessing import Pool 
from scipy.sparse import csr_matrix

from collections import defaultdict
from importlib import import_module

sys.path.insert(0, '/src')


def _getpredictiontimestampsForStayWorker(job_data):
    """
    For each stay duration row, create multiple rows for each date in duration range
    """
    stay_row = job_data[0]
    stay = job_data[1]
    time_delta = job_data[2]

    # Get date range bw start & end
    if not pd.isna(stay.dateoftransfer): 
        dates = pd.date_range(start=stay.startdate, end=stay.dateoftransfer)
    else: 
        dates = pd.date_range(start=stay.startdate, end=stay.enddate)

    if len(dates) <= 1: 
        # Ignore anything with only 1 day
        return None
    else: 
        # Otherwise, we always drop first day since census is just before midnight... 
        dates = dates[1:]

    # Construct prediction timestamps
    prediction_timestamps = [date + time_delta for date in dates]

    # Drop prediction on transfer day if time of transfer is before prediction time
    if not pd.isna(stay.dateoftransfer) and prediction_timestamps[-1] >= stay.dateoftransfer: 
        prediction_timestamps = prediction_timestamps[:-1]
    
    N = len(prediction_timestamps)    
    pids = [stay.masterpatientid]*N
    fids = [stay.facilityid]*N 
    stay_rows = [stay_row]*N

    retval = pd.DataFrame({'masterpatientid': pids, 
                           'facilityid': fids, 
                           'predictiontimestamp': prediction_timestamps,
                           'stayrowindex': stay_rows})
    return retval

# def _assignStayToPredictionWorker(job_data):
#     # Get prediction times matching this stay...
#     # job_data has 1 stay row (startdate, enddate, index)
#     # job_data has prediction time rows matched to pid, fid of stay
#     stay_idx = job_data[0]
#     stay_start_date = job_data[1]
#     stay_end_date = job_data[2]
#     predict_times = job_data[3]
#     mask = (predict_time.predictiondate >= stay_start_date).values & \
#            (predict_time.predictiondate < stay_end_date).values
#     predict_time_subset = predict_time_rows[mask]
#     retval = (predict_time_subset.index.values, stay_idx)
#     return retval
#
# def get_stay_for_prediction_times(predict_times, stays):
#     # Map predict time rows to stay rows, with np.nan for those that
#     # are unmappable.
#
#     print('Grouping predict_times...')
#     predict_times_by_pid_fid = predict_times.copy().groupby(['masterpatientid', 'facilityid'])
#
#     # Make jobs data list
#     print("Preparing jobs data")
#     jobs_data = []
#     for idx, stay in stays.iterrows():
#         if idx % 10000 == 0: print(f'Working on {idx}...')
#         predict_times_indices = predict_times_by_pid_fid.groups[(stay.masterpatientid, stay.facilityid)]
#         if len(predict_times_indices) == 0: continue
#         predict_times_rows = predict_times.loc[predict_times_indices]
#         job_data = (idx, stay.startdate, stay.enddate, predict_times_rows)
#         jobs_data.append(job_data)
#
#     # Next, set up pool and run jobs.
#     print("Running jobs...")
#     with Pool(min(os.cpu_count() - 4, 32)) as pool:
#         stay_assignments = pool.map(_assignStayToPredictionWorker, jobs_data)
#
#     # Get assignments and return...
#     print("Constructing stay assignments...")
#     # Initialize to np.nan:
#     stay_indices = np.zeros(len(predict_times))
#     stay_indices[:] = np.nan
#
#     # Iterate through stay assignments and assign...
#     for stay_assignment in stay_assignments:
#         stay_indices[ stay_assignment[0] ] = stay_assignment[1]
#     return stay_indices

def get_prediction_timestamps(stays, time_of_day='07:00:00'): 
    """
        Build dataframe of prediction date times from stays and a 
        time of day, expressed as a string like "07:00:00". 

        This procedure relies on the assumption (which seems correct on inspection)
        that the census is an automatic process that runs just before midnight 
        each day.
        Normalise the stay ranges into multiple rows of one row per date
    """

    print("Constructing jobs data")
    time_delta = pd.to_timedelta(time_of_day)
    jobs_data = []
    for idx, stay in stays.iterrows(): 
        jobs_data.append((idx, stay.copy(), time_delta))
    
    print("Launching jobs")
    with Pool(min(os.cpu_count() - 4, 24)) as pool: 
        ptimes_list = pool.map(_getpredictiontimestampsForStayWorker, jobs_data)

    print("Concatenating data frames")
    ptimes_list = [el for el in ptimes_list if el is not None]
    prediction_times = pd.concat(ptimes_list, axis=0)
    prediction_times['stayrowindex'] = prediction_times.stayrowindex.astype(np.int)
    prediction_times['facilityid'] = prediction_times.facilityid.astype(np.int)
    prediction_times = prediction_times.sort_values(['masterpatientid', 'facilityid', 'predictiontimestamp'])
    prediction_times = prediction_times.reset_index(drop=True)
    return prediction_times

def deduper(df, unique_keys=[]):
    df = df.drop_duplicates(subset=unique_keys, keep='last')
    assert df.duplicated(subset=unique_keys).sum() == 0, f'''Still have dupes!'''

    return df


def _convertToCSRWorker(job_data): 
    sp_arr = job_data[0]
    col_idx = job_data[1]
    fill_value = job_data[2]

    dense_column = np.asarray(sp_arr) # Convert to float32?  
    row_indices = list(np.nonzero(dense_column != fill_value)[0])
    if len(row_indices) == 0: 
        return (None, None, None)
    else: 
        values = list(dense_column[row_indices])
        col_indices = [col_idx] * len(row_indices)
        return (values, row_indices, col_indices)

def convert_df_to_csr_matrix(df, fill_value=0): 

    # Set up jobs data
    print('Creating jobs data')
    jobs_data = []
    for col_idx, col_name in enumerate(df.columns.values): 
        job_data = (df[col_name].copy(), col_idx, fill_value)
        jobs_data.append(job_data)

    print('Running jobs')
    with Pool(min(os.cpu_count() - 4, 24)) as pool: 
        conversion_list = pool.map(_convertToCSRWorker, jobs_data)
    print('Done with jobs...')

    print('Consolidating lists...')
    row_indices, col_indices, values = [], [], []
    for new_vals, new_rows, new_cols in conversion_list: 
        if new_rows is not None: 
            values.extend(new_vals)
            row_indices.extend(new_rows)
            col_indices.extend(new_cols)
    
    print('Constructing csr matrix...')
    retval = csr_matrix((values, (row_indices, col_indices)), 
                         shape=df.shape, 
                         dtype=np.float32)
    return retval



def _forwardFillByStayWorker(job_data): 
    colname = job_data[0]
    stay_ids = job_data[1]
    sp_array = job_data[2]

    values = np.asarray(sp_array)
    df = pd.DataFrame({'StayID': stay_ids, 
                       'Value': values})

    # Assumes stay ids and values are already sorted in time... 
    ffilled_values = df.groupby('StayID')['Value'].cumsum()
    new_column = pd.Series(pd.arrays.SparseArray(ffilled_values, fill_value=0))

    return (colname, new_column)


def forward_fill_by_stay(df, stay_assignments): 
    # Note - assumes df rows and stay_assignments are sorted by increasing time... 
    print('Constructing jobs data...')
    jobs_data = []
    for colname in df.columns.values: 
        column = df[colname]
        jobs_data.append((colname, stay_assignments, column))
    
    print('Starting jobs...')
    MAX_CPU = 32
    with Pool(min(os.cpu_count() - 4, MAX_CPU)) as pool: 
        new_columns_list = pool.map(_forwardFillByStayWorker, jobs_data)

    print('Constructing new df...')
    new_columns = [el[1] for el in new_columns_list]    
    new_df = pd.concat(new_columns, keys=df.columns.values, axis=1)
    return new_df

def get_rehosp_target(ptimes, stays, num_days=4): 
    indices = ptimes.stayrowindex.values
    dates_of_transfer = stays.dateoftransfer.dt.date.values[indices]
    prediction_dates = ptimes.predictiontimestamp.dt.date.values
    time_diffs = [ transfer_date - predict_date for transfer_date, predict_date in zip(dates_of_transfer, prediction_dates)]
    time_diffs_in_days = [diff.days if type(diff) is datetime.timedelta else 1e6 for diff in time_diffs]
    time_diffs_in_days = np.array(time_diffs_in_days)
    mask = (time_diffs_in_days <= num_days)
    target = np.zeros(len(time_diffs_in_days))
    target[mask] = 1    
    return target


# ==================================================================================================

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
