import boto3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import os 
import sys

from collections import defaultdict, namedtuple 
from multiprocessing import Pool 


def _mapVitalsToPtimesWorker(job_data): 
    group_key = job_data[0]
    vitals_for_key = job_data[1]
    ptimes_for_key = job_data[2]
    time_bin = job_data[3]

    start_delta = pd.to_timedelta(time_bin[0])
    end_delta = pd.to_timedelta(time_bin[1])

    # Construct time interval index and add to ptimes_for_key
    ptimes_for_key = ptimes_for_key.copy()
    dates = ptimes_for_key.predictiontimestamp.dt.date.values
    pdates = pd.Series([pd.Timestamp(date) for date in dates])
    start_dates = pdates + start_delta
    end_dates = pdates + end_delta
    interval_index = pd.IntervalIndex.from_arrays(start_dates, end_dates, 'both')
    ptimes_for_key = ptimes_for_key.set_index(interval_index)

    # Iterate through vitals_for_key trying to find matching rows in ptimes_for_key 
    # based on the index... What are the returned values? 
    vitals_indices, ptime_indices = [], []
    for vitals_idx, vitals_record in vitals_for_key.iterrows(): 
        vitals_indices.append(vitals_record.RowIndex)
        try: 
            ptime_record = ptimes_for_key.loc[vitals_record.date]
            if len(ptime_record.shape) > 1: 
                ptime_record = ptime_record.iloc[0]
            ptime_indices.append(ptime_record.RowIndex)
        except: 
            ptime_indices.append(None)


#    return (vitals_for_key, ptimes_for_key)
    return (vitals_indices, ptime_indices)


def _featurizeVitalsWorker(job_data): 
    num_ptimes = job_data[0]
    bin_num = job_data[1]
    category = job_data[2]
    vitals_data = job_data[3]

    grouped_vitals = vitals_data.groupby('PredictionTimesRowIndex')
    stats = ['count', 'min', 'max', 'mean']
    features_list = []
    for stat in stats: 
        features = np.zeros(num_ptimes)
        if stat == 'count': 
            stats_series = grouped_vitals['value'].count()
        elif stat == 'min': 
            stats_series = grouped_vitals['value'].min()
        elif stat == 'max': 
            stats_series = grouped_vitals['value'].max()
        else: 
            stats_series = grouped_vitals['value'].mean()
        indices = stats_series.index.values
        values = stats_series.values
        features[indices] = values
        features = pd.arrays.SparseArray(features, fill_value=0)
        features_list.append(features)
    colnames = ['vitals_' + stat + f'_{category}_bin{bin_num}' for stat in stats]
    df = pd.DataFrame({k:v for k,v in zip(colnames, features_list)})

    return df 


class BasicTimeBinnedVitals: 
    def __init__(self): 
        pass

    def featurize(self, prediction_times, vitals_data, time_bins=None): 
        # TODO - how do we do baseline measurements?  

        vitals_categories = ['Pulse', 'BP - Systolic', 'Pain Level', 
                             'Respiration', 'O2 sats', 'Temperature', 
                             'Weight', 'Blood Sugar', 'DiastolicValue']

        print('Making copies of prediction times and vitals...')
        vitals_data = vitals_data.copy()
        vitals_data['RowIndex'] = np.arange(len(vitals_data))
        ptimes = prediction_times.copy()
        ptimes['RowIndex'] = np.arange(len(ptimes))

        # Default time bins (deltas for start and end intervals relative to midnight)
        if time_bins is None: 
            #time_bins = [('-5 hours', '7 hours')] 
            time_bins = [('-5 hours', '7 hours'), 
                         ('-17 hours', '-5 hours'), 
                         ('-29 hours', '-17 hours'), 
                         ('-41 hours', '-29 hours')]
        print(f"Using relative time bins: {time_bins}")

        # Construct jobs_data for mapping vitals records to ptimes records. 
        # We reuse these a lot. 
        grouped_ptimes = ptimes.groupby(['masterpatientid', 'facilityid'])
        group_key_to_ptimes = {}
        for group_key, group in grouped_ptimes: 
            group_key_to_ptimes[group_key] = group
        grouped_vitals = vitals_data.groupby(['masterpatientid', 'facilityid'])
        num_groups = 0
        for group_key, group in grouped_vitals: 
            num_groups += 1
        print(f"{num_groups} vitals groupings...")

        print(f'Constructing jobs data for mapping vitals times to prediction times....')
        raw_jobs_data = []
        num_iter = 0
        for group_key, vitals_group in grouped_vitals: 
            num_iter += 1
            if num_iter % 10000 == 0: 
                print(f"Working on {num_iter}")
            if group_key in group_key_to_ptimes: 
                ptimes_group = group_key_to_ptimes[group_key]
                job_data = (group_key, vitals_group, ptimes_group)
                raw_jobs_data.append(job_data)

        print('Done...')

        features_for_time_bin_list = []
        for time_bin_idx, time_bin in enumerate(time_bins): 
            print(f'Constructing features for time bin {time_bin}...')
            jobs_data = []
            for raw_job_data in raw_jobs_data: 
                job_data = [raw_job_data[0], 
                            raw_job_data[1], 
                            raw_job_data[2], 
                            time_bin]
                jobs_data.append(job_data)
            
            # Running jobs
            print("\tMapping vitals to prediction times...")
            with Pool(min(os.cpu_count() - 4, 40)) as pool:
                vitals_to_ptime = pool.map(_mapVitalsToPtimesWorker, jobs_data) 

            # Compile single list mapping vitals rows to vitals data
            vitals_to_ptime_array = np.zeros(len(vitals_data))
            vitals_to_ptime_array[:] = np.nan
            for vitals_rows, ptime_rows in vitals_to_ptime: 
                for vitals_row, ptime_row in zip(vitals_rows, ptime_rows): 
                    if ptime_row is not None: 
                        vitals_to_ptime_array[vitals_row] = ptime_row
            vitals_data['PredictionTimesRowIndex'] = vitals_to_ptime_array

            print(f'\tSetting up jobs data for each vitals category')
            featurize_jobs_data = []
            for vitals_category in vitals_categories: 
                if vitals_category == 'DiastolicValue': 
                    vitals_for_category = vitals_data[ ~vitals_data.diastolicvalue.isna() ].copy()
                    vitals_for_category['Value'] = vitals_for_category.diastolicvalue
                else: 
                    vitals_for_category = vitals_data[vitals_data.vitalsdescription == vitals_category].copy()
                mask = (~vitals_for_category.PredictionTimesRowIndex.isna()).values
                vitals_for_category = vitals_for_category[mask]
                vitals_for_category['PredictionTimesRowIndex'] = vitals_for_category.PredictionTimesRowIndex.astype(np.int)
                featurize_jobs_data.append((len(ptimes), 
                                            time_bin_idx, 
                                            vitals_category, 
                                            vitals_for_category))

            print('\tRunning featurize jobs...')
            with Pool(min(os.cpu_count() - 4, 40)) as pool: 
                features_df_list = pool.map(_featurizeVitalsWorker, featurize_jobs_data)
            features_for_time_bin_list.append(pd.concat(features_df_list, axis=1))

        print('Constructing final feature dataframe')
        features = pd.concat(features_for_time_bin_list, axis=1)
        return features





        