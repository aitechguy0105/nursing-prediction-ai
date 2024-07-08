import copy
import gc
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')


# from shared.utils import cumsum_job


class BaseFeaturizer(object):
    def __init__(self):
        self.ignore_columns = ['masterpatientid', 'censusdate', 'facilityid']

    def sorter_and_deduper(self, df, sort_keys=[], unique_keys=[]):
        """
        input: dataframe
        output: dataframe
        desc: 1. sort the dataframe w.r.t sort_keys.
              2. drop duplicates w.r.t unique keys.(keeping latest values)
        """
        df.sort_values(by=sort_keys, inplace=True)
        df.drop_duplicates(
            subset=unique_keys,
            keep='last',
            inplace=True
        )
        assert df.duplicated(subset=unique_keys).sum() == 0, f'''Still have dupes!'''
        return df

    def downcast_dtype(self, df):
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

    def convert_to_int16(self, df):
        """
        Convert entire dataframe to int16 columns
        """
        for col in df:
            df[col] = df[col].astype('int16')
        return df

    def add_na_indicators(self, df, ignore_cols):
        """
        - Add additional columns to indicate NaN indicators for each column in cols
        Later the NaN will be filled by either Median or Mean values
        """
        log_message(message_type='info', message=f'Add Na Indicators')
        # Boolean in each cell indicates whether its None
        missings = df.drop(columns=ignore_cols).isna()
        missings.columns = 'na_indictator_' + missings.columns
        missings_sums = missings.sum()

        return pd.concat([df, missings.loc[:, (missings_sums > 0)]], axis=1)

    def add_datepart(self, df, fldname, drop=True, time=False, errors='raise'):
        """
        - making extra features from date. all the feature names are present in attr list.

        :param df: dataframe
        :param fldname: string
        :param drop: bool
        :param time: bool
        :param errors: string
        :return: dataframe
        """
        log_message(message_type='info', message=f'Extracting date features for {fldname}')
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start',
                'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr:
            df[fldname + '_' + n] = getattr(fld.dt, n.lower())
        if drop:
            df.drop(fldname, axis=1, inplace=True)

        return df

    def cumsum_job(self, cols, df):
        # Filling all NaN with 0
        filled = df.groupby('masterpatientid')[cols].fillna(0)
        filled['masterpatientid'] = df.masterpatientid

        cumsum_all_time = filled.groupby('masterpatientid')[cols].cumsum()
        cumsum_all_time.columns = ('cumsum_all_' + cumsum_all_time.columns)

        # Drop existing dx_cols, med_cols, alert_cols, order_cols columns
        df = df.drop(columns=cols)

        # =============For Memory optimisation convert cumsum to int16=============
        cumsum_all_time = self.convert_to_int16(cumsum_all_time)

        df = pd.concat(
            [df, cumsum_all_time], axis=1
        )  # cumsum is indexed the same as original in the same order
        # =======================Trigger Garbage collector and free up memory============================
        del cumsum_all_time
        gc.collect()

        # Rolling sum (Cumulative sum) of last 7 days
        log_message(message_type='info', message=f'Rolling window for last 7 days..')
        cumsum_7_day = (
            filled.groupby('masterpatientid')[cols].rolling(7, min_periods=1).sum().reset_index(0, drop=True))
        cumsum_7_day.columns = 'cumsum_7_day_' + cumsum_7_day.columns

        log_message(message_type='info', message=f'Rolling window for last 14 days..')
        cumsum_14_day = (
            filled.groupby('masterpatientid')[cols]
                .rolling(14, min_periods=1)
                .sum()
                .reset_index(0, drop=True)
        )
        cumsum_14_day.columns = 'cumsum_14_day_' + cumsum_14_day.columns

        log_message(message_type='info', message=f'Rolling window for last 30 days..')
        cumsum_30_day = (
            filled.groupby('masterpatientid')[cols]
                .rolling(30, min_periods=1)
                .sum()
                .reset_index(0, drop=True)
        )
        cumsum_30_day.columns = 'cumsum_30_day_' + cumsum_30_day.columns

        rollings = pd.concat(
            [cumsum_7_day, cumsum_14_day, cumsum_30_day], axis=1
        )

        # =======================Trigger Garbage collector and free up memory============================
        log_message(message_type='info', message=f'cleaning up and reducing memory..')
        del cumsum_7_day
        del cumsum_14_day
        del cumsum_30_day
        gc.collect()

        # =============For Memory optimisation convert cumsum to int16=============
        rollings = self.convert_to_int16(rollings)

        df = df.merge(
            rollings, how='left', left_index=True, right_index=True
        )  # rollings were sorted so we explictly join via index

        # =======================Trigger Garbage collector and free up memory============================
        del rollings
        gc.collect()

        log_message(message_type='info', message='Feature generation for other columns completed successfully')

        return df

    def get_cumsum_features(self, cols, df):
        """
        :return: dataframe
        """
        # Get the total CPU cores for parallelising the jobs
        n_cores = os.cpu_count() * 2
        # Get all unique patients
        uq_patient = df.masterpatientid.unique()
        # Split the unique patients into groups as per CPU cores
        patient_grp = np.array_split(uq_patient, 35)
        df_split = []
        for grp in patient_grp:
            # Create dataframe chunks as per masterpatientid groups selected above
            # If check to handle the condition where unique masterpatientid is less than array_split value(35)        
            if len(grp) == 0:
                continue
            df_split.append(
                df[df.masterpatientid.isin(grp)]
            )

        with Pool(n_cores) as pool:
            proc_before = copy.copy(pool._pool)
            
            # Pass static parameters (excluding iterative parameter) to the target function.
            func = partial(self.cumsum_job, cols)
            # Spawn multiple process & call target function while passing iterative parameter
            mapped = pool.map_async(func, df_split)

            while not mapped.ready():
                if any(proc.exitcode for proc in proc_before):
                    raise Exception('One of the subproceses has died')
                mapped.wait(timeout=10)
            else:
                df = pd.concat(mapped.get())

        return df
    
    def drop_columns(self, df, pattern):
        """
        :return: dataframe
        """
        return df.drop(
            df.columns[df.columns.str.contains(
            pattern)].tolist(),
            axis=1
        )