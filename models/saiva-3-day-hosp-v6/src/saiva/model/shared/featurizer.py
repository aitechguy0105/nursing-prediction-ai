import copy
import gc
import os
import sys
from functools import partial
from multiprocessing import Pool
from datetime import timedelta
from typing import Union, Callable, Literal

import numpy as np
import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay



# from .utils import cumsum_job


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
        int_cols = df.columns[df.dtypes == 'int64']
        int_cols = int_cols[int_cols!='masterpatientid']
        df.loc[:, int_cols] = df[int_cols].apply(pd.to_numeric, downcast='unsigned')
        
        # Convert all float64 columns to float32
        float_cols = df.columns[df.dtypes == 'float64']
        float_cols = float_cols[float_cols!='masterpatientid']
        df.loc[:, float_cols] = df[float_cols].astype('float32')

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
            
        #censusdate featurization
        if fldname=='censusdate': 
            df['day_name'] = df.censusdate.dt.day_name().astype('category')
            df['month_name'] = df.censusdate.dt.month_name().astype('category')
            df['day_of_year'] = df['censusdate'].dt.dayofyear
            dic = {12:'w', 1:'w', 2:'w',
                    3:'sp', 4:'sp', 5:'sp',
                    6:'su', 7:'su', 8:'su',
                    9:'a', 10:'a', 11:'a'}
            df['season'] = df[fldname + '_Month'].map(dic).astype('category')
            df['is_weekday'] = df.censusdate.dt.weekday<5
            
            # adding holiday and business features
            num_day_to_Mon = pd.to_datetime(df.censusdate).min().weekday() 
            first_day = pd.to_datetime(df.censusdate).min() - pd.DateOffset(days=num_day_to_Mon+1)

            num_day_to_Sun = 6 - pd.to_datetime(df.censusdate).max().weekday() 
            last_day = pd.to_datetime(df.censusdate).max() + pd.DateOffset(days=num_day_to_Sun)

            us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar()) 
            all_days_range = pd.date_range(start=first_day, end=last_day)

            business_days_df = pd.DataFrame(all_days_range, columns=['censusdate'])
            business_days_df['is_office_holiday'] = business_days_df.censusdate.isin(us_bd.holidays)
            business_days_df['is_business_day'] = business_days_df.censusdate\
                                                    .isin(pd.date_range(start=first_day, end=last_day, freq=us_bd))
            
            # compute Days to next reduced work day (Weekend or Office Holiday)
            business_days_df.loc[~business_days_df.is_business_day, 'is_business_day'] = None
            reversed_ = business_days_df.is_business_day.iloc[::-1]
            s = (reversed_ != reversed_.shift()).cumsum()
            business_days_df.loc[:,'days_to_next_reduced_workday'] = s.groupby(s).cumcount().iloc[::-1]
            business_days_df.loc[~business_days_df.is_business_day.isnull(), 'days_to_next_reduced_workday'] = \
                                        business_days_df[~business_days_df.is_business_day.isnull()]\
                                            ['days_to_next_reduced_workday'] + 1

            # compute Days since last reduced work day
            s = (business_days_df["is_business_day"] != business_days_df["is_business_day"].shift()).cumsum()
            business_days_df.loc[:,'days_since_last_reduced_workday'] = s.groupby(s).cumcount()
            business_days_df.loc[~business_days_df.is_business_day.isnull(), 'days_since_last_reduced_workday'] = \
                                        business_days_df[~business_days_df.is_business_day.isnull()]\
                                        ['days_since_last_reduced_workday'] + 1
            
            business_days_df = business_days_df.drop(columns='is_business_day')
            df = df.merge(business_days_df, on='censusdate', how='left')

        return df
    
    def get_consecutive_censusdate(self, cols, df):
        """
        cols: list of string, the columns name of events in df
        df: dataframe, with columns masterpatientid, censusdate, 
            and columns in cols
            
        return:
              df: dataframe, with columns masterpatientid, censusdate, and columns 
                  in cols, censusdate are consecutive
              non_events_df: dataframe, with columns masterpatientid, censusdate,
                           which is copy from input dataframe
        """
        #creat a extra dataframe which having the same patient days as the original census data
        #so that we can use this dataframe to remove the non patient days
        other_cols = [i for i in df.columns if i not in cols]
        non_events_df = df[other_cols].copy()
        events_df = df[['masterpatientid','censusdate']+cols]

        non_events_df.loc[:,'censusdate'] = pd.to_datetime(non_events_df['censusdate'])
        min_dates = non_events_df.groupby('masterpatientid')['censusdate'].min()
        max_dates = non_events_df.groupby('masterpatientid')['censusdate'].max()
        masterpatientids = min_dates.index.values
        
        #to make sure start and end date for each masterpatientid are correct
        assert sum(masterpatientids==max_dates.index.values)==len(masterpatientids)
        
        merged_list = list(zip(min_dates, max_dates))        
        consecutive_dates = pd.DataFrame()
        consecutive_dates['masterpatientid']=masterpatientids
        consecutive_dates['min_dates']=min_dates.values
        consecutive_dates['max_dates']=max_dates.values
        consecutive_dates['censusdate'] = [pd.date_range(x, y) for x, y in merged_list]
        consecutive_dates = consecutive_dates.explode('censusdate').reset_index(drop=True)
        
        df = consecutive_dates[['masterpatientid', 'censusdate']]\
                .merge(events_df, 
                       how='left',
                       on=['masterpatientid', 'censusdate'])
        return df, non_events_df 


    def cumsum_job(self, cols, df, cumidx=False, slidding_windows=[2,7,14,30]):
        """
        cols: string, the a list of column name that we want to do cumsum job
        df: dataframe, the dataframe with facilityid,'masterpatientid', censusdate, and columns 
            with column name in cols
        cumidx: boolean,  
        """
        #insert some rows into the df to make the censusdate consecutive for each patient, 
        #all values of cols in this rows are filled with Nan
        df, non_events_df = self.get_consecutive_censusdate(cols, df)
        
        # Filling all NaN with 0
        filled = df.groupby('masterpatientid')[cols].fillna(0)
        filled['masterpatientid'] = df.masterpatientid
        if cumidx:
            filled[cols] = filled[cols]>0
            col_prefix = 'cumidx'
        else:
            col_prefix = 'cumsum'
            
        cumsum_all_time = filled.groupby('masterpatientid')[cols].cumsum()
        cumsum_all_time.columns = (col_prefix +'_all_' + cumsum_all_time.columns)

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
        
        cumsum_dfs = []
        for windows in slidding_windows:
            cumsum = (
                filled.groupby('masterpatientid')[cols].rolling(windows, min_periods=1).sum().reset_index(0, drop=True))
            cumsum.columns = col_prefix + f'_{windows}_day_' + cumsum.columns
            cumsum_dfs.append(cumsum)
            
        rollings = pd.concat(cumsum_dfs, axis=1)    

        # =======================Trigger Garbage collector and free up memory============================
        del cumsum_dfs
        gc.collect()

        # =============For Memory optimisation convert cumsum to int16=============
        rollings = self.convert_to_int16(rollings)

        df = df.merge(
            rollings, how='left', left_index=True, right_index=True
        )  # rollings were sorted so we explictly join via index

        # =======================Trigger Garbage collector and free up memory============================
        del rollings
        gc.collect()
        
        df = non_events_df.merge(df,on=['masterpatientid', 'censusdate'],how='left')
        assert df.shape[0] == non_events_df.shape[0]
        
        return df

    def get_cumsum_features(self, cols, df, cumidx=False, slidding_windows=[2,7,14,30]):
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
            func = partial(self.cumsum_job, cols, cumidx=cumidx, slidding_windows=slidding_windows)
            # Spawn multiple process & call target function while passing iterative parameter

            mapped = pool.map_async(func, df_split)

            while not mapped.ready():
                if any(proc.exitcode for proc in proc_before):
                    raise Exception('One of the subproceses has died')
                mapped.wait(timeout=10)
                total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
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

    def n_days_last_event(self, df, col):

        """
        Compute the number of days since last event for one particular column.

        Parameters:
        -----------
            df : pandas.DataFrame
                Pivoted table with at least 3 columns: 'masterpatientid', 'censusdate' and one specified in `col
            col : string
                Column name (event name) in `df` that we want to count days of last event

        Returns:
        --------
            result : pandas.Series of int
                Series of days since last event in a column `col` aligned with an order of `masterpatientid`
                and `censusdate` in the original `df`
        """

        # Create `event_date` column and fill it
        df['event_date'] = pd.NaT
        df.loc[df[col], 'event_date'] = df.loc[df[col], 'censusdate']
        df['event_date'] = df.groupby('masterpatientid')['event_date'].ffill()

        # Find the difference between today and event date
        df['days_since_last_event'] = (df['censusdate'] - df['event_date']).dt.days
        # The rows before the first event for given `masterpatientid` happened have NaNs, fill it with 9999
        df['days_since_last_event'].fillna(9999, inplace=True)

        result = df['days_since_last_event'].astype('int16')
        del df
        return result

    def apply_n_days_last_event(self, df, cols):
        """
        Compute the number of days since the last event happened for the given list of columns.

        Parameters:
        -----------
        df : pandas.DataFrame
            Pivoted table with required columns: 'masterpatientid', 'censusdate'
        cols : list
            List of column names in `df` to process; names of events

        Returns:
        --------
        result : pandas.DataFrame
            Dataframe with the first 2 columns: 'masterpatientid', 'censusdate' and event columns with
                computed number of days since last event.
        """

        # Create a new copy of dataframe (to make sure the original dataframe will not be modified) and sort
        df = df.sort_values(by=['masterpatientid', 'censusdate'])
        df.reset_index(drop=True, inplace=True)
        # find when event happended
        df[cols] = (df[cols]>0)
        result = df[['masterpatientid','censusdate']].copy()

        # get days_since_last_event for each event in cols
        with Pool(max(os.cpu_count()-2, 1)) as pool:
            proc_before = copy.copy(pool._pool)

            args = [(df[['masterpatientid', 'censusdate', col]].copy(), col) for col in cols]
            mapped = pool.starmap_async(self.n_days_last_event, args)

            while not mapped.ready():
                if any(proc.exitcode for proc in proc_before):
                    raise Exception('One of the subproceses has died')
                mapped.wait(timeout=10)
            else:
                colwise_results = mapped.get()

        for i, col in enumerate(cols):
            result[col+'_days_since_last_event'] = colwise_results[i]
        #for col in cols:
        #    sub_df = df[['masterpatientid','censusdate',col]].copy()
        #    result[col+'_days_since_last_event'] = n_days_last_event(sub_df, col).astype(int)

        return result

    def apply_n_days_last_event_v2(self, df, cols):
        """ It is another implementation of the `apply_n_days_last_event` method. See the documentation there
        """
        # Create a new copy of dataframe (to make sure the original dataframe will not be modified)
        df = df.copy()
        # Convert censusdate to float
        df['censusdate'] = df['censusdate'].astype(int) / 10**9
        df.set_index(['masterpatientid', 'censusdate'], drop=True, inplace=True)
        df.sort_index(inplace=True)
        # find when event happended
        df[cols] = (df[cols]>0) # in case not boolean were passed
        event_dates = np.where(df, df.index.get_level_values('censusdate').values.reshape([-1,1]), np.nan)
        event_dates = pd.DataFrame(event_dates, index=df.index, columns=df.columns, dtype='float32')
        del df
        event_dates = event_dates.groupby('masterpatientid').ffill()
        deltas = ((event_dates.index.get_level_values('censusdate').values.reshape([-1,1]) - event_dates) / 86400).fillna(9999).astype('int16')
        del event_dates
        deltas.columns = [f'{col}_days_since_last_event' for col in deltas.columns]
        deltas.reset_index(inplace=True)
        # Convert censusdate back to datetime
        deltas['censusdate'] = pd.to_datetime(deltas['censusdate'], unit='s')
        return deltas
    
    def days_since_bed_change(self, df):
        '''
        df: dataframe, must has cloumns 'facilityid', 'masterpatientid','bedid','censusdate' 
            the 'masterpatientid', 'censusdate', must be sort in ascending order
            
        return: dataframe, with one more column 'days_since_bed_change'
        '''
        group_cols = ['facilityid', 'masterpatientid']
        if 'bedid' in df.columns and not df['bedid'].isnull().all():
            group_cols.append('bedid')
        is_consective = df.groupby(group_cols)\
                    ['censusdate'].diff()==pd.Timedelta('1d')
        # mark data with consecutive date as group 1,2,3... 
        group = (~is_consective).cumsum()
        df['group'] = group
        df['is_consective'] = is_consective
        # cumsum for 'is_consective'(True) for each group
        df['day_since_bed_change'] = df.groupby(['facilityid', 'masterpatientid','group'])\
                        .is_consective.apply(np.cumsum)
        # change first days of bed cahnging(False) and Null to 0
        df.loc[df['day_since_bed_change']==False, 'day_since_bed_change'] = 0
        df.loc[df['day_since_bed_change'].isnull(), 'day_since_bed_change'] = 0
        df.drop(columns=['group','is_consective'], inplace=True)
        df['day_since_bed_change'] = df['day_since_bed_change'].astype('int16')
        return df
    
    def get_target(self, incident_df, census_df, incident_type, numdays=3):
        """
        incident_df: dataframe, with columns 'masterpatientid', 'incidentdate'
        census_df: data frame, with columns 'masterpatientid', 'censusdate', including all the patient days
        incident_type: string, type of incident, eg: rth, fall, ...
        """  
        census_df = census_df[['masterpatientid', 'censusdate']].copy()
        census_df.loc[:,'censusdate'] = pd.to_datetime(census_df['censusdate'])
        incident_df.loc[:,'incidentdate'] = pd.to_datetime(incident_df['incidentdate']).dt.date
        # some patients may have more than one inclident in one day, only keep one incident date in
        # one day for each patient
        incident_df = incident_df.drop_duplicates()
        # sutract 1 day from incidentdate
        incident_df.loc[:,'incidentdate'] = (pd.to_datetime(incident_df['incidentdate'])\
                                                              - timedelta(days=1))
        #merge the incident dataframe with census dataframe, keep all patient days in census data 
        census_df = census_df.merge(incident_df,
                          left_on=['masterpatientid', 'censusdate'],
                          right_on=['masterpatientid', 'incidentdate'],
                          how='outer')
        census_df.loc[census_df.censusdate.isnull(), 'censusdate'] = census_df.loc[census_df.censusdate.isnull(),\
                                                                               'incidentdate']
        # sort the merged table with 'masterpatientid', 'censusdate' in ascending order
        census_df= census_df.sort_values(['masterpatientid', 'censusdate']).reset_index(drop=True)
        #backward filling the null 'incidentdate' 3 position
        census_df[f'positive_date_{incident_type}'] = census_df.groupby('masterpatientid')\
                                                        ['incidentdate'].bfill(limit=numdays)
        # create postive target if incidentdate-censusdate<=numdays
        census_df.loc[:,f'target_{numdays}_day_{incident_type}'] = ((census_df[f'positive_date_{incident_type}']-\
                                                           census_df.censusdate).dt.days<=numdays)    
        census_df.loc[(~census_df[f'positive_date_{incident_type}'].isnull())&\
                      (~census_df[f'target_{numdays}_day_{incident_type}']), f'positive_date_{incident_type}'] = None
        # add 1 day to the positive_date to get back the original incident date
        census_df.loc[:,f'positive_date_{incident_type}'] = (pd.to_datetime(census_df[f'positive_date_{incident_type}'])\
                                                      + timedelta(days=1))
        census_df.drop(columns = 'incidentdate',inplace=True)
        return census_df
    
    def sanity_check(self, df):
        """ Brief check on whether there is something wrong with `df`.
        The first failed check raises an `AssertionError`.
        
        Parameters:
        ----------
            df : pandas.DataFrame
                Dataframe to be checked
        """
        assert not df.duplicated(subset=['masterpatientid', 'censusdate']).any(),\
            "There are duplicated patient-days in the dataframe"
        assert not df.columns.duplicated().any(),\
            "There are duplicated columns in the dataframe"
        assert not df.columns.str.contains('_x$|_y$').any(),\
            "There are columns with `_x` and/or `_y` suffixes in the name"
        assert not df.columns.str.contains('_masterpatientid|_facilityid').any(),\
            "There are duplicated `masterpatientid` and/or `facilityid` columns"
        assert len(df) == len(self.census_df),\
            "Number of rows in the given dataframe is different from `census_df`"
        return

    def _conditional_ops_preprocessing(
        self,
        df,
        event_date_column,
        event_reported_date_column,
        groupby_column,
        missing_event_dates,
        normalize_dates=True,
        other_columns_to_keep=[],
        lowercase_groupby_column=False
    ):
        
        # Validate input
        columns_to_use = ['masterpatientid', event_date_column, event_reported_date_column] + other_columns_to_keep
        if groupby_column:
            columns_to_use.append(groupby_column)
        for col in columns_to_use:
            assert isinstance(col, str), f"`{col}` argument must be a string"
            assert col in df.columns, f"`{col}` is not a column of `df`"
        assert df[event_reported_date_column].notnull().all(), \
            "`event_reported_date_column` column is not allowed to contain null values"
        
        df = df[columns_to_use].copy()
        
        # Handle NaNs in `event_date_column`
        if missing_event_dates == 'drop':
            df.dropna(subset=[event_date_column], inplace=True)
        elif missing_event_dates == 'fill':
            df[event_date_column].fill(df[event_reported_date_column], inpalce=True)
        else:
            NotImplementedError("Unknown argument value: `missing_event_dates` must be 'drop' or 'fill'")
        
        census = self.census_df[['masterpatientid', 'censusdate']].drop_duplicates()
        
        # Normalize date columns to remove time information
        if normalize_dates:
            census['censusdate'] = census['censusdate'].dt.normalize()
            df[event_date_column] = df[event_date_column].dt.normalize()
            df[event_reported_date_column] = df[event_reported_date_column].dt.normalize()
        df[event_reported_date_column] = df[[event_date_column, event_reported_date_column]].max(axis=1)
        if not groupby_column is None:
            if lowercase_groupby_column:
                df[groupby_column] = df[groupby_column].str.lower()
            df.dropna(subset=[groupby_column], inplace=True)
        
        return census, df
    
    def conditional_days_since_last_event(
        self,
        df,
        prefix,
        event_date_column,
        event_reported_date_column,
        groupby_column=None,
        missing_event_dates='drop',
        lowercase_groupby_column=False
    ):
        """ Generates the number of days since the last event, considering the date
        the event was recorded. In other words, the event will not be considered until it's reported.
        If the `groupby_column` argument is provided, a separate column will be generated for every
        value of this column.
        The missing event dates will either be dropped or replaced by the date the event was reported,
        depending on the `missing_event_dates` argument.

        Parameters:
        -----------
            df : pd.DataFrame
                The data to be used for calculating the results, shoud contain 'masterpatientid' and
                the columns specified in `event_date_columns`, `event_reported_date_column` and
                `groupby_column` (if provided)

            prefix : str
                Will be used for naming the new column(s)

            event_date_column : str
                The name of the column of `df` that contains the dates when the event happenned.

            event_reported_date_column : str
                The name of the column of `df` that contains the dates when the event was recorded.

            groupby_column : str or None
                If not None, specifies the name of the column in `df` with different types of events.
                For example, different types of assessments.

            missing_event_dates : str ('drop' or 'fill')
                If 'drop' all the rows with missing values will be dropped.
                If 'fill' all the missing values in the column `event_date_column` will be replaced
                by ones in the column `event_reported_date_column`.

        Return:
        -------
            df : pd.DataFrame
                The dataframe with the columns 'masterpatientid', 'censusdate' and
                f'{prefix}_{groupby_value}_days_since_last_event'

        """

        def propagate_event_dates(df):
            res = census.merge(
                df,
                left_on=['masterpatientid', 'censusdate'],
                right_on=['masterpatientid', event_reported_date_column],
                how='outer'
            )
            res['censusdate'].fillna(res[event_reported_date_column], inplace=True)
            columns_to_drop = [col for col in [event_reported_date_column, groupby_column] if col]
            res.drop(columns_to_drop, axis=1, inplace=True)
            res.sort_values(['masterpatientid', 'censusdate'], inplace=True)
            res[event_date_column] = res.groupby('masterpatientid')[event_date_column].ffill()
            res[event_date_column] = res.groupby('masterpatientid')[event_date_column].cummax()
            return res

        census, df = self._conditional_ops_preprocessing(
            df=df,
            event_date_column=event_date_column,
            event_reported_date_column=event_reported_date_column,
            groupby_column=groupby_column,
            missing_event_dates=missing_event_dates,
            lowercase_groupby_column=lowercase_groupby_column
        )   

        # Sort and drop duplicates in the main DataFrame
        # If several everts were reported on the same day, the only one happened the latest will be kept
        groupby_columns = [col for col in ['masterpatientid', groupby_column] if col]
        df.sort_values(groupby_columns+[event_reported_date_column, event_date_column], inplace=True)
        df.drop_duplicates(subset=groupby_columns+[event_reported_date_column], keep='last', inplace=True)

        if groupby_column:

            # Forward fill event dates and calculate delta
            df = df.groupby(groupby_column).apply(propagate_event_dates).reset_index()
            df['delta'] = (df['censusdate'] - df[event_date_column]).dt.days

            # Pivot the DataFrame for the final output
            df = df[groupby_columns + ['censusdate', 'delta']].set_index(
                ['masterpatientid', 'censusdate', groupby_column]
            )['delta'].unstack()
            df.columns = f'{prefix}_' + df.columns + '_days_since_last_event'
            df.columns.name = None

        else:

            # Forward fill event dates and calculate delta
            df = propagate_event_dates(df)
            df['delta'] = (df['censusdate'] - df[event_date_column]).dt.days

            # Pivot the DataFrame for the final output
            df = df[groupby_columns + ['censusdate', 'delta']].set_index(
                ['masterpatientid', 'censusdate']
            )
            df.columns = [f'{prefix}_days_since_last_event']

        df = df.fillna(9999).astype('int16')
        df.reset_index(inplace=True)
        return df   

    @staticmethod
    def _conditional_cumsum_job(
        census,
        df,
        prefix,
        event_date_column,
        event_reported_date_column,
        missing_event_dates='drop',
        cumidx=False,
        sliding_windows=[2,7,14,30]
    ):
            
        def deltas_rolling_sum(
            df,
            sliding_window
        ):
            res = census.merge(
                df,
                left_on=['masterpatientid', 'censusdate'],
                right_on=['masterpatientid', event_reported_date_column],
                how='outer'
            )
            res['censusdate'].fillna(res[event_reported_date_column], inplace=True)
            columns_to_drop = [event_reported_date_column]
            res.drop(columns_to_drop, axis=1, inplace=True)
            delta_columns = [col for col in res.columns if isinstance(col, int) and col < sliding_window]
                
            res[delta_columns] = res[delta_columns].fillna(0)
            res.sort_values(['masterpatientid', 'censusdate'], inplace=True) # this is necessary for proper working of rolling windows
            res.reset_index(drop=True, inplace=True) # And this one too, without it res and the result of rolling are not aligned
            for delta in delta_columns:
                res[delta] = res.groupby(
                    'masterpatientid'
                ).rolling(f'{sliding_window-delta}d', on='censusdate')[delta].sum().reset_index(drop=True)
                    
            col_prefix = 'cumidx' if cumidx else 'cumsum'
            res[f'{col_prefix}_{sliding_window}_day'] = res[delta_columns].sum(axis=1).astype('int16')
                
            return res.set_index(['masterpatientid', 'censusdate'])[[f'{col_prefix}_{sliding_window}_day']]
            
        groupby_columns = ['masterpatientid', event_reported_date_column]
            
        if cumidx:
            df.sort_values([event_date_column, event_reported_date_column], inplace=True)
            df.drop_duplicates(subset=groupby_columns, inplace=True)
        df['delta'] = (df[event_reported_date_column] - df[event_date_column]).dt.days
            
        cumsum_all = census.join(
            df.groupby(groupby_columns)[['delta']].count().groupby('masterpatientid').cumsum().astype('int16'),
            how='outer',
            on = ['masterpatientid', 'censusdate']
        ).set_index(['masterpatientid', 'censusdate']).sort_index()
        # forward fill grouped by masterpatientid ( currently groupby ffill in pandas drops groupby key, so need to save key before operation )
        cumsum_all_index = cumsum_all.index
        cumsum_all = cumsum_all.groupby('masterpatientid').ffill()
        cumsum_all.index = cumsum_all_index

        cumsum_all.columns = ['cumidx_all' if cumidx else 'cumsum_all']

        df = df.loc[df['delta']<max(sliding_windows)]
        df = pd.pivot_table(
            df,
            values=event_date_column,
            index=groupby_columns,
            columns=['delta'],
            aggfunc='count',
            fill_value=0
        ).reset_index()

        df_list =[
            deltas_rolling_sum(df, sliding_window=sliding_window) for sliding_window in sliding_windows
        ] + [cumsum_all]

        # assert all dataframes have same index name ['masterpatientid', 'censusdate']
        for i in range(len(df_list)-1):
            assert df_list[i].index.names == df_list[i+1].index.names    
        df = pd.concat(df_list, axis=1, ignore_index=False)

        df.columns = df.columns + f'_{prefix}'

        return census.join(df, on=['masterpatientid', 'censusdate'], how='left').fillna(0)

    def conditional_cumsum_features(
        self,
        df,
        prefix,
        event_date_column,
        event_reported_date_column,
        groupby_column = None,
        missing_event_dates='drop',
        cumidx=False,
        sliding_windows=[2,7,14,30],
        lowercase_groupby_column=False
    ):
        """ Generates the rolling sums of the events, considering the date
            the event was recorded. In other words, the event will not be considered until it's reported.
            If the `groupby_column` argument is provided, a separate column will be generated for every
            value of this column.
            The missing event dates will either be dropped or replaced by the date the event was reported,
            depending on the `missing_event_dates` argument.

            Parameters:
            -----------
                df : pd.DataFrame
                    The data to be used for calculating the results, shoud contain 'masterpatientid' and
                    the columns specified in `event_date_columns`, `event_reported_date_column` and
                    `groupby_column` (if provided)

                prefix : str
                    Will be used for naming the new column(s). For rolling sums the names of the columns
                    will be either f'cumsum_{n}_day_{prefix}' (if there is no groupby_columns) or
                    f'cumsum_{n}_day_{prefix}_{groupby_value}'

                event_date_column : str
                    The name of the column of `df` that contains the dates when the event happenned.

                event_reported_date_column : str
                    The name of the column of `df` that contains the dates when the event was recorded.

                groupby_column : str or None
                    If not None, specifies the name of the column in `df` with different types of events.
                    For example, different types of assessments.

                missing_event_dates : str ('drop' or 'fill')
                    If 'drop' all the rows with missing values will be dropped.
                    If 'fill' all the missing values in the column `event_date_column` will be replaced
                    by ones in the column `event_reported_date_column`.
                    
                cumidx : bool
                    If True the number of days when the event happened at least one is counted.
                    If False the total number of events is counted.
                
                sliding_windows : list
                    List of windows sizes (in days) to calculate the rolling sums over.

            Return:
            -------
                df_res : pd.DataFrame
                    The dataframe with the columns 'masterpatientid', 'censusdate' and
                    multiple columns with rolling sums

        """
        
        census, df = self._conditional_ops_preprocessing(
            df,
            event_date_column,
            event_reported_date_column,
            groupby_column,
            missing_event_dates,
            lowercase_groupby_column=lowercase_groupby_column
        )
        
        pool_size = os.cpu_count() - 1
        df_split = []
        
        # If there is no `groupby_column` (e.g. description) split masterpatientid-wise
        if groupby_column is None:
            uq_patient = census.masterpatientid.unique()
            # Split the unique patients into groups as per CPU cores
            patient_grp = np.array_split(uq_patient, pool_size)
            
            for grp in patient_grp:       
                if len(grp) == 0:
                    continue
                df_split.append(
                    (
                        census[census.masterpatientid.isin(grp)],
                        df[df.masterpatientid.isin(grp)],
                        prefix
                    )
                )
        # If there is `groupby_colum` split this column-wise
        else: 
            groups = df[groupby_column].unique()
            for group in groups:
                df_split.append(
                    (
                        census,
                        df.loc[df[groupby_column]==group],
                        f'{prefix}_{group}'
                    )
                )
        
        df_split.sort(key=lambda x: -len(x[1])) # Put the larger chunks in front
    
        # Pass static parameters (excluding iterative parameter) to the target function.
        func = partial(
            self._conditional_cumsum_job,
            event_date_column=event_date_column,
            event_reported_date_column=event_reported_date_column,
            missing_event_dates=missing_event_dates,
            cumidx=cumidx,
            sliding_windows=sliding_windows
        )
        
        with Pool(pool_size) as pool:
            proc_before = copy.copy(pool._pool)

            # Spawn multiple processes & call target function while passing iterative parameter
            mapped = pool.starmap_async(func, df_split)

            while not mapped.ready():
                if any(proc.exitcode for proc in proc_before):
                    raise Exception('One of the subproceses has died')
                mapped.wait(timeout=10)
                total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            else:
                output = mapped.get()


        # Concatenating the results coming from multiple processes
        if groupby_column:
            df_res = pd.concat(
                [out.set_index(['masterpatientid', 'censusdate']) for out in output],
                axis=1,
                sort=False
            ).reset_index()
        else:
            df_res = census.merge(
                pd.concat(output, axis=0, sort=False),
                how='left',
                on=['masterpatientid', 'censusdate']
            )
            
        return df_res

    def conditional_get_last_values(
        self,
        *,
        df: pd.DataFrame,
        prefix: str,
        event_date_column: str,
        event_reported_date_column: str,
        event_deleted_date_column: Union[str, None],
        value_columns: list,
        groupby_column: Union[str, None] = None,
        missing_event_dates: Literal['drop', 'fill'] = 'drop',
        n: int = 1,
        reset: Union[pd.MultiIndex, None] = None
    ) -> pd.DataFrame:
        """ Using the conditional logic returns the `n` last values of the `value_columns`

        Parameters:
        -----------
            df: dataframe to operate on
            prefix: used in the resulted column names
            event_date_column: name of the column that specifies when the event happened
            event_reported_date_column: name of the column that specifies when the record considered to be created
            event_deleted_date_column: name of the column that specifies when the record considered to be deleted
            value_columns: list of the columns to apply the logic to
            groupby_column: name of the column with group labels if the logic should be applied group-wise
            missing_event_dates: what to do with missing values in event_date_column - drop or fill with event_reported_date_columns
            n: number of the last values to return
            reset: optionally multiindex with the timestamps to reset search (i.e. if you want the last 2 values but only after the last
                admission, you can provide this parameter with the admission, and they will mask all the values before the admission).
                If provided the multiindex should contains 3 levels: 'masterpatientid', event_reported_date_column, event_date_column    


        """
        census, df = self._conditional_ops_preprocessing(
            df = df,
            event_date_column = event_date_column,
            event_reported_date_column = event_reported_date_column,
            groupby_column = groupby_column,
            missing_event_dates = missing_event_dates,
            normalize_dates = False,
            other_columns_to_keep = value_columns + ([event_deleted_date_column] if event_deleted_date_column else [])
        )
        
        # If groupby_column is provided we just apply the logic for each group
        if groupby_column:
            df = df.groupby(groupby_column).apply(
                lambda x: self.conditional_get_last_values(
                    df = x,
                    prefix = '',
                    event_date_column = event_date_column,
                    event_reported_date_column = event_reported_date_column,
                    event_deleted_date_column = event_deleted_date_column,
                    value_columns = value_columns,
                    groupby_column = None,
                    missing_event_dates = missing_event_dates,
                    n = n,
                    reset = reset
                )
            ).reset_index().drop(columns=['level_1'], errors='ignore')
            df.set_index([groupby_column, 'masterpatientid', 'censusdate', 'facilityid'],inplace=True)
            df = df.unstack(groupby_column)
            df.columns = prefix + '_' + df.columns.get_level_values(1) + df.columns.get_level_values(0)
            df.reset_index(inplace=True)   
            return df
                
        df.set_index(['masterpatientid', event_reported_date_column, event_date_column], inplace=True)

        # Merging with reset dataframe
        if not reset is None:
            assert all([col in reset.names for col in ['masterpatientid', event_date_column, event_reported_date_column]])
            reset = reset.reorder_levels(df.index.names)
            reset_df = pd.DataFrame(index=reset, columns=df.columns)
            df = pd.concat([df] + [reset_df] * n, axis=0)

        # Truncating time from `event_reported_date_column`    
        df.index = pd.MultiIndex.from_arrays(
            [
                df.index.get_level_values('masterpatientid'),
                df.index.get_level_values(event_reported_date_column).normalize(),
                df.index.get_level_values(event_date_column)
            ],
            names=df.index.names
        )
        
        df.sort_index(level=['masterpatientid', event_reported_date_column, event_date_column], inplace=True)
        df.reset_index(inplace=True)
        # At this point the dataframe is sorted by masterpatientid, date (w/o time) when the event was created,
        # the timestamp when the event happened 
        
        df = self.conditional_aggregator(
            census = self.census_df,
            df = df,
            value_headers = value_columns,
            createddate_header = event_reported_date_column,
            effectivedate_header = event_date_column,
            deleteddate_header = event_deleted_date_column,
            result_headers = [f'__LAST_VALUES__{value_column}' for value_column in value_columns],
            agg_func=lambda x: x.values[-n:]
        )
        for value_column in value_columns:
            for i in range(1, n+1):
                col_name = f'{prefix}_{value_column}_{self.humanify_number(i)}_previous_value'
                df[col_name] = df[f'__LAST_VALUES__{value_column}'].apply(
                    lambda x: x[-i] if (x is not None) and (len(x)>=i) and pd.notnull(x[min(-i+1,-1):]).all() else np.nan
                )
        
        df.drop([f'__LAST_VALUES__{value_column}' for value_column in value_columns], axis=1, inplace=True)
        
        # fill NaNs with '__RESET__' to make sure that they are not forward filled later
        df.fillna('__RESET__', inplace=True)
        
        df = df.merge(
            self.census_df[['masterpatientid', 'censusdate']],
            on = ['masterpatientid', 'censusdate'],
            how='outer',
        )
        
        df.sort_values(['masterpatientid', 'censusdate'], inplace=True)
        
        grouped = df.groupby('masterpatientid')
        for col in [c for c in df.columns if c != 'masterpatientid']:
            df[col] = grouped[col].ffill()
        
        df.replace('__RESET__', np.nan, inplace=True)
        
        return self.census_df[['masterpatientid', 'censusdate']].merge(
            df,
            how='left',
            on=['masterpatientid', 'censusdate']
        )   

    @staticmethod
    def conditional_aggregator(
        *,
        census: pd.DataFrame,
        df: pd.DataFrame,
        value_headers: list,
        effectivedate_header: str,
        createddate_header: str,
        deleteddate_header: Union[str, None],
        result_headers: Union[list, None] = None,
        window_size: Union[int, None] = None,
        agg_func: Union[Callable, str] = np.mean
    ) -> pd.DataFrame:
        """
        Generates the rolling aggreagation of event values, considering the date
        the event was recorded and (if necessary) the date the event was deleted.
        In other words, the event will not be considered until it's reported and/or
        it's deleted.

        Parameters:
        -----------
            census: dataframe with columns 'masterpatientid', 'censusdate'
            df: dataframe, with columns 'masterpatientid', 'censusdate', effectivedate_header, createddate_header, *value_headers
            value_headers: list of the columns to apply the aggration to
            effectivedate_header: name of the column that specifies when the event actually happpened
            createddate_header: name of the column that specifies when the respective record considered to be created
            deleteddate: name of the column that specifies when the respective record considered to be deleted
            result_headers: list of the names for resulting columns, will be generated automatically if not provided
            windowsize: int, the window size for rolling window in days
            aggregation_method: function or name of the function to pass to `.agg()` method

        Return:
        -------
            df_res : pd.DataFrame
                The dataframe with the columns 'masterpatientid', 'censusdate' and result_headers
                columns containing the rolling window aggregation result
        """

        if result_headers is None:
            result_headers = [f'{value_header}_{agg_func if isinstance(agg_func, str) else agg_func.__name__}_{window_size}_day']
        
        df.sort_values(['masterpatientid', effectivedate_header, createddate_header], inplace=True, ignore_index=True)
        date_cols = [
            col for col in [effectivedate_header, createddate_header, deleteddate_header] if pd.notnull(col)
        ]
        df[date_cols] = df[date_cols].apply(
            lambda x: pd.to_datetime(x).dt.normalize(),
            axis=1
        )
        
        def census_apply_fn(row):
            # get the masterpatientid, censusdate
            masterpatientid = row['masterpatientid']
            censusdate = row['censusdate']
            
            mpid_idx_left = df['masterpatientid'].searchsorted(masterpatientid, side='left')
            mpid_idx_right = df['masterpatientid'].searchsorted(masterpatientid, side='right')
            df_sub = df.iloc[mpid_idx_left:mpid_idx_right]
            
            effective_date_right = df_sub[effectivedate_header].searchsorted(censusdate, side='right')
            if window_size is not None:
                datestart = censusdate - pd.Timedelta(days=window_size-1)
                effective_date_left = df_sub[effectivedate_header].searchsorted(datestart, side='left')
                df_sub = df_sub.iloc[effective_date_left:effective_date_right]
            else:
                df_sub = df_sub.iloc[:effective_date_right]
            df_sub = df_sub[df_sub[createddate_header] <= censusdate]
            
            if deleteddate_header is not None:
                df_sub = df_sub[
                    (df_sub[deleteddate_header].isnull())|(censusdate < df_sub[deleteddate_header])
                ]
            
            if len(df_sub) == 0:
                for header in result_headers:
                    row[header] = None
            else:
                for it, header in enumerate(result_headers):
                    row[header] = df_sub[value_headers[it]].agg(agg_func)
            return row
        
        return census.apply(census_apply_fn, axis=1)

    def conditional_featurizer_all(self, df, groupby_column, event_date='orderdate', 
        event_reported_date='ordercreateddate', prefix=None, sliding_windows=[2,7,14,30]):
            # get count of days since last event
            days_last_event_df = self.conditional_days_since_last_event(
                df=df,
                prefix=prefix,
                event_date_column=event_date,
                event_reported_date_column=event_reported_date,
                groupby_column=groupby_column,
                missing_event_dates='drop'
            )

            cumidx_df = self.conditional_cumsum_features(
                df,
                prefix, 
                event_date,
                event_reported_date,
                groupby_column = groupby_column,
                missing_event_dates='drop',
                cumidx=True,
                sliding_windows=sliding_windows
            )

            # Do cumulative summation on all diagnosis columns
            cumsum_df = self.conditional_cumsum_features(
                df,
                prefix, # not sure what this should be set to
                event_date,
                event_reported_date,
                groupby_column = groupby_column,
                missing_event_dates='drop',
                cumidx=False,
                sliding_windows=sliding_windows
            )

            return days_last_event_df, cumidx_df, cumsum_df

    @staticmethod
    def pivot_patient_date_event_count(df, groupby_column=None, 
                                       date_column='censusdate', prefix=None, fill_value=0,
                                       rename_date=True, aggfunc='sum'):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column]).dt.normalize()

        # prepend each value in groupby_column with prefix + '_' if given
        if prefix:
            df[groupby_column] = prefix + '_' + df[groupby_column].astype(str)

        df['indicator'] = 1
        df = df.pivot_table(index=['masterpatientid', date_column], columns=[groupby_column], 
                            values='indicator', fill_value=fill_value, aggfunc=aggfunc).reset_index()
        
        df.columns.name = None

        if rename_date:
            df.rename(columns={date_column: 'censusdate'}, inplace=True)
        
        return df

    @staticmethod
    def humanify_number(number: int) -> str:
        """Convert the ordinal numbers from integer to the string with respective endings
            For example, 1 -> '1st'; 103 -> '103rd'
        """
        if 10 <= number % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
        return f'{number}{suffix}'