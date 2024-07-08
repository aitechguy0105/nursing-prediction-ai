import gc
import sys

import pandas as pd
from eliot import log_message
import numpy as np

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class VitalFeatures(BaseFeaturizer):
    def __init__(self, census_df, vitals, training=False):
        self.census_df = census_df
        self.vitals = vitals
        self.training = training
        super(VitalFeatures, self).__init__()

    def get_vtl_feature_df(self, sorted_full_vital_df, vital_type, rolling_window_list):
        """
        1. set the 'date' as index.
        2. calculate the 'count', 'min', 'max', 'mean' values of vitals in a span of 'rolling_days'.
        3. concat the output, reset the index and return the dataframe.
        Setting `date` as index the pandas calculate rolling window based on date and not on last n records
        ie. given dates : 1, 7, 8, 10 & missing dates :  2,3,4,5,6,9
        3d Rolling window for 8 will consider rows 7 and 8 (as 6 is missing). 
        avg will be calculated as sum of 7, 8 divided by 2 ie. 2 days
        """
        vital_type_df = sorted_full_vital_df.loc[sorted_full_vital_df.vitalsdescription == vital_type].copy()
        
        log_message(message_type='info', vital_type=f'Featurizing {vital_type}...', shape=vital_type_df.shape)

        vital_type_df = vital_type_df.set_index('date')
        result_df = pd.DataFrame()  # empty dataframe
        for rolling_days in rolling_window_list:
            #         start_time = timeit.default_timer()
            vital_rolling_df = vital_type_df.groupby(
                vital_type_df.masterpatientid,
                as_index=False)['value'].rolling(
                rolling_days,
                min_periods=1).agg(
                {f'vtl_{vital_type}_3_day_count': 'count', f'vtl_{vital_type}_3_day_min': 'min',
                 f'vtl_{vital_type}_3_day_max': 'max', f'vtl_{vital_type}_3_day_mean': 'mean'}
            ).reset_index(drop=True)
            
            # concatenate all the rolling windows into one data frame
            result_df = pd.concat([result_df, vital_rolling_df], axis=1)

        # make the 'date' index a column again
        vital_type_df = vital_type_df.reset_index()
        # since they have the same order, you can just concatenate the dataframes by columns
        vital_type_df = pd.concat([vital_type_df, result_df], axis=1)
        
        # convert the datetime to a date
        vital_type_df['vtl_date'] = vital_type_df.pop('date').dt.normalize()

        # keep only the last row per masterpatientid and date combination
        vital_type_df = vital_type_df.drop_duplicates(
            subset=['masterpatientid', 'vtl_date'],
            keep='last'
        )
        vital_type_df.drop(['vitalsdescription', 'value'], inplace=True, axis=1)
        vital_type_df = self.downcast_dtype(vital_type_df)
        
        return vital_type_df

    def handle_na_values(self, df):
        vtl_cols = [col for col in df.columns if col not in self.ignore_columns]

        # Forward fill only the vitals columns
        ffilled = df.groupby('masterpatientid')[vtl_cols].fillna(method='ffill')
        ffilled['masterpatientid'] = df.masterpatientid

        df.loc[:, vtl_cols] = ffilled.loc[:, vtl_cols]
        return df

    def generate_features(self):
        """
        1. Applying sorting and deduping on vitals dataframe.
        2. Masking out all the impossible values of vitals.
        3. Creating another dataframe diastolic, merging it with vitals by making vitals description as 'BP - Diastolic'
        4. Passing each vitals to 'get_vtl_feature_df' function with a rolling window of 3 day.
        5. Removing duplicates and merging all the above dataframes into a single dataframe
        6. Merging vitals with base to form base1 and returning the dataframe.
        :param base: dataframe
        :return: base1 : dataframe
        """
        log_message(message_type='info', message='Vitals Processing...')
        # if client column exists drop it since it has been added to masterpatientid to make it unique
        if 'client' in self.vitals.columns:
            self.vitals = self.vitals.drop(columns=['client'])

        self.vitals = self.sorter_and_deduper(
            self.vitals,
            sort_keys=['masterpatientid', 'date'],
            unique_keys=['masterpatientid', 'date', 'vitalsdescription']
        )
        # diastolic value not in range(30,200) will be made Nan.
        self.vitals.loc[:, 'diastolicvalue'] = (
            self.vitals.loc[:, 'diastolicvalue'].mask(self.vitals.loc[:, 'diastolicvalue'] > 200).mask(
                self.vitals.loc[:, 'diastolicvalue'] < 30).values)
        # BP - Systolic value not in range(40,300) will be made Nan.
        self.vitals.loc[self.vitals.vitalsdescription == 'BP - Systolic', 'value'] = (
            self.vitals.loc[self.vitals.vitalsdescription == 'BP - Systolic', 'value'].mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'BP - Systolic', 'value'] > 300).mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'BP - Systolic', 'value'] < 40).values)
        # Respiration value not in range(6,50) will be made Nan.
        self.vitals.loc[self.vitals.vitalsdescription == 'Respiration', 'value'] = (
            self.vitals.loc[self.vitals.vitalsdescription == 'Respiration', 'value'].mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Respiration', 'value'] > 50).mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Respiration', 'value'] < 6).values)
        # Temperature value not in range(80,200) will be made Nan.
        self.vitals.loc[self.vitals.vitalsdescription == 'Temperature', 'value'] = (
            self.vitals.loc[self.vitals.vitalsdescription == 'Temperature', 'value']
                .mask(self.vitals.loc[self.vitals.vitalsdescription == 'Temperature', 'value'] > 200).mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Temperature', 'value'] < 80).values)
        # Pulse value not in range(20,300) will be made Nan.
        self.vitals.loc[self.vitals.vitalsdescription == 'Pulse', 'value'] = (
            self.vitals.loc[self.vitals.vitalsdescription == 'Pulse', 'value'].mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Pulse', 'value'] > 300).mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Pulse', 'value'] < 20).values)
        # O2 sats value below 80 will be made Nan.
        self.vitals.loc[self.vitals.vitalsdescription == 'O2 sats', 'value'] = (
            self.vitals.loc[self.vitals.vitalsdescription == 'O2 sats', 'value'].mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'O2 sats', 'value'] < 80).values)
        # Weight value not in range(80,660) will be made Nan.
        self.vitals.loc[self.vitals.vitalsdescription == 'Weight', 'value'] = (
            self.vitals.loc[self.vitals.vitalsdescription == 'Weight', 'value'].mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Weight', 'value'] > 660).mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Weight', 'value'] < 80).values)
        # Blood Sugar value not in range(25,450) will be made Nan.
        self.vitals.loc[self.vitals.vitalsdescription == 'Blood Sugar', 'value'] = (
            self.vitals.loc[self.vitals.vitalsdescription == 'Blood Sugar', 'value'].mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Blood Sugar', 'value'] > 450).mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Blood Sugar', 'value'] < 25).values)
        # Pain Level value not in range(0,10) will be made Nan.
        self.vitals.loc[self.vitals.vitalsdescription == 'Pain Level', 'value'] = (
            self.vitals.loc[self.vitals.vitalsdescription == 'Pain Level', 'value'].mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Pain Level', 'value'] > 10).mask(
                self.vitals.loc[self.vitals.vitalsdescription == 'Pain Level', 'value'] < 0).values)
        # drop all rows where we masked out values since we don't know what that value really is
        self.vitals = self.vitals.dropna(subset=['value'], axis=0)

        vitals = self.vitals.set_index(keys=['masterpatientid', 'facilityid', 'date']).drop(columns='patientid')
        # diastolic contains index + diastolicvalue column
        diastolic = vitals.pop('diastolicvalue')
        diastolic = diastolic.dropna()

        vitals = vitals.reset_index()
        diastolic = diastolic.reset_index()
        # add Diastolic values to the vitals
        diastolic = diastolic.rename(columns={"diastolicvalue": "value"})
        diastolic['vitalsdescription'] = 'BP - Diastolic'
        # concatenate the two dataframes by rows
        vitals = pd.concat([vitals, diastolic], axis=0, sort=False)

        # Drop bmi & warnings
        vitals.drop(['bmi', 'warnings'], inplace=True, axis=1)

        ## New code
        vitals = vitals.sort_values(by=['masterpatientid', 'date'])
        for vital_type in vitals.vitalsdescription.unique():
            if vital_type.lower() == 'height':
                # don't featurize these vital types!
                continue

            rolling_df = self.get_vtl_feature_df(vitals, vital_type, ['3d'])
            self.census_df = self.drop_columns(self.census_df, '_x$|_y$')
            self.census_df = self.census_df.merge(
                rolling_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'vtl_date']
            )

        # =============Delete & Trigger garbage collection to free-up memory ==================
        del vitals
        del diastolic
        gc.collect()

        # Drop unwanted columns
        self.census_df.drop(
            self.census_df.columns[self.census_df.columns.str.contains(
                'date_of_transfer|_masterpatientid|_facilityid|vtl_date|_x$|_y$|bedid|censusactioncode|payername|payercode')].tolist(),
            axis=1,
            inplace=True
        )

        # handle NaN by adding na indicators
        self.census_df = self.add_na_indicators(self.census_df, self.ignore_columns)

        # Handle NaN values
        self.census_df = self.handle_na_values(self.census_df)
        assert self.census_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        return self.census_df
