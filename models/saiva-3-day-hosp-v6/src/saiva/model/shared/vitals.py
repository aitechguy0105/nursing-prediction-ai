from .featurizer import BaseFeaturizer
import gc
import sys
from omegaconf import OmegaConf
import pandas as pd
from eliot import log_message, start_action
import numpy as np
import time


class VitalFeatures(BaseFeaturizer):
    def __init__(self, census_df, vitals, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.vitals = vitals
        self.training = training
        self.config = config
        super(VitalFeatures, self).__init__()

    def supporting_vitals_features(self, df):
        
        vtl_nd_mean_diff_use_calendar_days_shift = self.config.featurization.vitals.vtl_nd_mean_diff_use_calendar_days_shift
        
        df['date'] = pd.to_datetime(df['date']).dt.date
        grouped_df = df.groupby(['masterpatientid', 'date', 'vitalsdescription'])[
            'value'].agg({'max', 'min', 'mean'}).reset_index()
        grouped_df['vtl_intra_day_range'] = grouped_df['max'] - \
            grouped_df['min']

        if vtl_nd_mean_diff_use_calendar_days_shift:
            grouped_df['date'] = pd.to_datetime(grouped_df['date'])
            grouped_df.set_index('date', inplace=True)
            grouped_df['1d_prev_value'] = grouped_df.groupby(
                ['masterpatientid', 'vitalsdescription'])['mean'].transform(lambda s: s.shift(1, freq='D'))
            grouped_df['2d_prev_value'] = grouped_df.groupby(
                ['masterpatientid', 'vitalsdescription'])['mean'].transform(lambda s: s.shift(2, freq='D'))
            grouped_df['3d_prev_value'] = grouped_df.groupby(
                ['masterpatientid', 'vitalsdescription'])['mean'].transform(lambda s: s.shift(3, freq='D'))
            grouped_df.reset_index(inplace=True)

        else:
            grouped_df['1d_prev_value'] = grouped_df.groupby(
                ['masterpatientid', 'vitalsdescription'])['mean'].shift(1)
            grouped_df['2d_prev_value'] = grouped_df.groupby(
                ['masterpatientid', 'vitalsdescription'])['mean'].shift(2)
            grouped_df['3d_prev_value'] = grouped_df.groupby(
                ['masterpatientid', 'vitalsdescription'])['mean'].shift(3)

        grouped_df['vtl_1d_mean_diff'] = grouped_df['mean'] - \
            grouped_df['1d_prev_value']
        grouped_df['vtl_2d_mean_diff'] = grouped_df['mean'] - \
            grouped_df['2d_prev_value']
        grouped_df['vtl_3d_mean_diff'] = grouped_df['mean'] - \
            grouped_df['3d_prev_value']

        grouped_df.drop(['min', 'max', 'mean', '1d_prev_value',
                        '2d_prev_value', '3d_prev_value'], axis=1, inplace=True)
        
        grouped_df = grouped_df.pivot(index=['masterpatientid', 'date'],
                                      columns=['vitalsdescription'],
                                      values=[col for col in grouped_df.columns if 'vtl_' in col])
        grouped_df.columns = ['_'.join(col)
                              for col in grouped_df.columns.values]
        grouped_df = grouped_df.reset_index()
        grouped_df['date'] = pd.to_datetime(grouped_df['date'])
        count_df = df.groupby(['masterpatientid', 'date'])['value'].agg(
            {'count'}).reset_index()
        count_df['date'] = pd.to_datetime(count_df['date'])
        count_df.set_index('date', inplace=True)
        count_df['vtl_4d_rolling_count'] = count_df.groupby(['masterpatientid'])['count'].transform(lambda s: s.rolling('4d', min_periods=1).sum())
        count_df['vtl_3d_rolling_count'] = count_df.groupby(['masterpatientid'])['count'].transform(lambda s: s.rolling('3d', min_periods=1).sum())
        count_df['vtl_2d_rolling_count'] = count_df.groupby(['masterpatientid'])['count'].transform(lambda s: s.rolling('2d', min_periods=1).sum())
        count_df.reset_index(inplace=True)
        del count_df['count']

        final_df = grouped_df.merge(
            count_df, on=['masterpatientid', 'date'])
        final_df['date'] = pd.to_datetime(final_df['date'])
        return final_df

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
        vital_type_df = sorted_full_vital_df.loc[sorted_full_vital_df.vitalsdescription == vital_type].copy(
        )

        log_message(message_type='info',
                    vital_type=f'Vitals - Featurizing {vital_type}', vital_type_df_shape = vital_type_df.shape)

        vtl_use_correct_rolling_windows = self.config.featurization.vitals.vtl_use_correct_rolling_windows
        # this change is done due to SAIV-4997. Please read the ticket to understand the code logic.
        if vtl_use_correct_rolling_windows:
            vital_type_df['date'] = pd.to_datetime(vital_type_df['date']).dt.normalize()
        vital_type_df = vital_type_df.set_index('date')
        result_df = pd.DataFrame()  # empty dataframe
        for rolling_days in rolling_window_list:
            days = rolling_days.replace('d', '')
            vital_rolling_df = vital_type_df.groupby(
                vital_type_df.masterpatientid,
                as_index=False)['value'].rolling(
                rolling_days,
                min_periods=1).agg(
                {f'vtl_{vital_type}_{days}_day_count': 'count', f'vtl_{vital_type}_{days}_day_min': 'min',
                 f'vtl_{vital_type}_{days}_day_max': 'max', f'vtl_{vital_type}_{days}_day_mean': 'mean'}
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
        vital_type_df.drop(['vitalsdescription', 'value'],
                           inplace=True, axis=1)
        vital_type_df = self.downcast_dtype(vital_type_df)

        return vital_type_df

    def handle_na_values(self, df):
        vtl_cols = [col for col in df.columns if col not in self.ignore_columns]

        # Forward fill only the vitals columns.
        # Nan vtl_{vital_type}_3_day_count columns will be filled by 0.
        for column in vtl_cols:
            if '_count' in column and column in df.columns:
                df[column].fillna(0, inplace=True)
        if self.config.featurization.vitals.vtl_ffill_na:
            ffilled = df.groupby('masterpatientid')[
                vtl_cols].fillna(method='ffill')
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
        with start_action(action_type=f"Vitals - generating vitals features", vitals_df_shape = self.vitals.shape):
            start = time.time()
            generate_na_indicators = self.config.featurization.vitals.generate_na_indicators


            #get copies of vital_warnings and masterpatient_census data for featuring days since last warning features
            vital_warnings = self.vitals[['masterpatientid', 'date', 'vitalsdescription', 'warnings']].copy()
            masterpatient_census = self.census_df[['masterpatientid', 'censusdate']].copy()

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
                self.vitals.loc[self.vitals.vitalsdescription ==
                                'Temperature', 'value']
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

            vitals = self.vitals.set_index(
                keys=['masterpatientid', 'date']).drop(columns='patientid')
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

            # Drop warnings
            vitals.drop(['warnings'], inplace=True, axis=1)

            # New code
            vitals = vitals.sort_values(by=['masterpatientid', 'date'])
            log_message(
                message_type='info',
                message=f'Vitals - creating rolling window features.', 
                vitals_shape = vitals.shape
            )
            for vital_type in vitals.vitalsdescription.unique():
                if vital_type.lower() == 'height':
                    # don't featurize these vital types!
                    continue

                rolling_df = self.get_vtl_feature_df(
                    vitals, vital_type, ['1d', '2d', '3d'])
                self.census_df = self.census_df.merge(
                    rolling_df,
                    how='left',
                    left_on=['masterpatientid', 'facilityid', 'censusdate'],
                    right_on=['masterpatientid', 'facilityid', 'vtl_date']
                ) 
            log_message(
                message_type='info', 
                message=f'Vitals - creating supporting vital features. vitals shape.',
                vitals_shape = vitals.shape)
            supporting_vital_features_df = self.supporting_vitals_features(vitals)

            self.census_df = self.census_df.merge(
                supporting_vital_features_df,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['masterpatientid', 'date']
            )
            del self.census_df['date']
            # =============Delete & Trigger garbage collection to free-up memory ==================
            del vitals
            del diastolic
            gc.collect()

            # Drop unwanted columns
            self.census_df.drop(
                self.census_df.columns[self.census_df.columns.str.contains(
                    'vtl_date')].tolist(),
                axis=1,
                inplace=True
            )

            # handle NaN by adding na indicators
            if generate_na_indicators:
                log_message(
                    message_type='info', 
                    message='Vitals - creating na indicators and handling na values.'
                )
                self.census_df = self.add_na_indicators(
                    self.census_df, self.ignore_columns)


            # Handle NaN values
            self.census_df = self.handle_na_values(self.census_df)
            
            assert self.census_df.duplicated(
                subset=['masterpatientid', 'censusdate']).any() == False

            #=========================== number of days since last warnings features ===========================
            log_message(
                message_type='info', 
                message=f'Vitals - creating vitals warning features.', 
                vital_warnings_shape = vital_warnings.shape
            )
            vital_warnings = vital_warnings.rename(columns={'date':'censusdate'})
            # preparing data for BP
            BP_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='BP - Systolic')&\
                                       (~vital_warnings.warnings.isnull()), :]
            BP_df['vtl_systolic_high_warning'] = BP_df.warnings.str.contains('Systolic High').astype(int)
            BP_df['vtl_diastolic_high_warning'] = BP_df.warnings.str.contains('Diastolic High').astype(int)
            BP_df['vtl_systolic_low_warning'] = BP_df.warnings.str.contains('Systolic Low').astype(int)
            BP_df['vtl_diastolic_low_warning'] = BP_df.warnings.str.contains('Diastolic Low').astype(int)
            BP_df = BP_df.groupby(['masterpatientid', 'censusdate'])\
                        [['vtl_systolic_high_warning', 'vtl_systolic_low_warning', 'vtl_diastolic_high_warning', 'vtl_diastolic_low_warning']]\
                        .max().reset_index()
            warning_df = masterpatient_census.merge(BP_df,
                                                    on=['masterpatientid', 'censusdate'], how = 'outer')
            del BP_df

            #preparing data for Blood Sugar
            blood_sugar_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Blood Sugar')&\
                                                (~vital_warnings.warnings.isnull()), :]
            blood_sugar_df['vtl_blood_sugar_high_warning'] = blood_sugar_df.warnings.str.contains('High').astype(int)
            blood_sugar_df['vtl_blood_sugar_low_warning'] = blood_sugar_df.warnings.str.contains('Low').astype(int)
            blood_sugar_df = blood_sugar_df.groupby(['masterpatientid', 'censusdate'])\
                            [['vtl_blood_sugar_high_warning', 'vtl_blood_sugar_low_warning']]\
                            .max().reset_index()
            warning_df = warning_df.merge(blood_sugar_df, 
                                          on=['masterpatientid', 'censusdate'], how = 'outer')
            del blood_sugar_df

            #preparing data for O2 sats
            O2_sats_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='O2 sats')&\
                                            (~vital_warnings.warnings.isnull()),:]
            O2_sats_df['vtl_O2_sats_high_warning'] = O2_sats_df.warnings.str.contains('High').astype(int)
            O2_sats_df['vtl_O2_sats_low_warning'] = O2_sats_df.warnings.str.contains('Low').astype(int)
            O2_sats_df = O2_sats_df.groupby(['masterpatientid', 'censusdate'])\
                            [['vtl_O2_sats_high_warning', 'vtl_O2_sats_low_warning']]\
                            .max().reset_index()
            warning_df = warning_df.merge(O2_sats_df, 
                                  on=['masterpatientid', 'censusdate'], how = 'outer')
            del O2_sats_df

            #preparing data for Pain Level
            pain_level_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Pain Level')&\
                                               (~vital_warnings.warnings.isnull()),:]
            pain_level_df['vtl_pain_level_high_warning'] = pain_level_df.warnings.str.contains('High').astype(int)
            pain_level_df = pain_level_df.groupby(['masterpatientid', 'censusdate'])\
                            [['vtl_pain_level_high_warning']]\
                            .max().reset_index()
            warning_df = warning_df.merge(pain_level_df, 
                                  on=['masterpatientid', 'censusdate'], how = 'outer')
            del pain_level_df

            #preparing data for Pulse
            pulse_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Pulse')&\
                                          (~vital_warnings.warnings.isnull()),:]
            pulse_df['vtl_pulse_high_warning'] = pulse_df.warnings.str.contains('High').astype(int)
            pulse_df['vtl_pulse_low_warning'] = pulse_df.warnings.str.contains('Low').astype(int)
            pulse_df = pulse_df.groupby(['masterpatientid', 'censusdate'])\
                            [['vtl_pulse_high_warning', 'vtl_pulse_low_warning']]\
                            .max()\
                            .reset_index()
            warning_df = warning_df.merge(pulse_df, 
                                  on=['masterpatientid', 'censusdate'], how = 'outer')
            del pulse_df

            #preparing data for Respiration
            respiration_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Respiration')&\
                                                (~vital_warnings.warnings.isnull()),:]
            respiration_df['vtl_respiration_high_warning'] = respiration_df.warnings.str.contains('High').astype(int)
            respiration_df['vtl_respiration_low_warning'] = respiration_df.warnings.str.contains('Low').astype(int)
            respiration_df = respiration_df.groupby(['masterpatientid', 'censusdate'])\
                            [['vtl_respiration_high_warning', 'vtl_respiration_low_warning']]\
                        .max().reset_index()
            warning_df = warning_df.merge(respiration_df, 
                                  on=['masterpatientid', 'censusdate'], how = 'outer')
            del respiration_df

            #preparing data for Temperature
            temperature_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Temperature')&\
                                                (~vital_warnings.warnings.isnull()),:]
            temperature_df['vtl_temperature_high_warning'] = temperature_df.warnings.str.contains('High').astype(int)
            temperature_df['vtl_temperature_low_warning'] = temperature_df.warnings.str.contains('Low').astype(int)
            temperature_df = temperature_df.groupby(['masterpatientid', 'censusdate'])\
                            [['vtl_temperature_high_warning', 'vtl_temperature_low_warning']]\
                            .max().reset_index()
            warning_df = warning_df.merge(temperature_df, 
                                  on=['masterpatientid', 'censusdate'], how = 'outer')
            del temperature_df

            #preparing data for Weight warnings
            weight_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Weight')\
                                               &(~vital_warnings.warnings.isnull()), 
                                           ['masterpatientid', 'censusdate']]
            weight_df['vtl_weight_warning'] = 1
            weight_df = weight_df.drop_duplicates().reset_index(drop=True)
            warning_df = warning_df.merge(weight_df, 
                          on=['masterpatientid', 'censusdate'], how = 'outer')
            del weight_df

            #preparing data for Weight + or - warnings
            weight_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Weight')\
                                           &(~vital_warnings.warnings.isnull()), 
                                       ['masterpatientid', 'censusdate', 'warnings']]
            weight_df['vtl_weight_increase_warnings'] = weight_df.warnings.str.contains('\+').astype(int)
            weight_df['vtl_weight_decrease_warnings'] = weight_df.warnings.str.contains('\-').astype(int)
            weight_df = weight_df.groupby(['masterpatientid', 'censusdate'])\
                                        [['vtl_weight_increase_warnings','vtl_weight_decrease_warnings']]\
                                        .max().reset_index()
            warning_df = warning_df.merge(weight_df, 
                          on=['masterpatientid', 'censusdate'], how = 'outer')
            del weight_df

            #preparing data for specific Weight + or - warnings, like '-5.0% change', '-7.5% change', '+10.0% change'
            # use regular expression to extract warning texts
            warning_text = pd.Series(vital_warnings.warnings[(vital_warnings['vitalsdescription']=='Weight')&\
                                 (~vital_warnings.warnings.isnull())]).str\
                                 .replace("[\[].*?[\]]", '', regex=True).str.split(' ; ')

            if len(warning_text)>0:
                warning_text_ = warning_text.apply(pd.Series).stack().reset_index(drop=True)
            else:
                warning_text_ = warning_text.apply(pd.Series).explode().reset_index(drop=True)

            s_unique = warning_text_[warning_text_.str.contains('change')].unique()

            weight_df = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Weight')\
                                           &(~vital_warnings.warnings.isnull()), 
                                           ['masterpatientid', 'censusdate']]
            warnings = vital_warnings.loc[(vital_warnings['vitalsdescription']=='Weight')\
                                          &(~vital_warnings.warnings.isnull()),'warnings']
            #specific Weight warnings featurization
            for text in s_unique:
                name = text.replace(' ', '_')
                weight_df['vtl_weight_'+ name] = warnings.str.contains(text, regex=False).astype(int) 
            weight_df = weight_df.groupby(['masterpatientid', 'censusdate']).max().reset_index()
            warning_df = warning_df.merge(weight_df, 
                          on=['masterpatientid', 'censusdate'], how = 'left')
            assert warning_df.duplicated(subset=['masterpatientid','censusdate']).any() == False
            del weight_df
            cols = warning_df.columns[2:].tolist()
            warning_df[cols] = warning_df[cols].fillna(0)
            warning_df = self.downcast_dtype(warning_df)
            # get count of days since last event
            log_message(
                message_type='info', 
                message=f'Vitals - creating days since last event feature.', 
                warning_df_shape = warning_df.shape
            )
            
            days_last_event_df = self.apply_n_days_last_event(warning_df, cols)

            assert days_last_event_df.duplicated(subset=['masterpatientid','censusdate']).any() == False

            self.census_df = self.census_df.merge(days_last_event_df, on=['masterpatientid', 'censusdate'])
            self.sanity_check(self.census_df)
            del days_last_event_df
            log_message(
                message_type='info', 
                message=f'Vitals - exiting vitals, final dataframe shape.', 
                census_df_shape = self.census_df.shape, 
                time_taken=round(time.time() - start, 2)
            )
            return self.census_df
