import gc
import sys
from datetime import datetime
import time
import pandas as pd
import numpy as np
from eliot import log_message, start_action

from .featurizer import BaseFeaturizer


class DiagnosisFeatures(BaseFeaturizer):
    def __init__(self, census_df, diagnosis, diagnosis_lookup_ccs_s3_file_path, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.diagnoses_df = diagnosis
        self.training = training
        self.diagnosis_lookup_ccs_s3_file_path = diagnosis_lookup_ccs_s3_file_path
        self.config = config
        super(DiagnosisFeatures, self).__init__()

    def generate_features(self):
        """
        - CSV file contains diagnosis mappings with ICD-10 code and categories, label etc
        - Merge these categories into self.diagnoses_df df and use all possible CCS labels as new columns
        - All diagnosis names becomes individaul columns
        - Diagnosis name columns are added to parent df
        """
        with start_action(action_type=f"Diagnosis - generating diagnosis features, diagnosis shape", 
                          diagnoses_df_shape = self.diagnoses_df.shape):
            
            start = time.time()
            self.diagnoses_df = self.sorter_and_deduper(
                self.diagnoses_df,
                sort_keys=['masterpatientid', 'onsetdate', 'diagnosiscode'],
                unique_keys=['masterpatientid', 'onsetdate', 'diagnosiscode']
            )
            generate_na_indicators = self.config.featurization.diagnosis.generate_na_indicators
            lookup_ccs = pd.read_csv(f's3://{self.diagnosis_lookup_ccs_s3_file_path}')
            log_message(
                message_type='info', 
                message=f'Diagnosis - loading diagnosis lookup file. lookup_ccs shape.', 
                lookup_ccs_shape = lookup_ccs.shape
            )
            lookup_ccs.columns = lookup_ccs.columns.str.replace("'", "")
            lookup_ccs = lookup_ccs.apply(lambda x: x.str.replace("'", ""))
            self.diagnoses_df['indicator'] = 1
            self.diagnoses_df['diagnosiscode'] = self.diagnoses_df.diagnosiscode.str.replace('.', '')
            self.diagnoses_df['onsetdate'] = pd.to_datetime(self.diagnoses_df.onsetdate).dt.normalize()

            self.diagnoses_df_merged = self.diagnoses_df.merge(
                lookup_ccs,
                how='left',
                left_on=['diagnosiscode'],
                right_on=['ICD-10-CM CODE']
            )
            self.diagnoses_df_merged['ccs_label'] = self.diagnoses_df_merged['Default CCSR CATEGORY DESCRIPTION IP']
            if self.config.featurization.diagnosis.use_conditional_functions:
                log_message(
                    message_type='info', 
                    message = f'Diagnosis - using conditional functions'
                )
                if self.config.featurization.diagnosis.generate_na_indicators:
                    raise NotImplementedError("Diagnosis - generate_na_indicators not yet implemented for conditional_functions.")
                # get count of days since last event
                days_last_event_df = self.conditional_days_since_last_event(
                    df=self.diagnoses_df_merged,
                    prefix='dx',
                    event_date_column='onsetdate',
                    event_reported_date_column='createddate',
                    groupby_column='ccs_label',
                    missing_event_dates='drop'
                )
                log_message(
                    message_type='info', 
                    message=f'Diagnosis - created count of days since last event features.', 
                    days_last_event_df_shape = days_last_event_df.shape
                )
                # Do cumulative summation on all diagnosis columns
                cumidx_df = self.conditional_cumsum_features(
                    self.diagnoses_df_merged,
                    'dx', # not sure what this should be set to
                    'onsetdate',
                    'createddate',
                    groupby_column = 'ccs_label',
                    missing_event_dates='drop',
                    cumidx=True,
                    sliding_windows=[2,7,14,30]
                )
                log_message(
                    message_type='info', 
                    message=f'Diagnosis - cumulative summation, patient days with any events cumsum.',
                    cumidx_df_shape = cumidx_df.shape
                )
                # Do cumulative summation on all diagnosis columns
                cumsum_df = self.conditional_cumsum_features(
                    self.diagnoses_df_merged,
                    'dx', # not sure what this should be set to
                    'onsetdate',
                    'createddate',
                    groupby_column = 'ccs_label',
                    missing_event_dates='drop',
                    cumidx=False,
                    sliding_windows=[2,7,14,30]
                )
                log_message(
                    message_type='info', 
                    message=f'Diagnosis - cumulative summation, total number of events cumsum.',
                    cumsum_df_shape = cumsum_df.shape
                )
                final_df = self.census_df.merge(cumidx_df, on=['masterpatientid','censusdate'])
                final_df = final_df.merge(cumsum_df, on=['masterpatientid','censusdate'])
                final_df = final_df.merge(days_last_event_df, on=['masterpatientid','censusdate'])

                del days_last_event_df

            else:
                # drop any records where ccs_label is null
                self.diagnoses_df_merged = self.diagnoses_df_merged.dropna(subset=['ccs_label'])
                if self.config.featurization.diagnosis.pivot_aggfunc_sum:
                    aggfunc='sum'
                else:
                    aggfunc='min'

                diagnosis_pivoted = self.pivot_patient_date_event_count(self.diagnoses_df_merged.copy(),
                                        groupby_column='ccs_label', date_column='onsetdate', prefix='dx', fill_value=0,
                                        aggfunc=aggfunc)

                # ===============================Downcast===============================
                diagnosis_pivoted = self.downcast_dtype(diagnosis_pivoted)

                # This merge works only because census_df has all the dates from training_start_date onwards
                events_df = self.census_df.merge(
                    diagnosis_pivoted,
                    how='outer',
                    on=['masterpatientid', 'censusdate'],
                )
                assert events_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

                # =============Delete & Trigger garbage collection to free-up memory ==================
                del diagnosis_pivoted
                gc.collect()

                # handle NaN by adding na indicators

                if generate_na_indicators:
                    log_message(
                        message_type='info', 
                        message='Diagnosis - creating na indicators and handling na values.'
                    )
                    events_df = self.add_na_indicators(events_df, self.ignore_columns)
                cols = [col for col in events_df.columns if col.startswith('dx')]

                ############################################################################################
                # get sub dataframe for all event, the dataframe without na indicator columns 
                events_df_ = events_df[['facilityid','masterpatientid','censusdate']+cols].copy()
                events_df_[cols] = events_df_[cols].fillna(0)
                events_df_ = self.downcast_dtype(events_df_)

                # get count of days since last event
                days_last_event_df = self.apply_n_days_last_event(events_df_, cols)
                log_message(
                    message_type='info', 
                    message=f'Diagnosis - created count of days since last event features.',
                    days_last_event_df_shape = days_last_event_df.shape
                )
                ############################################################################################

                # Do cumulative summation on all diagnosis columns
                cumidx_df = self.get_cumsum_features(cols, events_df, cumidx=True)
                log_message(
                    message_type='info', 
                    message=f'Diagnosis - cumulative summation, patient days with any events cumsum.',
                    cumidx_df_shape = cumidx_df.shape
                )
                cumsum_df = self.get_cumsum_features(cols, events_df_, cumidx=False)
                log_message(
                    message_type='info', 
                    message=f'Diagnosis - cumulative summation, total number of events cumsum.',
                    cumsum_df_shape = cumsum_df.shape
                )
                final_df = self.census_df.merge(cumidx_df, on=['facilityid','masterpatientid','censusdate'])
                final_df = final_df.merge(cumsum_df, on=['facilityid','masterpatientid','censusdate'])
                final_df = final_df.merge(days_last_event_df, on=['masterpatientid','censusdate'])

                del days_last_event_df
                del events_df
                del events_df_

            self.sanity_check(final_df)

            del self.census_df
            log_message(
                message_type = 'info', 
                message = f'Diagnosis - diagnosis features created.', 
                final_dataframe_shape = final_df.shape,
                time_taken=round(time.time()-start,2)
            )
            return final_df, self.diagnoses_df_merged
