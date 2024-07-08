import gc
import sys
import time
import pandas as pd
import numpy as np
from eliot import log_message, start_action

from .featurizer import BaseFeaturizer


class AdmissionFeatures(BaseFeaturizer):
    def __init__(self, census_df, admissions, training=False):
        super(AdmissionFeatures, self).__init__()
        self.admissions_df = admissions
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.training = training
        
    
    def generate_features(self):
        """
        - last_hospitalisation_date, days_since_last_hosp per masterpatientid & censusdate
        - Boolean indicating re-hospitalised in next 3 or 7 days
        """
        with start_action(action_type=f"Admissions - generating admissions features.", admissions_df_shape = self.admissions_df.shape):
            start = time.time()
            self.admissions_df.sort_values([
                'masterpatientid', 'dateofadmission',
                'admissionstatus', 'to_from_type', 'primaryphysicianid', 'admittedfrom' #added to avoid randomness during deduplication
            ], inplace=True)
            
            self.admissions_df['dateofadmission'] = pd.to_datetime(self.admissions_df.dateofadmission.dt.date)
            # keep only the last admission before the midnight
            self.admissions_df.drop_duplicates(subset=['masterpatientid', 'dateofadmission'], keep='last', inplace=True)
            
            # Merge census into admissions to get a row for each census date
            admissions_census = pd.merge(
                self.census_df, 
                self.admissions_df, 
                left_on = ['masterpatientid','censusdate'],
                right_on = ['masterpatientid','dateofadmission'],
                how='left'
            )
            admissions_census.sort_values(by=['masterpatientid','censusdate'], inplace=True)
            
            # ================================================================================================
            # admissionstatus & to_from_type needs to be populated in daily_prediction table for filtering purpose
            admissions_status = admissions_census[
                ['masterpatientid', 'admissionstatus', 'to_from_type', 'censusdate']].copy()
            admissions_status[['admissionstatus']] = admissions_status.groupby(
                "masterpatientid"
            )[['admissionstatus']].fillna(method="ffill")
            admissions_status[['to_from_type']] = admissions_status.groupby(
                "masterpatientid"
            )[['to_from_type']].fillna(method="ffill")
            admissions_status['to_from_type'].fillna(value='Unknown', inplace=True)
            admissions_status['admissionstatus'].fillna(value='AdmissionStatusUnknown', inplace=True)
            admissions_status['admissionstatus'] = admissions_status['admissionstatus'].astype('category')
            log_message(message_type='info', 
                        message=f'Admissions - admissions_status features created.', 
                        admissions_status_shape = admissions_status.shape
                       )
            # ================================================================================================
            admissions_physician = admissions_census[['masterpatientid', 'primaryphysicianid', 'censusdate']].copy()
            admissions_physician[['primaryphysicianid']] = admissions_physician.groupby("masterpatientid")[
                ['primaryphysicianid']].fillna(method="ffill")
            admissions_physician['primaryphysicianid'] = "primaryphysician_" + \
                            admissions_physician['primaryphysicianid'].replace(np.nan,'unknown').astype(str)
            admissions_physician['primaryphysicianid'] = admissions_physician['primaryphysicianid'].astype('category')
            log_message(
                message_type='info', 
                message=f'Admissions - admissions_physician features created.', admissions_physician_shape = admissions_physician.shape)
            # ================================================================================================
            admissions_admitted_from = admissions_census[['masterpatientid', 'admittedfrom', 'censusdate']].copy()
            admissions_admitted_from[['admittedfrom']] = admissions_admitted_from.groupby("masterpatientid")[
                ['admittedfrom']].fillna(method="ffill")
            admissions_admitted_from['admittedfrom'].fillna(
                value='AdmittedFromUnknown', inplace=True
            )
            admissions_admitted_from['admittedfrom'] = admissions_admitted_from['admittedfrom'].astype('category')
            log_message(
                message_type='info', 
                message=f'Admissions - admissions_admitted_from features created.',
                admissions_admitted_from_shape = admissions_admitted_from.shape
                       )
            # ================================================================================================

            final_df = self.census_df.merge(
                admissions_status,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['masterpatientid', 'censusdate']
            )
            final_df = final_df.merge(
                admissions_physician,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['masterpatientid', 'censusdate']
            )
            final_df = final_df.merge(
                admissions_admitted_from,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['masterpatientid', 'censusdate']
            )
            #TODO: Remove later. Add dateofadmission in the final frame
            final_df = final_df.merge(
                self.admissions_df[['masterpatientid','dateofadmission']], 
                left_on = ['masterpatientid','censusdate'],
                right_on = ['masterpatientid','dateofadmission'],
                how='left'
            )
            # ================================================================================================
            # drop unwanted columns
            # values indicating whether a column is duplicate
            final_df = self.drop_columns(
                final_df,
                '_masterpatientid|_facilityid|_x$|_y$'
            )
            # =============Trigger garbage collection to free-up memory ==================
            del admissions_status
            del admissions_physician
            del admissions_admitted_from
            gc.collect()
            self.sanity_check(final_df)
            log_message(
                message_type='info', 
                message=f'Admissions - Final merged data.', 
                final_df_shape = final_df.shape,
                time_taken=round(time.time() - start, 2)
            )
            return final_df
