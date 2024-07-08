import gc
import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class AdmissionFeatures(BaseFeaturizer):
    def __init__(self, census_df, admissions, training=False):
        super(AdmissionFeatures, self).__init__()
        self.admissions_df = admissions
        self.census_df = census_df
        self.training = training

    def generate_features(self):
        """
        - last_hospitalisation_date, days_since_last_hosp per masterpatientid & censusdate
        - Boolean indicating re-hospitalised in next 3 or 7 days
        """
        log_message(message_type='info', message='Admissions Processing...')
        self.admissions_df['dateofadmission'] = pd.to_datetime(self.admissions_df.dateofadmission.dt.date)
        self.admissions_df = self.sorter_and_deduper(
            self.admissions_df,
            sort_keys=['masterpatientid', 'dateofadmission'],
            unique_keys=['masterpatientid', 'dateofadmission']
        )
        # Merge census into admissions to get a row for each census date
        # admissions_census = self.census_df.merge(self.admissions_df, how='left', on=['masterpatientid',''])
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
#         admissions_status = admissions_census[
#             ['masterpatientid', 'admissionstatus', 'to_from_type', 'censusdate']].copy()
#         admissions_status[['admissionstatus']] = admissions_status.groupby(
#             "masterpatientid"
#         )[['admissionstatus']].fillna(method="ffill")
#         admissions_status[['to_from_type']] = admissions_status.groupby(
#             "masterpatientid"
#         )[['to_from_type']].fillna(method="ffill")
#         admissions_status['to_from_type'].fillna(value='Unknown', inplace=True)
#         admissions_status['admissionstatus'].fillna(value='AdmissionStatusUnknown', inplace=True)
        
#         # admissionstatus is one-hot-encoded & also retained single column for populating it in daily_prediction table
#         admissions_status['admissionstatus_duplicate'] = admissions_status['admissionstatus']
#         admissions_status.loc[:, 'indicator'] = 1
#         admissions_pivoted = admissions_status.loc[:,
#                              ['masterpatientid', 'censusdate', 'admissionstatus', 'to_from_type',
#                               'admissionstatus_duplicate', 'indicator']].pivot_table(
#             index=['masterpatientid', 'censusdate', 'admissionstatus', 'to_from_type'],
#             columns=['admissionstatus_duplicate'],
#             values='indicator',
#             fill_value=0
#         ).reset_index()
#         admissions_pivoted.columns = 'admissions_' + admissions_pivoted.columns
#         admissions_pivoted.columns = admissions_pivoted.columns.str.replace(' ', '_')
        
#         admissions_pivoted = admissions_pivoted.rename(
#             columns={'admissions_to_from_type': 'to_from_type', 'admissions_admissionstatus':'admissionstatus'})
        
        # ================================================================================================
#         admissions_physician = admissions_census[['masterpatientid', 'primaryphysicianid', 'censusdate']].copy()
#         admissions_physician[['primaryphysicianid']] = admissions_physician.groupby("masterpatientid")[
#             ['primaryphysicianid']].fillna(method="ffill")
#         admissions_physician['primaryphysicianid'].fillna(value='AttendingPhysicianUnknown', inplace=True)
#         admissions_physician['primaryphysicianid'] = 'primaryphysicianid_' + admissions_physician[
#             'primaryphysicianid'].astype(str)
#         admissions_physician.loc[:, 'indicator'] = 1
#         admissions_physician_pivoted = admissions_physician.loc[:,
#                                        ['masterpatientid', 'censusdate', 'primaryphysicianid',
#                                         'indicator']].pivot_table(
#             index=['masterpatientid', 'censusdate'],
#             columns=['primaryphysicianid'],
#             values='indicator',
#             fill_value=0
#         ).reset_index()
#         admissions_physician_pivoted.columns = 'admissions_' + admissions_physician_pivoted.columns

#         # ================================================================================================
#         admissions_admitted_from = admissions_census[['masterpatientid', 'admittedfrom', 'censusdate']].copy()
#         admissions_admitted_from[['admittedfrom']] = admissions_admitted_from.groupby("masterpatientid")[
#             ['admittedfrom']].fillna(method="ffill")
#         admissions_admitted_from['admittedfrom'].fillna(
#             value='AdmittedFromUnknown', inplace=True
#         )
#         admissions_admitted_from['admittedfrom'] = 'admittedfrom_' + admissions_admitted_from['admittedfrom'].astype(
#             str)
#         admissions_admitted_from.loc[:, 'indicator'] = 1
#         admissions_admitted_from_pivoted = admissions_admitted_from.loc[:,
#                                            ['masterpatientid', 'censusdate', 'admittedfrom', 'indicator']].pivot_table(
#             index=['masterpatientid', 'censusdate'],
#             columns=['admittedfrom'],
#             values='indicator',
#             fill_value=0
#         ).reset_index()
#         admissions_admitted_from_pivoted.columns = 'admissions_' + admissions_admitted_from_pivoted.columns

        # ================================================================================================
        admissions = self.admissions_df.merge(self.census_df, on=['masterpatientid'])
        filtered_admissions = admissions[admissions.dateofadmission < admissions.censusdate]
        # applying groupby last_admissions on 'masterpatientid', 'censusdate'. Taking the last row
        # from the group and renaming dateofadmission to last_admission_date.
        last_admissions = filtered_admissions.groupby(['masterpatientid', 'censusdate']).tail(
            n=1).loc[:, ['masterpatientid', 'censusdate', 'dateofadmission']].rename(
            columns={'dateofadmission': 'last_admission_date'})
        last_admissions['days_since_last_admission'] = (
                last_admissions.censusdate - last_admissions.last_admission_date).dt.days
        last_admissions.columns = 'admissions_' + last_admissions.columns
        # ================================================================================================
        first_admissions = filtered_admissions.groupby(['masterpatientid', 'censusdate']).head(
            n=1).loc[:, ['masterpatientid', 'censusdate', 'dateofadmission', ]].rename(
            columns={'dateofadmission': 'first_admission_date'})
        first_admissions['days_since_first_admission'] = (
                first_admissions.censusdate - first_admissions.first_admission_date).dt.days
        first_admissions.columns = 'admissions_' + first_admissions.columns
        # ==========================================Downcast=============================================
        last_admissions = self.downcast_dtype(last_admissions)
        first_admissions = self.downcast_dtype(first_admissions)

        # ================================================================================================
        final_df = self.census_df.merge(
            last_admissions,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['admissions_masterpatientid', 'admissions_censusdate']
        )
        final_df = final_df.merge(
            first_admissions,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['admissions_masterpatientid', 'admissions_censusdate']
        )
#         final_df = final_df.merge(
#             admissions_pivoted,
#             how='left',
#             left_on=['masterpatientid', 'censusdate'],
#             right_on=['admissions_masterpatientid', 'admissions_censusdate']
#         )
        final_df = self.drop_columns(final_df, '_x$|_y$')
#         final_df = final_df.merge(
#             admissions_physician_pivoted,
#             how='left',
#             left_on=['masterpatientid', 'censusdate'],
#             right_on=['admissions_masterpatientid', 'admissions_censusdate']
#         )
#         final_df = final_df.merge(
#             admissions_admitted_from_pivoted,
#             how='left',
#             left_on=['masterpatientid', 'censusdate'],
#             right_on=['admissions_masterpatientid', 'admissions_censusdate']
#         )
        #TODO: Remove later. Add dateofadmission in the final frame
        final_df = final_df.merge(
            self.admissions_df[['masterpatientid','dateofadmission']], 
            left_on = ['masterpatientid','censusdate'],
            right_on = ['masterpatientid','dateofadmission'],
            how='left'
        )
        # ================================================================================================
        # Add na indicator columns for admissions_days_since_first_admission & admissions_days_since_last_admission
        cols = ['admissions_days_since_first_admission', 'admissions_days_since_last_admission']
        df_with_na_indicators = final_df[cols].isnull().astype(int).add_suffix('_na_indicator')
        final_df = pd.concat([final_df, df_with_na_indicators], axis=1)
        # ================================================================================================
        # Fill na with group by mean for admissions_days_since_first_admission and admissions_days_since_last_admission columns
        final_df['admissions_days_since_first_admission'] = final_df[
            'admissions_days_since_first_admission'
        ].fillna(final_df.groupby('facilityid')['admissions_days_since_first_admission'].transform('mean'))
        final_df['admissions_days_since_last_admission'] = final_df[
            'admissions_days_since_last_admission'
        ].fillna(final_df.groupby('facilityid')['admissions_days_since_last_admission'].transform('mean'))

        # ================================================================================================
        # Binning and one-hot-encoding for admissions_days_since_first_admission & admissions_days_since_last_admission
        bin_labels = [6, 5, 4, 3, 2, 1, 0]
        bins = [0, 1, 8, 14, 30, 60, 100, 100000]
        final_df['admissions_days_since_first_admission_bins'] = pd.cut(
            x=final_df["admissions_days_since_first_admission"], bins=bins, labels=bin_labels, right=True
        )
        final_df['admissions_days_since_last_admission_bins'] = pd.cut(
            x=final_df["admissions_days_since_last_admission"], bins=bins, labels=bin_labels, right=True
        )
        
        #TODO: ADD later. Convert the category column to int type
#         final_df['admissions_days_since_first_admission_bins'] = pd.to_numeric(final_df['admissions_days_since_first_admission_bins'], errors='coerce', downcast='integer') 
#         final_df['admissions_days_since_last_admission_bins'] = pd.to_numeric(final_df['admissions_days_since_last_admission_bins'], errors='coerce', downcast='integer') 
#         # Fill 0 for any patients who dont have prior hospitalisations
#         final_df.admissions_days_since_first_admission_bins.fillna(0, inplace = True)
#         final_df.admissions_days_since_last_admission_bins.fillna(0, inplace = True)
        
        # ================================================================================================
        # df.columns.duplicated() returns a list containing boolean
        # values indicating whether a column is duplicate
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        log_message(message_type='info', message='Created Base 12')
        # ================================================================================================
        # drop unwanted columns
        final_df = self.drop_columns(
            final_df,
            'date_of_transfer|_masterpatientid|_facilityid|_x$|_y$|bedid|censusactioncode|payername|payercode|'+
            'admissions_censusdate|admissions_last_admission_date|admissions_first_admission_date'
        )
        # =============Trigger garbage collection to free-up memory ==================
        del last_admissions
        del first_admissions
#         del admissions_pivoted
#         del admissions_physician_pivoted
#         del admissions_admitted_from_pivoted
        gc.collect()

        assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        return final_df
