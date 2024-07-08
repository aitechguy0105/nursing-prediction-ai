import gc
import sys
import time
import pandas as pd
import numpy as np
from eliot import log_message, start_action
from datetime import timedelta

from .featurizer import BaseFeaturizer


class RehospFeatures(BaseFeaturizer):
    def __init__(self, census_df, rehosps, adt_df, config, train_start_date=None, training=False):
        super(RehospFeatures, self).__init__()
        self.rehosps_df = rehosps
        self.adt_df = adt_df
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.training = training
        self.config = config
        if train_start_date:
            self.train_start_date = pd.to_datetime(train_start_date)

    def generate_features(self):
        with start_action(action_type=f"generating rehosps_df features", 
                          rehosps_df_shape = self.rehosps_df.shape):
            start = time.time()
            fill_planned_with_no = self.config.featurization.rehosp.fill_planned_with_no
            if fill_planned_with_no:
                log_message(message_type='info', message='Rehosp - Filling NULL Planned with No.')
                self.rehosps_df['planned'].fillna('No', inplace=True)
                self.adt_df['planned'].fillna('No', inplace=True)
            self.rehosps_df = self.sorter_and_deduper(
                self.rehosps_df,
                sort_keys=['masterpatientid', 'dateoftransfer'],
                unique_keys=['masterpatientid', 'dateoftransfer', 'facilityid', 'planned']
            )
            generate_na_indicators = self.config.featurization.rehosp.generate_na_indicators
            self.rehosps_df['dateoftransfer'] = pd.to_datetime(self.rehosps_df.dateoftransfer.dt.date)
            self.rehosps_df = self.rehosps_df[self.rehosps_df['planned']=='No']
            self.rehosps_df = self.rehosps_df[~self.rehosps_df.duplicated(subset=['masterpatientid', 'dateoftransfer'])]

            self.adt_df.sort_values(['masterpatientid', 'begineffectivedate'], inplace=True)
            self.adt_df.reset_index(drop=True, inplace=True)
            # for those consecutive rth in the patient census log, we only keep the first one
            rehosp = self.adt_df.copy() 
            rehosp.loc[:,'dateoftransfer_shift']=rehosp.groupby(['facilityid', 'patientid']).dateoftransfer.shift(periods=1)
            rehosp = rehosp[(rehosp.dateoftransfer_shift.isnull()) & (~rehosp.dateoftransfer.isnull())]

            rehosp = rehosp.loc[(rehosp.dateoftransfer<=self.census_df.censusdate.max())&(rehosp['planned']=='No'),
                                     ['masterpatientid', 'dateoftransfer']]             
            rehosp['dateoftransfer'] = pd.to_datetime(rehosp['dateoftransfer'].dt.date)
            rehosp['censusdate'] = rehosp['dateoftransfer']
            rehosp.drop_duplicates(subset=['masterpatientid', 'censusdate'], inplace=True)

            rehosp = self.census_df[['masterpatientid', 'censusdate']].merge(rehosp, 
                             on = ['masterpatientid', 'censusdate'],
                             how='outer')
            assert rehosp.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
            ##############################################Get UPT Data####################################################

            if self.training: # We don't need to create a target during prediction
                # Get RTH in Date Range
                log_message(message_type='info', message='Rehosp - including death as target during training.')
                rehosps_df = self.rehosps_df[(self.rehosps_df.dateoftransfer>=self.train_start_date)]
                rehosps = rehosps_df[['facilityid','masterpatientid', 'dateoftransfer']]
                rehosps = rehosps[~rehosps[['facilityid', 'masterpatientid', 'dateoftransfer']].duplicated()]

                # GET Death in Date Range
                census = self.census_df.copy()
                census['censusdate'] = pd.to_datetime(census['censusdate']) + timedelta(days=1)
                adt_df = self.adt_df[(self.adt_df.begineffectivedate>=self.train_start_date)]
                death = adt_df.loc[adt_df['actiontype']=='Death', ['facilityid', 'masterpatientid', 'begineffectivedate']]
                death['begineffectivedate'] = pd.to_datetime(death['begineffectivedate'].dt.date)
                death = death.merge(census,
                                    left_on=['facilityid', 'masterpatientid', 'begineffectivedate'],
                                    right_on=['facilityid', 'masterpatientid', 'censusdate'])
                death.drop(columns='censusdate', inplace=True)
                del census

                # Combine RTH and Death to UPT Dataframe 
                UPT = rehosps.merge(death, 
                                   left_on = ['facilityid', 'masterpatientid', 'dateoftransfer'],
                                   right_on = ['facilityid', 'masterpatientid', 'begineffectivedate'],
                                   how = 'outer')

                # Create a upt date column, which is transfer date and death date combined
                UPT['incidentdate'] = UPT['dateoftransfer'].fillna(UPT['begineffectivedate'])

                UPT = UPT[['masterpatientid', 'incidentdate']]

                next_hosp = self.get_target(UPT, self.census_df.copy(), 'model_upt', numdays=3)
                next_hosp = self.downcast_dtype(next_hosp)
            ############################################################################################################

            rehosp.sort_values(['masterpatientid', 'censusdate'], inplace=True)
            # get the number of prior hosp
            rehosp['transfered'] = ~rehosp['dateoftransfer'].isnull()
            rehosp['hosp_count_prior_hosp']=rehosp.groupby(['masterpatientid'])['transfered'].cumsum()
            # get #days since last hosp
            rehosp['hosp_days_since_last_hosp'] = (rehosp['censusdate'] - \
                                              pd.to_datetime(rehosp.groupby(['masterpatientid'])\
                                                             .dateoftransfer.ffill())).dt.days
            rehosp.drop(columns=['dateoftransfer', 'transfered'], inplace=True)

            # ===============================Downcast===============================
            rehosp = self.downcast_dtype(rehosp)
            final_df = self.census_df.merge(rehosp, on=['masterpatientid', 'censusdate'], how='left')

            # df.columns.duplicated() returns a list containing boolean
            # values indicating whether a column is duplicate
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]

            # drop unwanted columns
            final_df = self.drop_columns(
                final_df,
               'hosp_censusdate|last_hosp_date|next_hosp_date|_masterpatientid|_facilityid|orderdate|createddate'
            )

            # =============Trigger garbage collection to free-up memory ==================
            del rehosp
            gc.collect()

            # handle NaN by adding na indicators

            if generate_na_indicators:
                log_message(
                    message_type='info', 
                    message=f'Rehosp - creating na indicators and handling na values.', 
                    final_df_shape = final_df.shape
                )
                final_df = self.add_na_indicators(final_df, self.ignore_columns)

            if self.training:
                final_df = final_df.merge(
                    next_hosp,
                    how='left',
                    on=['masterpatientid', 'censusdate'],
                )
                del next_hosp

            final_df['hosp_count_prior_hosp'].fillna(0, inplace=True)
            final_df['hosp_days_since_last_hosp'].fillna(9999, inplace=True)

            assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
            log_message(
                message_type='info', 
                message=f'Rehosp - exiting rehosp', 
                final_df_shape=final_df.shape,
                time_taken=round(time.time() - start, 2)
            )
            return final_df