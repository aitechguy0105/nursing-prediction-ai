import gc
import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class RehospFeatures(BaseFeaturizer):
    def __init__(self, census_df, rehosps, training=False):
        super(RehospFeatures, self).__init__()
        self.rehosps_df = rehosps
        self.census_df = census_df
        self.training = training

    def generate_target_features(self, rehosp):
        # next_hosp dataframe is formed from rehosp df where dateoftransfer > censusdate and applying
        # groupby on 'masterpatientid', 'censusdate'. Taking the first row from the group and renaming
        # dateoftransfer to next_hosp_date.
        next_hosp = rehosp[rehosp.dateoftransfer >= rehosp.censusdate].groupby(
            ['masterpatientid', 'censusdate']).head(n=1).loc[:, ['masterpatientid', 'censusdate', 'dateoftransfer']
                    ].rename(columns={'dateoftransfer': 'next_hosp_date'})

        # Check whether paient was re-hospitalised in next 3 or 7 days and create boolean column for the same
        next_hosp['target_3_day_hosp'] = (next_hosp.next_hosp_date - next_hosp.censusdate) <= pd.to_timedelta('4 days')
        next_hosp['target_7_day_hosp'] = (next_hosp.next_hosp_date - next_hosp.censusdate) <= pd.to_timedelta('8 days')

        next_hosp.columns = 'hosp_' + next_hosp.columns

        return next_hosp

    def generate_features(self):
        log_message(message_type='info', message='Rehosp Processing...')
        self.rehosps_df = self.sorter_and_deduper(
            self.rehosps_df,
            sort_keys=['masterpatientid', 'dateoftransfer'],
            unique_keys=['masterpatientid', 'dateoftransfer']
        )
        self.rehosps_df['dateoftransfer'] = pd.to_datetime(self.rehosps_df.dateoftransfer.dt.date)
        rehosp = self.rehosps_df.merge(self.census_df, on=['masterpatientid'])
        last_hosp = rehosp[rehosp.dateoftransfer < rehosp.censusdate].copy()
        last_hosp['count_prior_hosp'] = last_hosp.groupby(
            ['masterpatientid', 'censusdate']).dateoftransfer.cumcount() + 1
        # applying groupby last_hosp on 'masterpatientid', 'censusdate'. Taking the last row
        # from the group and renaming dateoftransfer to last_hosp_date.
        last_hosp = last_hosp.groupby(['masterpatientid', 'censusdate']).tail(
            n=1).loc[:, ['masterpatientid', 'censusdate', 'dateoftransfer', 'count_prior_hosp']].rename(
            columns={'dateoftransfer': 'last_hosp_date'})

        last_hosp['days_since_last_hosp'] = (last_hosp.censusdate - last_hosp.last_hosp_date).dt.days

        last_hosp.columns = 'hosp_' + last_hosp.columns

        next_hosp = self.generate_target_features(rehosp)

        # ===============================Downcast===============================
        last_hosp = self.downcast_dtype(last_hosp)
        next_hosp = self.downcast_dtype(next_hosp)

        final_df = self.census_df.merge(
            last_hosp,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['hosp_masterpatientid', 'hosp_censusdate']
        )

        final_df = final_df.merge(
            next_hosp,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['hosp_masterpatientid', 'hosp_censusdate']
        )

        # df.columns.duplicated() returns a list containing boolean
        # values indicating whether a column is duplicate
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        log_message(message_type='info', message='Created Base 12')

        # drop unwanted columns
        final_df = self.drop_columns(
            final_df,
            'last_hosp_date|next_hosp_date|_masterpatientid|_facilityid|orderdate|createddate|_x$|_y$|bedid|censusactioncode|payername|payercode'
        )
        
        # =============Trigger garbage collection to free-up memory ==================
        del last_hosp
        del next_hosp
        gc.collect()

        # handle NaN by adding na indicators
        log_message(message_type='info', message='Add Na Indicators...')
        final_df = self.add_na_indicators(final_df, self.ignore_columns)
        if 'na_indictator_hosp_target_3_day_hosp' in final_df.columns and 'na_indictator_hosp_target_7_day_hosp' in final_df.columns:
            final_df.drop(['na_indictator_hosp_target_3_day_hosp', 'na_indictator_hosp_target_7_day_hosp'], axis=1, inplace=True)

        assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        return final_df
