import gc
import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class RehospFeatures(BaseFeaturizer):
    def __init__(self, census_df, rehosps):
        super(RehospFeatures, self).__init__()
        self.rehosps_df = rehosps
        self.census_df = census_df

    def generate_features(self):
        log_message(message_type='info', message='Rehosp Processing...')
        self.rehosps_df = self.sorter_and_deduper(
            self.rehosps_df,
            sort_keys=['masterpatientid', 'dateoftransfer'],
            unique_keys=['masterpatientid', 'dateoftransfer']
        )

        self.rehosps_df['dateoftransfer'] = pd.to_datetime(self.rehosps_df.dateoftransfer.dt.date)
        rehosp = self.rehosps_df.merge(self.census_df, on=['masterpatientid'])

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

        final_df = self.census_df.merge(
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
        final_df.drop(
            final_df.columns[final_df.columns.str.contains(
                'last_hosp_date|next_hosp_date|_masterpatientid|_facilityid|orderdate|bedid|createddate|_x$|_y$'
            )].tolist(),
            axis=1,
            inplace=True
        )

        # =============Trigger garbage collection to free-up memory ==================
        del next_hosp
        gc.collect()

        final_df = final_df.fillna(False)

        assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        return final_df[
            ['masterpatientid', 'facilityid', 'censusdate', 'hosp_target_3_day_hosp', 'hosp_target_7_day_hosp']
        ]
