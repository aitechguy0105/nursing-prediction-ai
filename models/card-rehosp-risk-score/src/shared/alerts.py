import sys
from datetime import timedelta

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class AlertFeatures(BaseFeaturizer):
    def __init__(self, census_df, alerts):
        self.census_df = census_df
        self.alerts = alerts
        super(AlertFeatures, self).__init__()

    def calculate_points(self, row):
        if row.alerts_count in (1, 2):
            return 1
        elif row.alerts_count in (3, 4):
            return 2
        elif row.alerts_count in (5, 6):
            return 3
        elif row.alerts_count >= 7:
            return 4
        elif row.alerts_count == 0:
            return 0

    def generate_features(self):
        """
            - alertdescription values are made as columns
            - For each type of `triggereditemtype` create column indicating its count for a
              given masterpatientid & createddate
            """
        log_message(message_type='info', message='Alerts Processing...')

        self.alerts = self.sorter_and_deduper(
            self.alerts,
            sort_keys=['masterpatientid', 'createddate'],
            unique_keys=['masterpatientid', 'createddate', 'alertdescription']
        )

        self.alerts['createddate'] = pd.to_datetime(self.alerts.createddate.dt.date)
        # Merge census into alerts to get a row for each census date
        alerts_df = self.alerts.merge(self.census_df, on=['masterpatientid'])

        # Filter for alerts less than census_date & greater than last 3 days (72 hrs) from census_date
        condition = (alerts_df.createddate < alerts_df.censusdate) & (
                alerts_df.createddate >= (alerts_df.censusdate - timedelta(days=3)))

        last_alerts_df = alerts_df[condition]

        # Do Cumulative summation and pick the last row to get the overall alerts count for last 3 days
        last_alerts_df['alerts_count'] = last_alerts_df.groupby(
            ['masterpatientid', 'censusdate']).createddate.cumcount() + 1
        last_alerts_df.drop_duplicates(subset=['masterpatientid', 'censusdate'], keep='last', inplace=True)

        # Merge this with overall census 
        final_df = self.census_df.merge(
            last_alerts_df,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['masterpatientid', 'censusdate']
        )
        # Fill all Nan with 0
        final_df['alerts_count'].fillna(0, inplace=True)

        # Calculate the score based on alerts_count
        final_df['alert_score'] = final_df.apply(self.calculate_points, axis=1)

        return final_df[['masterpatientid', 'facilityid', 'censusdate', 'alert_score']]
