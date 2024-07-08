import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer
from datetime import timedelta


class AdmissionFeatures(BaseFeaturizer):
    def __init__(self, census_df, admissions):
        super(AdmissionFeatures, self).__init__()
        self.admissions_df = admissions
        self.census_df = census_df

    def generate_features(self):
        """
        - count_readmissions, admissions_proximity_score per masterpatientid & censusdate
        """
        log_message(message_type='info', message='Admissions Processing...')
        self.admissions_df = self.sorter_and_deduper(
            self.admissions_df,
            sort_keys=['masterpatientid', 'dateofadmission'],
            unique_keys=['masterpatientid', 'dateofadmission']
        )
        self.admissions_df['dateofadmission'] = pd.to_datetime(self.admissions_df.dateofadmission.dt.date)
        # Merge census into admissions to get a row for each census date
        admissions = self.admissions_df.merge(self.census_df, on=['masterpatientid'])

        # Filter for admissions less than census_date & greater than last 180 days from census_date
        condition = (admissions.dateofadmission < admissions.censusdate) & (
                admissions.dateofadmission >= (admissions.censusdate - timedelta(days=180)))

        last_admissions = admissions[condition]

        last_admissions['count_readmissions'] = last_admissions.groupby(
            ['masterpatientid', 'censusdate']).dateofadmission.cumcount() + 1

        # applying groupby last_admissions on 'masterpatientid', 'censusdate'. Taking the last row
        # from the group and renaming dateofadmission to last_admission_date.
        last_admissions = last_admissions.groupby(['masterpatientid', 'censusdate']).tail(
            n=1).loc[:, ['masterpatientid', 'censusdate', 'dateofadmission', 'count_readmissions']].rename(
            columns={'dateofadmission': 'last_admission_date'})
        last_admissions['days_since_last_admission'] = (
                last_admissions.censusdate - last_admissions.last_admission_date).dt.days

        # ===============================Downcast===============================
        last_admissions = self.downcast_dtype(last_admissions)

        final_df = self.census_df.merge(
            last_admissions,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['masterpatientid', 'censusdate']
        )
        # If days_since_last_admission == NULL, mark them as last admitted before 180 days
        final_df['days_since_last_admission'].fillna(181, inplace=True)
        # If count_readmissions == NULL, mark them as no readmissions in last 180 days
        final_df['count_readmissions'].fillna(0, inplace=True)

        # Binning and one-hot-encoding for admissions_days_since_first_admission & admissions_days_since_last_admission
        bin_labels = [7, 6, 5, 4, 3, 1]
        bins = [0, 1, 2, 3, 6, 13, 1000]
        final_df['admissions_proximity_score'] = pd.cut(
            x=final_df["days_since_last_admission"], bins=bins, labels=bin_labels, right=True
        )

        # If count_readmissions > 0, mark the score as 3 or else 0
        final_df['readmissions_score'] = [3 if ele > 0 else 0 for ele in final_df["count_readmissions"]]

        # df.columns.duplicated() returns a list containing boolean
        # values indicating whether a column is duplicate
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        final_df = final_df[
            ['masterpatientid', 'facilityid', 'censusdate', 'admissions_proximity_score', 'readmissions_score']
        ]

        assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        return final_df
