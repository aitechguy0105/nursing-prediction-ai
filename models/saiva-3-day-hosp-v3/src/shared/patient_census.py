"""
bedid, censusactioncode, payername, payercode are retained in demographics
featurisation file and dropped from rest of the merges.
"""

import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class PatientCensus(BaseFeaturizer):
    def __init__(self, census_df, train_start_date, test_end_date):
        self.census_df = census_df
        self.train_start_date = train_start_date
        self.prediction_date = test_end_date
        super(PatientCensus, self).__init__()

    def generate_features(self) -> pd.DataFrame:
        """
        desc: 1. Creates base df having censusdate from train_start_date to test_end_date.
              2. merging base with patient_census w.r.t censusdate.
        :return: dataframe
        """
        log_message(message_type='info', message='Create a Base by merging Census')
        self.census_df = self.sorter_and_deduper(
            self.census_df,
            sort_keys=['masterpatientid', 'censusdate'],
            unique_keys=['masterpatientid', 'censusdate']
        )

        base = pd.DataFrame({'censusdate': pd.date_range(
            start=self.train_start_date, end=self.prediction_date)}
        )
        base = base.merge(self.census_df, how='left', on=['censusdate'])
        base = self.downcast_dtype(base)

        # Drop unwanted columns
        base.drop(
            ['beddescription', 'roomratetypedescription', 'carelevelcode', 'patientid'],
            axis=1,
            inplace=True
        )
        # Drop all rows having NaN masterpatientid
        base = base[base['masterpatientid'].notna()]

        # have to have only one row per masterpatientid, censusdate pair
        assert base.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

        return base
