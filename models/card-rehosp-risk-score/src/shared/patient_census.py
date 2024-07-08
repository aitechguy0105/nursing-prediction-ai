import sys
import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class PatientCensus(BaseFeaturizer):
    def __init__(self, census_df, start_date, end_date):
        self.census_df = census_df
        self.start_date = start_date
        self.end_date = end_date        
        super(PatientCensus, self).__init__()

    def generate_features(self):
        """
        desc: Get all patients in the census for a given prediction_date
        :return: dataframe
        """
        log_message(message_type='info', message='Create a Base by merging Census')

        self.census_df = self.sorter_and_deduper(
            self.census_df,
            sort_keys=['masterpatientid', 'censusdate'],
            unique_keys=['masterpatientid', 'censusdate']
        )
        
        base = pd.DataFrame({'censusdate': pd.date_range(
            start=self.start_date, end=self.end_date)}
        )
        base = base.merge(self.census_df, how='left', on=['censusdate'])
        
        # Drop unwanted columns
        base.drop(
            ['patientid', 'payercode', 'bedid', 'carelevelcode', 'beddescription', 'roomratetypedescription',
             'payercode'],
            axis=1,
            inplace=True
        )
        
        # Drop all rows having NaN masterpatientid
        base = base[base['masterpatientid'].notna()]

        return base
