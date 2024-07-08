import gc
import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class LabFeatures(BaseFeaturizer):
    def __init__(self, census_df, labs, training=False):
        self.census_df = census_df
        self.labs = labs
        self.training = training
        super(LabFeatures, self).__init__()

    def generate_features(self):
        """
        desc:
            -Process the Lab related features
    
        :param self.census_df: dataframe
        :param self.labs: dataframe
        :param self.training: bool
        """
        self.labs = self.sorter_and_deduper(
            self.labs,
            sort_keys=['masterpatientid', 'resultdate', 'profiledescription', 'reportdesciption'],
            unique_keys=['masterpatientid', 'resultdate', 'profiledescription', 'reportdesciption']
        )

        if self.training:
            lab_types = (self.labs['profiledescription'].value_counts()[:75].index.tolist())
        else:
            lab_types = (self.labs['profiledescription'].value_counts().index.tolist())

        self.labs = (
            self.labs[
                self.labs['profiledescription'].isin(lab_types)
            ].copy().reset_index()
        )

        # replacing multiple spaces with single space.
        # removing underscores occurences specifically from start and end of the string.
        # replacing spaces with underscores.
        self.labs['profiledescription'] = self.labs['profiledescription'].str.replace(r'\_+', r' ', regex=True).replace(r'\s+', r' ', regex=True).str.strip()
        self.labs['profiledescription'] = self.labs['profiledescription'].str.replace(r' ', r'_')
        
        self.labs['abnormalitydescription'] = self.labs['abnormalitydescription'].str.replace(r'\_+', r' ', regex=True).replace(r'\s+', r' ', regex=True).str.strip()
        self.labs['abnormalitydescription'] = self.labs['abnormalitydescription'].str.replace(r' ', r'_')
        
        self.labs['lab_result']  = self.labs['profiledescription'].astype(str) + "__" + self.labs['abnormalitydescription'].astype(str)

        self.labs = pd.concat(
            [
                self.labs,
                pd.get_dummies(self.labs['lab_result'], prefix='labs_')
            ],
            axis=1,
        )
        lab_cols = [c for c in self.labs.columns if c.startswith("labs__")]
        self.labs['resultdate'] = self.labs['resultdate'].dt.normalize()

        # there will be multiple days per patient - group the lab values by patient, day taking the max()
        #    i.e. organize it back by patient, day
        lab_results_grouped_by_day = self.labs.groupby(['masterpatientid', 'resultdate'], as_index=False)[
            lab_cols].max()
        assert lab_results_grouped_by_day.isna().any(axis=None) == False
        assert lab_results_grouped_by_day.duplicated(subset=['masterpatientid', 'resultdate']).any() == False

        merged_df = self.census_df.merge(
            lab_results_grouped_by_day,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['masterpatientid', 'resultdate']
        )

        # drop unwanted columns
        merged_df = self.drop_columns(
            merged_df,
            'date_of_transfer|_masterpatientid|_facilityid|resultdate|_x$|_y$|bedid|censusactioncode|payername|payercode'
        )

        assert merged_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        log_message(message_type='info', message='Feature processing activity completed')
        # =============Trigger garbage collection to free-up memory ==================
        del self.census_df
        gc.collect()

        # handle NaN by adding na indicators
        log_message(message_type='info', message='Add Na Indicators...')
        merged_df = self.add_na_indicators(merged_df, self.ignore_columns)
        cols = [col for col in merged_df.columns if col.startswith('labs__')]
        # Do cumulative summation on all order columns
        log_message(message_type='info', message='cumulative summation...')
        merged_df = self.get_cumsum_features(cols, merged_df)

        return merged_df
