import gc
import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class MedFeatures(BaseFeaturizer):
    def __init__(self, census_df, meds, training=False):
        self.census_df = census_df
        self.meds = meds
        self.training = training
        super(MedFeatures, self).__init__()

    def generate_features(self):
        """
        - gpiclass & gpisubclassdescription columns are extracted
          and added to parent df
        """
        log_message(message_type='info', message='Meds Processing...')
        self.meds = self.sorter_and_deduper(
            self.meds,
            sort_keys=['masterpatientid', 'orderdate', 'gpiclass', 'gpisubclassdescription'],
            unique_keys=['masterpatientid', 'orderdate', 'gpiclass', 'gpisubclassdescription']
        )

        # copy corresponding gpiclass value for all None gpisubclassdescription
        # gpisubclassdescription is the actual medication name which will be one hot encoded
        self.meds.loc[self.meds.gpisubclassdescription.isna(), 'gpisubclassdescription'] = self.meds.loc[
            self.meds.gpisubclassdescription.isna(), 'gpiclass']

        self.meds['orderdate'] = self.meds.orderdate.dt.date
        self.meds['indicator'] = 1
        meds_pivoted = self.meds.loc[:,
                       ['masterpatientid', 'orderdate', 'gpisubclassdescription', 'indicator']].pivot_table(
            index=['masterpatientid', 'orderdate'],
            columns=['gpisubclassdescription'],
            values='indicator',
            fill_value=0).reset_index()

        # Add med_ to all column names
        meds_pivoted.columns = 'med_' + meds_pivoted.columns

        meds_pivoted = meds_pivoted.drop_duplicates(
            subset=['med_masterpatientid', 'med_orderdate']
        )

        meds_pivoted['med_orderdate'] = pd.to_datetime(meds_pivoted.med_orderdate)

        # ===============================Downcast===============================
        meds_pivoted = self.downcast_dtype(meds_pivoted)

        final_df = self.census_df.merge(
            meds_pivoted,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['med_masterpatientid', 'med_orderdate']
        )
        
        # drop unwanted columns
        final_df = self.drop_columns(
            final_df,
            'date_of_transfer|_masterpatientid|discontinueddate|MAREndDate|_facilityid|orderdate|_x$|_y$|bedid|censusactioncode|payername|payercode'
        )
        
        # =============Delete & Trigger garbage collection to free-up memory ==================
        del self.census_df
        del meds_pivoted
        gc.collect()

        assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        
        # handle NaN by adding na indicators
        log_message(message_type='info', message='Add Na Indicators...')        
        final_df = self.add_na_indicators(final_df, self.ignore_columns)
        cols = [col for col in final_df.columns if col.startswith('med_')]
        # Do cumulative summation on all med columns
        log_message(message_type='info', message='cumulative summation...')        
        final_df = self.get_cumsum_features(cols, final_df)
        
        return final_df, self.meds