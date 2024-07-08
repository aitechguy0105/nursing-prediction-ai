import gc
import sys
import time
import pandas as pd
import numpy as np
from eliot import log_message, start_action
import functools 

from .featurizer import BaseFeaturizer
        
class MedFeatures(BaseFeaturizer):
    def __init__(self, census_df, meds, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.meds = meds
        self.training = training
        self.config = config
        super(MedFeatures, self).__init__()

    def conditional_generate_features(self):
        
        self.meds['gpisubclassdescription'] = self.meds['gpisubclassdescription'].str.lower()
        df_gpisub = self.meds[['masterpatientid', 'orderdate', 'ordercreateddate', 'gpisubclassdescription']]
        days_last_event_df_sub, cuminx_df_sub, cumsum_df_sub =  self.conditional_featurizer_all(df_gpisub, 'gpisubclassdescription', prefix='med_gpisub')

        self.meds['gpiclassdescription'] = self.meds['gpiclassdescription'].str.lower()
        df_gpiclass = self.meds[['masterpatientid', 'orderdate', 'ordercreateddate', 'gpiclassdescription']]
        days_last_event_df, cuminx_df, cumsum_df =  self.conditional_featurizer_all(df_gpiclass, 'gpiclassdescription', prefix='med_gpi')

        dfs_list = [self.census_df, days_last_event_df_sub, cuminx_df_sub, cumsum_df_sub, days_last_event_df, cuminx_df, cumsum_df]

        final_df = functools.reduce(
            lambda left, right: pd.merge(left, right, on=['masterpatientid', 'censusdate'])
            , dfs_list)

        assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        
        del days_last_event_df, cuminx_df, cumsum_df, days_last_event_df_sub, cuminx_df_sub, cumsum_df_sub
        self.sanity_check(final_df)
        
        return final_df, self.meds

    def generate_features(self):
        """
        - gpiclass & gpisubclassdescription columns are extracted
          and added to parent df
        """
        with start_action(action_type=f"Meds - generating meds features", meds_shape = self.meds.shape):
            start = time.time()
            self.meds = self.sorter_and_deduper(
                self.meds,
                sort_keys=['masterpatientid', 'orderdate', 'gpiclass', 'gpisubclassdescription'],
                unique_keys=['masterpatientid', 'orderdate', 'gpiclass', 'gpisubclassdescription']
            )
            generate_na_indicators = self.config.featurization.meds.generate_na_indicators
            # copy corresponding gpiclass value for all None gpisubclassdescription
            # gpisubclassdescription is the actual medication name which will be one hot encoded

            self.meds.loc[self.meds.gpisubclassdescription.isna(), 'gpisubclassdescription'] = self.meds.loc[
                self.meds.gpisubclassdescription.isna(), 'gpiclass']

            # use conditional functions if specified in config
            if self.config.featurization.meds.use_conditional_functions:
                if self.config.featurization.meds.generate_na_indicators:
                    raise NotImplementedError("generate_na_indicators not yet implemented for conditional_functions.")
                return self.conditional_generate_features()

            self.meds['gpisubclassdescription'] = self.meds['gpisubclassdescription'].str.lower()

            meds_pivoted = self.pivot_patient_date_event_count(self.meds,
                            groupby_column='gpisubclassdescription', date_column='orderdate', prefix='med_gpisub', 
                            fill_value=0)
            log_message(
                message_type='info', 
                message=f'Meds - created features from gpisubclassdescription.',
                meds_pivoted_shape = meds_pivoted.shape
            )
            #get gpiclassdescription pivoted table
            self.meds['gpiclassdescription'] = self.meds['gpiclassdescription'].str.lower()

            med_gpiclass_pivoted = self.pivot_patient_date_event_count(self.meds,
                    groupby_column='gpiclassdescription', date_column='orderdate', prefix='med_gpi', 
                    fill_value=0)
            log_message(
                message_type='info', 
                message=f'Meds - created features from gpiclassdescription.',
                med_gpiclass_pivoted_shape = meds_pivoted.shape
            )
            # ===============================Downcast===============================
            meds_pivoted = self.downcast_dtype(meds_pivoted)
            med_gpiclass_pivoted = self.downcast_dtype(med_gpiclass_pivoted)

            event_df = self.census_df.merge(
                meds_pivoted,
                how='outer',
                on=['masterpatientid', 'censusdate']
            )

            event_df = event_df.merge(
                med_gpiclass_pivoted,
                how='outer',
                on=['masterpatientid', 'censusdate']
            )
            assert event_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

            # drop unwanted columns
            event_df = self.drop_columns(
                event_df,
                 '_masterpatientid|discontinueddate|MAREndDate|_facilityid'
            )

            # =============Delete & Trigger garbage collection to free-up memory ==================
            del meds_pivoted
            del med_gpiclass_pivoted
            gc.collect()

            # handle NaN by adding na indicators

            if generate_na_indicators:
                log_message(
                    message_type='info', 
                    message='Meds - creating na indicators and handling na values.'
                )
                event_df = self.add_na_indicators(event_df, self.ignore_columns)
            cols = [col for col in event_df.columns if col.startswith('med_')]

            ###########################################################################
            # preparing data
            event_df = event_df.sort_values(['facilityid','masterpatientid','censusdate'])
            # get sub dataframe of the event_df, romoving all na indicator columns
            event_df_ = event_df[['facilityid','masterpatientid','censusdate'] + cols]
            event_df_[cols] = event_df_[cols].fillna(0)

            event_df_ = self.downcast_dtype(event_df_)

            # get counts of days since last event for all events
            log_message(
                message_type='info', 
                message=f'Meds - creating days since last event feature.'
            )
            days_last_event_df = self.apply_n_days_last_event(event_df_, cols)

            ###########################################################################

            # Do cumulative summation on all med columns
            log_message(
                message_type='info', 
                message='Meds - cumulative summation, patient days with any events cumsum.'
            )         
            cumidx_df = self.get_cumsum_features(cols, event_df, cumidx=True)

            log_message(
                message_type='info', 
                message='Meds - cumulative summation, total number of events cumsum.'
            ) 
            cumsum_df = self.get_cumsum_features(cols, event_df_, cumidx=False)

            final_df = self.census_df.merge(cumidx_df, on=['facilityid','masterpatientid','censusdate'])
            final_df = final_df.merge(cumsum_df, on=['facilityid','masterpatientid','censusdate'])
            final_df = final_df.merge(days_last_event_df, on=['masterpatientid','censusdate'])

            assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

            del event_df, event_df_, cumidx_df, cumsum_df, days_last_event_df
            self.sanity_check(final_df)
            log_message(
                message_type = 'info', 
                message = f'Meds - medication features created.',
                final_dataframe_shape = final_df.shape,
                time_taken=round(time.time()-start,2)
            )
            return final_df, self.meds