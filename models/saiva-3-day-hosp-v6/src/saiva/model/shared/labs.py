import gc
import sys
import time
import pandas as pd
import numpy as np
from eliot import log_message, start_action
from omegaconf import OmegaConf
import functools 

from .featurizer import BaseFeaturizer


class LabFeatures(BaseFeaturizer):

    def __init__(self, census_df, labs, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.labs = labs
        self.training = training
        self.config = config
        super(LabFeatures, self).__init__()

    def conditional_generate_features(self, lab_event_df):
        # use self.labs to get cumsum and cumidx on 'lab_result' 
        event_date = 'resultdate'
        event_reported_date = 'createddate'
        prefix = 'labs_'
        groupby_column = 'lab_result'

        # self.labs[[event_date, event_reported_date]] = self.labs[[event_date, event_reported_date]].apply(lambda x: pd.to_datetime(x).dt.normalize())

        cuminx_df = self.conditional_cumsum_features(
            self.labs[['masterpatientid', event_date, event_reported_date, groupby_column]],
            prefix, 
            event_date,
            event_reported_date,
            groupby_column = groupby_column,
            missing_event_dates='drop',
            cumidx=True,
        )

        # Do cumulative summation on all diagnosis columns
        cumsum_df = self.conditional_cumsum_features(
            self.labs[['masterpatientid', event_date, event_reported_date, groupby_column]],
            prefix, # not sure what this should be set to
            event_date,
            event_reported_date,
            groupby_column = groupby_column,
            missing_event_dates='drop',
            cumidx=False,
        )

        # using lab_event_df, perform days_since_last on 'profiledescription_level' and 'abnormalityid'
        dfs_list = [self.census_df, cuminx_df, cumsum_df]

        merged_df = functools.reduce(
            lambda left, right: pd.merge(left, right, on=['masterpatientid', 'censusdate'], how='outer')
            , dfs_list)

        del cuminx_df, cumsum_df 
        gc.collect()

        # lab_event_df[[event_date, event_reported_date]] = lab_event_df[[event_date, event_reported_date]].apply(lambda x: pd.to_datetime(x).dt.normalize())

        lab_event_df['abnormalityid'] = lab_event_df['abnormalityid'].astype(str)
        
        # preparing data for profiledescription+abnormalityid
        lab_event_df['profiledescription_level']=lab_event_df['profiledescription']+'_'+lab_event_df['abnormalityid']

        prefix = 'labs'
        # get days since last event on 'profiledescription_level' and merge to result
        merged_df = self.conditional_days_since_last_event(
                df=lab_event_df,
                prefix=prefix,
                event_date_column=event_date,
                event_reported_date_column=event_reported_date,
                groupby_column='profiledescription_level',
                missing_event_dates='drop'
            ).merge(merged_df, on=['masterpatientid','censusdate'])

        
        # get days since last event on 'profiledescription_level' and merge to result
        merged_df = self.conditional_days_since_last_event(
                df=lab_event_df,
                prefix=prefix,
                event_date_column=event_date,
                event_reported_date_column=event_reported_date,
                groupby_column='abnormalityid',
                missing_event_dates='drop'
            ).merge(merged_df, on=['masterpatientid','censusdate'])
        
        self.sanity_check(merged_df)
        return merged_df

    def generate_features(self):
        """
        desc:
            -Process the Lab related features
    
        :param self.census_df: dataframe
        :param self.labs: dataframe
        :param self.training: bool

        LabTestID is PK for profiledescription
        MasterLabReportID is PK for reportdesciption
        Labs have multiple versions indicated via `VersionNumber` we always need to pick the latest version
        `reportdesciption` is free text which keeps changing for different versions
        """
        with start_action(action_type=f"Labs - generating labs features", labs_shape = self.labs.shape):
            start = time.time()
            self.labs = self.sorter_and_deduper(
                self.labs,
                sort_keys=[
                    'masterpatientid', 'resultdate', 'profiledescription', 'MasterLabReportID', 'VersionNumber',
                    'abnormalityid', 'abnormalitydescription'
                ],
                unique_keys=['masterpatientid', 'resultdate', 'profiledescription', 'MasterLabReportID']
            )
            self.labs['profiledescription'] = self.labs['profiledescription'].str.lower()

            counts = self.labs['profiledescription'].value_counts()

            # preparation for days since
            if self.training:
                min_count = self.config.featurization.labs.min_lab_type_count_for_days_since
                top_lab_type_count = self.config.featurization.labs.top_lab_type_count
                union_lab_type_count = self.config.featurization.labs.union_lab_type_count
                num_lab_type = max(min(len(counts.loc[counts>=min_count]), top_lab_type_count+union_lab_type_count), top_lab_type_count)
                lab_types = list(set(counts[:num_lab_type].index)) 
                conf = OmegaConf.create({'featurization': {'labs': {'lab_types_for_days_since': lab_types}}})
                OmegaConf.save(conf, '/src/saiva/conf/training/generated/labs.yaml')
            else:
                lab_types = self.config.featurization.labs.lab_types_for_days_since or counts.index.tolist()
            self.labs = (
                self.labs[
                    self.labs['profiledescription'].isin(lab_types)
                ].copy().reset_index()
            )

            # if use_conditional_functions is True, then we need to use `createddate` column else ignore it
            if self.config.featurization.labs.use_conditional_functions:
                lab_event_df = self.labs[['masterpatientid', 'resultdate', 'profiledescription','abnormalityid', 'createddate']]
            else:
                lab_event_df = self.labs[['masterpatientid', 'resultdate', 'profiledescription','abnormalityid']]


            # replacing multiple spaces with single space.
            # removing underscores occurences specifically from start and end of the string.
            # replacing spaces with underscores.
            log_message(
                message_type='info', 
                message=f'Labs - creating features for profiledescription, abnormalitydescription & lab_result.'
            )
            self.labs['profiledescription'] = self.labs['profiledescription'].str.replace(r'\_+', r' ', regex=True).replace(r'\s+', r' ', regex=True).str.strip()
            self.labs['profiledescription'] = self.labs['profiledescription'].str.replace(r' ', r'_')

            self.labs['abnormalitydescription'] = self.labs['abnormalitydescription'].str.replace(r'\_+', r' ', regex=True).replace(r'\s+', r' ', regex=True).str.strip()
            self.labs['abnormalitydescription'] = self.labs['abnormalitydescription'].str.replace(r' ', r'_')

            self.labs['lab_result']  = self.labs['profiledescription'].astype(str) + "__" + self.labs['abnormalitydescription'].astype(str)

            # use conditional functions if specified in config
            if self.config.featurization.labs.use_conditional_functions:
                return self.conditional_generate_features(lab_event_df)


            if self.config.featurization.labs.pivot_aggfunc_sum:
                aggfunc='sum'
            else:
                aggfunc='min'

            lab_results_grouped_by_day = self.pivot_patient_date_event_count(self.labs, groupby_column='lab_result', 
                                        date_column='resultdate', prefix='labs_', aggfunc=aggfunc)

            assert lab_results_grouped_by_day.isna().any(axis=None) == False
            assert lab_results_grouped_by_day.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

            merged_df = self.census_df.merge(
                lab_results_grouped_by_day,
                how='outer',
                on=['masterpatientid', 'censusdate'],
            )

            # drop unwanted columns
            merged_df = self.drop_columns(
                merged_df,
                '_masterpatientid|_facilityid'
            )

            assert merged_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
            log_message(
                message_type='info', 
                message='Labs - Feature processing activity completed.',
                merged_df_shape = merged_df.shape
            )
            # =============Trigger garbage collection to free-up memory ==================

            gc.collect()

            cols = [col for col in merged_df.columns if col.startswith('labs__')]

            cumidx_df = self.get_cumsum_features(cols, merged_df,  cumidx=True)       
            log_message(
                message_type='info', 
                message='Labs - cumulative summation, patient days with any events cumsum.',
                cumidx_df_shape = cumidx_df.shape
            )
            
            cumsum_df = self.get_cumsum_features(cols, merged_df,  cumidx=False)
            log_message(
                message_type='info', 
                message='Labs - cumulative summation, total number of events cumsum created.',
                cumsum_df_shape = cumsum_df.shape
            )
            
            merged_df = self.census_df.merge(cumidx_df, on=['facilityid','masterpatientid','censusdate'])
            merged_df = merged_df.merge(cumsum_df, on=['facilityid','masterpatientid','censusdate'])
            assert merged_df.shape[0] == self.census_df.shape[0]

            del cumidx_df
            del cumsum_df

      #=============days since last event features for profiledescription+abnormalityid and abnormalityid=============
            log_message(message_type='info', message='Labs - preparing data for counts of days since last event features.')
            lab_event_df['resultdate'] = pd.to_datetime(lab_event_df['resultdate']).dt.normalize()
            lab_event_df['abnormalityid'] = lab_event_df['abnormalityid'].astype(str)

            # preparing data for profiledescription+abnormalityid
            log_message(message_type='info', message='Labs - preparing `profiledescription_level` pivot table.')
            desc_df = lab_event_df[['masterpatientid', 'resultdate']]
            desc_df['profiledescription_level']=lab_event_df['profiledescription']+'_'+lab_event_df['abnormalityid']
            desc_pivoted = self.pivot_patient_date_event_count(desc_df, groupby_column='profiledescription_level', date_column='resultdate', prefix='labs', fill_value=None)
            desc_pivoted.set_index(['masterpatientid', 'censusdate'], inplace=True)
            desc_pivoted = desc_pivoted.astype('float16')

            # get the column names for day since last event features  
            col_names = desc_pivoted.columns.tolist()

            events_df = self.census_df[['masterpatientid','censusdate']].set_index(['masterpatientid', 'censusdate']).join(
                desc_pivoted, 
                on=['masterpatientid', 'censusdate'],
                how='outer'
            )
            del desc_df, desc_pivoted
            # preparing data for abnormalityid
            log_message(
                message_type='info', 
                message='Labs - preparing `abnormalityid` pivot table.'
            )
            abnormal_df = lab_event_df[['masterpatientid', 'resultdate', 'abnormalityid']]

            abnormal_pivoted = self.pivot_patient_date_event_count(abnormal_df, groupby_column='abnormalityid', date_column='resultdate', prefix='labs', fill_value=None)
            abnormal_pivoted.set_index(['masterpatientid', 'censusdate'], inplace=True)
            abnormal_pivoted = abnormal_pivoted.astype('float16')

            # get the column names for day since last event features  
            col_names = col_names + abnormal_pivoted.columns.tolist()

            events_df = events_df.join(
                abnormal_pivoted,
                on=['masterpatientid', 'censusdate'],
                how='outer'
            )
            del abnormal_df, abnormal_pivoted, lab_event_df

            events_df.fillna(0, inplace=True)
            events_df = events_df.astype(bool)
            events_df.reset_index(inplace=True)

            # get counts of days since last event for all events
            days_last_event_df = self.apply_n_days_last_event_v2(events_df, col_names)
            log_message(
                message_type='info', 
                message='Labs - Created features to counts days since lab last event.',
                days_last_event_df_shape = days_last_event_df.shape
            )
            
            del events_df

            merged_df = merged_df.merge(days_last_event_df, on=['masterpatientid','censusdate'])
            del days_last_event_df
            self.sanity_check(merged_df)
            log_message(
                message_type='info', 
                message=f'Labs - exiting labs, final dataframe shape.', 
                merged_df_shape = merged_df.shape, 
                time_taken=round(time.time() - start, 2)
                        )
            return merged_df