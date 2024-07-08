import sys
import gc
import time
import pandas as pd
import numpy as np
from eliot import log_message, start_action
from datetime import timedelta

from .featurizer import BaseFeaturizer
from .constants import FALL_LABEL


class RiskFeatures(BaseFeaturizer):
    def __init__(self, census_df, risks_df, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.risks_df = risks_df
        self.training = training
        self.config = config
        super(RiskFeatures, self).__init__()

    def conditional_fall_events(self):
        event_date = 'incidentdate'
        event_reported_date = 'createddate'
        prefix = 'risk'
        groupby_column = 'fall'

        fall_event = self.risks_df.loc[self.risks_df['description'].str.contains(FALL_LABEL, regex=True), 
                                       ['masterpatientid', 'incidentdate', 'createddate']].assign(
                fall = lambda x: np.select([
                    pd.to_datetime(x.incidentdate).dt.hour.between(8,15),
                    pd.to_datetime(x.incidentdate).dt.hour.between(16,23),
                    pd.to_datetime(x.incidentdate).dt.hour.between(0,7),
                ],
                [
                    'fall_morning',
                    'fall_afternoon',
                    'fall_night'
                ])
            )
        # normalize dates. needed before _conditional_cumsum_job operations 
        fall_event[[event_date, event_reported_date]] = fall_event[[event_date, event_reported_date]].apply(lambda x: pd.to_datetime(x).dt.normalize()) 
        
        # Do cumulative index on all fall events
        fall_event_cumdix = self.conditional_cumsum_features(
            fall_event,
            prefix, 
            event_date,
            event_reported_date,
            groupby_column = groupby_column,
            missing_event_dates='drop',
            cumidx=True,
            sliding_windows=[1,2,3,4,5,6,7,14,30]
        ).merge(
            self._conditional_cumsum_job(
                self.census_df[['masterpatientid','censusdate']],
                fall_event.copy(),
                'risk_fall_all_day',
                event_date,
                event_reported_date,
                missing_event_dates='drop',
                cumidx=True,
                sliding_windows=[1,2,3,4,5,6,7,14,30]
            ), on=['masterpatientid','censusdate']
        )

        # Do cumulative summation on all fall events
        fall_event_cumsum = self.conditional_cumsum_features(
            fall_event,
            prefix, 
            event_date,
            event_reported_date,
            groupby_column = groupby_column,
            missing_event_dates='drop',
            cumidx=False,
            sliding_windows=[1,2,3,4,5,6,7,14,30]
        ).merge(
            self._conditional_cumsum_job(
                self.census_df[['masterpatientid','censusdate']],
                fall_event.copy(),
                'risk_fall_all_day',
                event_date,
                event_reported_date,
                missing_event_dates='drop',
                cumidx=False,
                sliding_windows=[1,2,3,4,5,6,7,14,30]
            ), on=['masterpatientid','censusdate']
        )

        return fall_event_cumdix, fall_event_cumsum



    def conditional_days_since_last_incident(self, incident_df):
        event_date = 'incidentdate'
        event_reported_date = 'createddate'
        prefix = 'risk'

        return self.conditional_days_since_last_event(
                df=incident_df,
                prefix=prefix,
                event_date_column=event_date,
                event_reported_date_column=event_reported_date,
                groupby_column='desc_id',
                missing_event_dates='drop'
            )
        

    def conditional_generate_features(self):
        incident_df = self.risks_df[['facilityid', 'masterpatientid', 'incidentdate', 'createddate']].copy()
        self.risks_df.loc[:,'description'] = self.risks_df['description'].str.lower()
        incident_df['desc_id'] = self.risks_df['description'] + '_' + self.risks_df['typeid'].astype(str)
    
        event_date = 'incidentdate'
        event_reported_date = 'createddate'

        incident_df[[event_date, event_reported_date]] = incident_df[[event_date, event_reported_date]].apply(lambda x: pd.to_datetime(x).dt.normalize())

        days_last_event_df = self.conditional_days_since_last_incident(incident_df)

        fall_event_cumdix, fall_event_cumsum = self.conditional_fall_events()

        
        ########################################## Merge Results & Return ###########################################
        final = self.census_df.merge(days_last_event_df, on=['masterpatientid','censusdate'])
        
        final = final.merge(fall_event_cumdix, on=['masterpatientid','censusdate'])
        
        final = final.merge(fall_event_cumsum, on=['masterpatientid','censusdate'])
        
        final['censusdate'] = pd.to_datetime(final['censusdate'])
        
        # get the fall targets
        log_message(message_type='info', message='getting fall target')
        fall= self.risks_df[self.risks_df.description.str.contains(FALL_LABEL, regex=True)]\
                    [['masterpatientid','incidentdate']]
        target_df = self.get_target(fall, self.census_df, 'fall')
        
        final = final.merge(target_df, on=['masterpatientid','censusdate'], how='left')
        del days_last_event_df
        del fall_event_cumdix
        del fall_event_cumsum
        
        self.sanity_check(final)

        gc.collect()
        return final



    def generate_features(self):
        """
        TypeID - Indicates the type ID
        Description - The description of the pick list item
        """
        # use conditional functions if specified in config
        with start_action(action_type=f"Risks - generating risks features", risks_shape = self.risks_df.shape):
            start = time.time()
            if self.config.featurization.risks.use_conditional_functions:
                return self.conditional_generate_features()

            ######################################Get days since last incident ################################
            log_message(message_type='info', message='Risks - preparing data for count of days since last event features')
            incident_df = self.risks_df[['facilityid', 'masterpatientid', 'incidentdate']].copy()

            self.risks_df.dropna(subset=['typeid'], inplace=True)
            self.risks_df['description'].fillna('nan_description', inplace=True)
            self.risks_df.loc[:,'description'] = self.risks_df['description'].str.lower()
            incident_df['desc_id'] = self.risks_df['description'] + '_' + self.risks_df['typeid'].astype(str)

            incident_pivoted = self.pivot_patient_date_event_count(incident_df, groupby_column='desc_id', 
                                        date_column='incidentdate', prefix='risk', fill_value=0)

            incident_pivoted = self.downcast_dtype(incident_pivoted)

            cols = incident_pivoted.columns[2:]

            del incident_df 

            events_df = self.census_df[['masterpatientid','censusdate']]\
                                        .merge(incident_pivoted, 
                                             on=['masterpatientid','censusdate'],
                                             how='outer')

            events_df[cols] = events_df[cols].fillna(0)

            events_df = events_df.sort_values(by=['masterpatientid', 'censusdate'])\
                        .reset_index(drop=True)

            #Get days since last incident
            log_message(message_type='info', message=f'Risks - creating days since last event feature.')
            days_last_event_df = self.apply_n_days_last_event(events_df, cols)

            ##########################################Get Cumsum features ###########################################
            fall_event = self.risks_df.loc[self.risks_df['description'].str.contains(FALL_LABEL, regex=True), 
                                           ['masterpatientid', 'incidentdate']]

            #all day
            fall_event['fall_all_day'] = 1
            #morning, 8am-4pm
            morning = (pd.to_datetime(fall_event.incidentdate).dt.hour>=8)&\
                        (pd.to_datetime(fall_event.incidentdate).dt.hour<=15)
            if sum(morning)>0:
                fall_event.loc[morning, 'fall_morning'] = 1

            #afternoon, 4pm-midnight
            afternoon = (pd.to_datetime(fall_event.incidentdate).dt.hour>=16)&\
                        (pd.to_datetime(fall_event.incidentdate).dt.hour<=23)
            if sum(afternoon)>0:
                fall_event.loc[afternoon, 'fall_afternoon'] = 1

            #night midnight-8am
            night = (pd.to_datetime(fall_event.incidentdate).dt.hour>=0)&\
                        (pd.to_datetime(fall_event.incidentdate).dt.hour<=7)
            if sum(night)>0:
                fall_event.loc[night, 'fall_night'] = 1

            fall_event['incidentdate'] = pd.to_datetime(fall_event.incidentdate).dt.normalize()
            fall_event.sort_values(['masterpatientid', 'incidentdate'], inplace=True)
            fall_event.reset_index(drop=True, inplace=True)

            # get count of falls for each patient day
            fall_event = fall_event.groupby(['masterpatientid', 'incidentdate']).count().reset_index()
            fall_event = self.downcast_dtype(fall_event)
            fall_event.rename(columns={'incidentdate':'censusdate'}, inplace=True)

            # add prefix to the column names 
            fall_event_cols = ['risk_'+ col for col in fall_event.columns[2:]]
            fall_event.columns = ['masterpatientid','censusdate'] + fall_event_cols

            fall_event = self.census_df[['masterpatientid','censusdate']]\
                                        .merge(fall_event, 
                                               on=['masterpatientid','censusdate'],
                                               how='outer')

            fall_event[fall_event_cols] = fall_event[fall_event_cols].fillna(0)

            fall_event = fall_event.sort_values(by=['masterpatientid', 'censusdate'])\
                        .reset_index(drop=True)

             # Do cumulative summation on all med columns
            fall_event_cumdix = self.get_cumsum_features(fall_event_cols, fall_event, cumidx=True, 
                                                         slidding_windows=[1,2,3,4,5,6,7,14,30])
            log_message(message_type='info', 
                        message=f'Risks - cumulative summation, patient days with any events cumsum.', fall_event_cumdix_shape = fall_event_cumdix.shape)

            fall_event_cumsum = self.get_cumsum_features(fall_event_cols, fall_event, cumidx=False, 
                                                         slidding_windows=[1,2,3,4,5,6,7,14,30])
            log_message(message_type='info', 
                        message=f'Risks - cumulative summation, total number of events cumsum.', fall_event_cumsum_shape = fall_event_cumsum.shape)
            ###################################################################################################################

            final = self.census_df.merge(days_last_event_df, on=['masterpatientid','censusdate'])

            final = final.merge(fall_event_cumdix, on=['masterpatientid','censusdate'])

            final = final.merge(fall_event_cumsum, on=['masterpatientid','censusdate'])

            # get the fall targets
            log_message(message_type='info', message='Risks - Creating fall target')
            fall= self.risks_df[self.risks_df.description.str.contains(FALL_LABEL, regex=True)]\
                        [['masterpatientid','incidentdate']]
            target_df = self.get_target(fall, self.census_df, 'model_fall')

            final = final.merge(target_df, on=['masterpatientid','censusdate'], how='left')
            del days_last_event_df
            del fall_event_cumdix
            del fall_event_cumsum

            self.sanity_check(final)
            log_message(
                message_type='info', 
                message=f'Risks - exiting risks', 
                final_dataframe_shape=final.shape,
                time_taken=round(time.time() - start, 2)
            )
            return final
