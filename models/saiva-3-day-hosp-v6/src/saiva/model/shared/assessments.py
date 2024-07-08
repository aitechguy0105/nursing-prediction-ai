import sys

import pandas as pd
from eliot import log_message, start_action
import time
from .featurizer import BaseFeaturizer


class AssessmentFeatures(BaseFeaturizer):
    def __init__(self, census_df, assessments_df, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.assessments_df = assessments_df
        self.config = config
        self.training = training
        super(AssessmentFeatures, self).__init__()

    def generate_features(self):
        """
        Description - Description of the standard assessment e.g.: RAP - Nutritional Status Assessment, 
                      RAP - Physical Restraints Assessment, Clinical Evaluation, etc
        """
        with start_action(action_type=f"Assessments - generating assessments features.", 
                          assessments_df_shape = self.assessments_df.shape):
            start = time.time()
            if self.config.featurization.assessments.use_conditional_functions:

                self.census_df['censusdate'] = pd.to_datetime(self.census_df['censusdate']).dt.normalize()
                days_last_event_df = self.conditional_days_since_last_event(
                    df=self.assessments_df,
                    prefix='assessments',
                    event_date_column='assessmentdate',
                    event_reported_date_column='createddate',
                    groupby_column='description',
                    missing_event_dates='drop',
                    lowercase_groupby_column=True
                )

            else:
                ######################################Get days since last assessment ################################
                log_message(
                    message_type='info',
                    message='Assessments - preparing data for count of days since last event features.'
                )
                asses_df = self.assessments_df[['masterpatientid', 'assessmentdate', 'description']].copy()
                asses_df['description'] = asses_df['description'].str.lower()

                asses_pivoted = self.pivot_patient_date_event_count(asses_df, groupby_column='description', 
                    date_column='assessmentdate', prefix='assessments')

                asses_pivoted= self.downcast_dtype(asses_pivoted)

                cols = asses_pivoted.columns[2:]

                del asses_df 

                events_df = self.census_df[['masterpatientid','censusdate']]\
                                            .merge(asses_pivoted, 
                                                 on=['masterpatientid','censusdate'],
                                                 how='outer')
                del asses_pivoted
                assert events_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
                events_df[cols] = events_df[cols].fillna(0)

                events_df = events_df.sort_values(by=['masterpatientid', 'censusdate'])\
                            .reset_index(drop=True)

                #Get days since last event
                
                days_last_event_df = self.apply_n_days_last_event(events_df, cols)
                days_last_event_df[days_last_event_df.columns[2:]]=days_last_event_df[days_last_event_df.columns[2:]]\
                                                                    .astype(int)
                log_message(
                    message_type = 'info', 
                    message = f'Assessments - count of days since last assessment event feature created.',
                    days_last_event_df_shape=days_last_event_df.shape
                )
                ################################################################################################

            final = self.census_df.merge(days_last_event_df, on=['masterpatientid','censusdate'])
            
            del days_last_event_df

            self.sanity_check(final)
            log_message(
                message_type='info', 
                message=f'Assessments - exiting assessments featurization code, final shape.', 
                final_shape=final.shape,
                time_taken=round(time.time() - start, 2)
            )
            return final
