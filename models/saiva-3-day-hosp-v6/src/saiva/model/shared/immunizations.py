import sys

import pandas as pd
from eliot import log_message, start_action
from .featurizer import BaseFeaturizer
import time

class ImmunizationFeatures(BaseFeaturizer):
    def __init__(self, census_df, immuns_df, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.immuns_df = immuns_df
        self.training = training
        self.config = config
        super(ImmunizationFeatures, self).__init__()

    def generate_features(self):
        """
        Mark the date since the patients immunized as 1 for each type of immunizationdesc
        """
        with start_action(action_type=f"Immunization - generating immunization features.",
                          immunization_shape = self.immuns_df.shape
                         ):
            start = time.time()
            # use conditional functions if specified in config
            if self.config.featurization.immunizations.use_conditional_functions:
                # set immunizationdate to max of immunizationdate and createddate 
                self.immuns_df['immunizationdate'] = self.immuns_df[['immunizationdate', 'createddate']].max(axis=1)

            self.immuns_df['immunizationdate'] = pd.to_datetime(self.immuns_df['immunizationdate']).dt.date
            self.immuns_df['immunizationdesc'] = self.immuns_df['immunizationdesc'].str.lower()
            self.immuns_df['indicator'] = 1

            immun_pivoted = pd.pivot_table(self.immuns_df,
                                index = ['facilityid','masterpatientid','immunizationdate'],
                                columns = 'immunizationdesc',
                                values='indicator',
                                fill_value=None).reset_index()

            self.census_df['censusdate'] = pd.to_datetime(self.census_df['censusdate']).dt.date
            final_df = self.census_df[['facilityid', 'masterpatientid', 'censusdate']].merge(immun_pivoted,
                              how='left',
                              left_on=['facilityid','masterpatientid', 'censusdate'],
                              right_on=['facilityid','masterpatientid', 'immunizationdate'])

            final_df = final_df.drop(columns='immunizationdate')
            final_df = final_df.sort_values(['facilityid','masterpatientid', 'censusdate']).reset_index(drop=True)

            cols = final_df.columns[3:].tolist()
            final_df[cols] = final_df.groupby('masterpatientid')[cols].ffill().fillna(0).astype(int)
            final_df.columns = ['facilityid','masterpatientid', 'censusdate'] + ['immun_'+col for col in cols]
            final_df['censusdate'] = pd.to_datetime(final_df['censusdate'])
            del immun_pivoted
            
            self.sanity_check(final_df)
            log_message(
                message_type = 'info', 
                message = f'Immunization - immunization features created.',
                final_dataframe_shape = final_df.shape,
                time_taken=round(time.time()-start, 2)
            )
        return final_df
