import sys
import time
import pandas as pd
from eliot import log_message, start_action
from .featurizer import BaseFeaturizer


class DemographicFeatures(BaseFeaturizer):
    def __init__(self, census_df, demo_df, training=False):
        # keep 3 key features + payername for later hospice detection
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate', 'payername', 
                                    'payertype', 'day_since_bed_change', 'count_allergy']]
        self.demo_df = demo_df
        self.training = training
        super(DemographicFeatures, self).__init__()

    def generate_features(self):
        """
        Encode demographic data.  Available columns are:
        DOB -> age but DOB missing for some...
        Gender - clean
        Education - messy; ignore for now.
        Citizenship - almost all US; ignore
        Race - messy; ignore for now.
        Religion - very messy; ignore!
        State - long tail; ignore; plenty of blanks and tons of people from IL, IN, AR, KY, TX for some reason.
        Primary Language - Mostly English, Unknown, Spanish, and a long tail of others...  Ignore for now.
        Bedid present only demographics dataframe. It will be dropped from rest of the places
        """
        with start_action(action_type=f"Demographics - generating demographics features, demographics shape",
                          demo_df_shape = self.demo_df.shape):
            start = time.time()
            df = self.census_df.merge(self.demo_df,
                                      how='left',
                                      left_on='masterpatientid',
                                      right_on='masterpatientid')

            df['demo_gender'] = df.gender == 'M'
            df['demo_age_in_days'] = (
                    df.censusdate - df.dateofbirth
            ).dt.days

            # ==================================One hot encoding =================================
            df['primarylanguage'] = df['primarylanguage'].astype('category')
            df['race'] = df['race'].astype('category')
            df['education'] = df['education'].astype('category')
            df['religion'] = df['religion'].astype('category')
            df['maritalstatus'] = df['maritalstatus'].astype('category')

            # Add bedid to ignore_columns since bedid is present only in demographics df
            ignore_cols = self.ignore_columns
            df = self.add_na_indicators(df, ignore_cols)
            log_message(
                message_type = 'info', 
                message = f'Demographics - na indicators created', 
                df_shape = df.shape
            )
            df = self.add_datepart(df, 'dateofbirth', drop=True)
            df = self.add_datepart(df, 'censusdate', drop=False)

            # drop unwanted columns
            df.drop(
                ['gender','state', 'citizenship'],
                axis=1,
                inplace=True
            )

            self.sanity_check(df)
            log_message(
                message_type = 'info', 
                message = f'Demographics - demographics features created.', 
                final_dataframe_shape = df.shape,
                time_taken=round(time.time() - start, 2)
            )
            return df
