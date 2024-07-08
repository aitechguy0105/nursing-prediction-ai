import sys

import pandas as pd

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class DemographicFeatures(BaseFeaturizer):
    def __init__(self, census_df, demo_df, training=False):
        self.census_df = census_df
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

        df = self.census_df.merge(self.demo_df,
                                  how='left',
                                  left_on='masterpatientid',
                                  right_on='masterpatientid')

        df['demo_gender'] = df.gender == 'M'
        df['demo_age_in_days'] = (
                df.censusdate - df.dateofbirth
        ).dt.days

        # ==================================One hot encoding =================================
        dummies = pd.concat(
            [
                pd.get_dummies(
                    df.primarylanguage, prefix='demo_primarylanguage'
                ),
                pd.get_dummies(df.race, prefix='demo_race'),
                pd.get_dummies(df.education, prefix='demo_education'),
                pd.get_dummies(df.religion, prefix='demo_religion'),
                pd.get_dummies(df.facilityid, prefix='demo_facility'),
            ],
            axis=1,
        )

        # Add bedid to ignore_columns since bedid is present only in demographics df
        ignore_cols = self.ignore_columns + ['bedid']
        df = pd.concat([df, dummies], axis=1)
        df = self.add_na_indicators(df, ignore_cols)
        df = self.add_datepart(df, 'dateofbirth', drop=True)
        df = self.add_datepart(df, 'censusdate', drop=False)

        # drop unwanted columns
        df.drop(
            ['gender', 'primarylanguage', 'religion', 'race', 'education', 'state', 'citizenship'],
            axis=1,
            inplace=True
        )
        df.drop(
            df.columns[df.columns.str.contains(
                '_masterpatientid|_facilityid|_x$|_y$'
            )].tolist(),
            axis=1,
            inplace=True
        )

        return df
