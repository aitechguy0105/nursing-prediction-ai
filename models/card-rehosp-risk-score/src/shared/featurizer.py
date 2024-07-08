import sys

import pandas as pd

sys.path.insert(0, '/src')


# from shared.utils import cumsum_job


class BaseFeaturizer(object):
    def __init__(self):
        self.ignore_columns = ['masterpatientid', 'censusdate', 'facilityid']

    def sorter_and_deduper(self, df, sort_keys=[], unique_keys=[]):
        """
        input: dataframe
        output: dataframe
        desc: 1. sort the dataframe w.r.t sort_keys.
              2. drop duplicates w.r.t unique keys.(keeping latest values)
        """
        df.sort_values(by=sort_keys, inplace=True)
        df.drop_duplicates(
            subset=unique_keys,
            keep='last',
            inplace=True
        )
        assert df.duplicated(subset=unique_keys).sum() == 0, f'''Still have dupes!'''
        return df

    def downcast_dtype(self, df):
        # convert float64 and int64 32 bit verison to save on memory
        df.loc[:, df.select_dtypes(include=['int64']).columns] = df.select_dtypes(
            include=['int64']).apply(
            pd.to_numeric, downcast='unsigned'
        )
        # Convert all float64 columns to float32
        df.loc[:, df.select_dtypes(include=['float64']).columns] = df.select_dtypes(
            include=['float64']
        ).astype('float32')

        return df

    def convert_to_int16(self, df):
        """
        Convert entire dataframe to int16 columns
        """
        for col in df:
            df[col] = df[col].astype('int16')
        return df
