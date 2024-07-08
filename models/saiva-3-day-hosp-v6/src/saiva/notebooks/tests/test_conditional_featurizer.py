import sys
import os

# add the directory containing the saiva module to the Python path
saiva_dir = os.path.join(os.getcwd(), 'saiva')
sys.path.append(saiva_dir)

import unittest
from model.shared.featurizer import BaseFeaturizer

import pandas as pd
from pandas.testing import assert_frame_equal


class TestFeaturizer(unittest.TestCase):
    def test_conditional_cumsum_job_1(self):
        # create a dataframe with random data
        census = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10'],
        })
        census['censusdate'] = pd.to_datetime(census['censusdate'])

        # create another dataframe with event data containing a masterpatientid, eventdate, createddate, and event_type
        df = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 
                                3, 3],
            'eventdate': ['2020-01-01', '2020-01-01', '2021-10-08', 
                          '2020-01-08', '2020-02-01', 
                          '2020-02-01', '2020-02-01'],
            'createddate': ['2020-01-01', '2020-01-01', '2021-10-08', 
                          '2020-01-08', '2020-02-01', 
                          '2020-02-01', '2020-02-01'],
            'event_type': ['A', 'A', 'A', 
                           'A', 'A',
                           'A', 'A']            
        })

        df[['eventdate', 'createddate']] = df[['eventdate', 'createddate']].apply(pd.to_datetime)


        # run the function
        df_result = BaseFeaturizer._conditional_cumsum_job( 
            census,
            df,
            'test',
            'eventdate',
            'createddate',
            missing_event_dates='drop',
            cumidx=False,
            sliding_windows=[2,7,14,30]
        )

        # downcast floats to ints 
        # float columns 
        float_cols = df_result.select_dtypes(include=['float64']).columns

        # downcast float columns to int64
        df_result[float_cols] = df_result[float_cols].astype('int64')

        # create df_expected 
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10'],
            'cumsum_2_day_test': [2, 2, 1,
                                  0, 1, 1,
                                  0, 2, 0],
            'cumsum_7_day_test': [2, 2, 1,
                                  0, 1, 1,
                                  0, 2, 0],
            'cumsum_14_day_test': [2, 2, 1,
                                   0, 1, 1,
                                   0, 2, 0],
            'cumsum_30_day_test': [2, 2, 1,
                                   0, 1, 2,
                                   0, 2, 0],
            'cumsum_all_test': [2, 2, 3,
                                0, 1, 2,
                                0, 2, 2],
        })
        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)



    def test_conditional_cumsum_job_2(self):
        # create a dataframe with random data
        census = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10'],
        })
        census['censusdate'] = pd.to_datetime(census['censusdate'])

        # create another dataframe with event data containing a masterpatientid, eventdate, createddate, and event_type
        df = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 
                                3, 3],
            'eventdate': ['2020-01-01', '2020-01-01', '2021-10-08', 
                          '2020-01-08', '2020-02-01', 
                          '2020-01-10', '2020-02-01'],
            'createddate': ['2020-01-01', '2020-01-02', '2021-10-08', 
                          '2020-01-07', '2020-02-01', 
                          '2020-02-01', '2020-03-05'],
            'event_type': ['A', 'A', 'A', 
                           'A', 'A',
                           'A', 'A']            
        })

        df[['eventdate', 'createddate']] = df[['eventdate', 'createddate']].apply(pd.to_datetime)


        # run the function
        df_result = BaseFeaturizer._conditional_cumsum_job( 
            census,
            df,
            'test',
            'eventdate',
            'createddate',
            missing_event_dates='drop',
            cumidx=False,
            sliding_windows=[2,7,14,30]
        )

        # downcast floats to ints 
        # float columns 
        float_cols = df_result.select_dtypes(include=['float64']).columns

        # downcast float columns to int64
        df_result[float_cols] = df_result[float_cols].astype('int64')

        # create df_expected 
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10'],
            'cumsum_2_day_test': [1, 2, 1,
                                  0, 1, 1,
                                  0, 0, 0],
            'cumsum_7_day_test': [1, 2, 1,
                                  0, 1, 1,
                                  0, 0, 0],
            'cumsum_14_day_test': [1, 2, 1,
                                   0, 1, 1,
                                   0, 0, 0],
            'cumsum_30_day_test': [1, 2, 1,
                                   0, 1, 2,
                                   0, 1, 0],
            'cumsum_all_test': [1, 2, 3,
                                0, 1, 2,
                                0, 1, 2],
        })
        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)

if __name__ == '__main__':
    unittest.main()