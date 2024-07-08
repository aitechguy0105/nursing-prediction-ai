import sys
import os

# add the directory containing the saiva module to the Python path
saiva_dir = os.path.join(os.getcwd(), 'saiva')
sys.path.append(saiva_dir)

import unittest
from saiva.model.shared.featurizer import BaseFeaturizer
from saiva.model.shared.assessments import AssessmentFeatures

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

class TestConditionalAggregator(unittest.TestCase):

    def test_conditional_get_last_values_job1(self):
        census = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2,
                                3, 3, 3, 3, 3, 3, 3, 3,
                                4, 4, 4, 4, 4, 4, 4, 4],
            'facilityid': [1] * 32,
            'censusdate': ['2020-01-01', '2020-01-02', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', 
                           '2020-01-02', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', 
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-10', '2020-01-11',
                           '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-08', '2020-01-09', '2020-01-14'
                           ],
        })
        census['censusdate'] = pd.to_datetime(census['censusdate'])

        # create another dataframe with event data containing a masterpatientid, eventdate, createddate, and event_type
        df = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2,
                                3, 3, 3, 3, 3,
                                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            'eventdate': ['2020-01-01 18:00:00', '2020-01-02 18:00:00', '2020-01-09 06:00:00', '2020-01-11 18:00:00', '2020-01-14 12:00:00', '2020-01-14 12:00:00', '2020-01-14 15:00:00', '2020-01-14 18:00:00',
                          '2020-01-02 06:00:00', '2020-01-06 12:00:00', '2020-01-08 18:00:00', '2020-01-08 18:00:00', '2020-01-11 00:00:00', '2020-01-10 12:00:00',
                          '2020-01-02 00:00:00', '2020-01-03 06:00:00', '2020-01-03 06:00:00', '2020-01-03 18:00:00', '2020-01-06 06:00:00',
                          '2020-01-02 18:00:00', '2020-01-03 12:00:00', '2020-01-03 18:00:00', '2020-01-04 06:00:00', '2020-01-05 00:00:00', '2020-01-05 18:00:00',
                                                 '2020-01-08 00:00:00', '2020-01-08 00:00:00', '2020-01-14 01:00:00', '2020-01-14 17:00:00', '2020-01-14 00:00:00'
                         ],
            'createddate': ['2020-01-02 18:00:00', '2020-01-09 18:00:00', '2020-01-12 12:00:00', '2020-01-11 06:00:00', '2020-01-14 00:00:00', '2020-01-14 12:00:00', '2020-01-13 12:00:00', '2020-01-14 12:00:00',
                            '2020-01-06 00:00:00', '2020-01-12 12:00:00', '2020-01-12 12:00:00', '2020-01-12 12:00:00', '2020-01-17 18:00:00', '2020-01-10 18:00:00',
                            '2020-01-05 00:00:00', '2020-01-10 00:00:00', '2020-01-10 06:00:00', '2020-01-11 12:00:00', '2020-01-07 12:00:00',
                            '2020-01-17 12:00:00', '2020-01-05 06:00:00', '2020-01-04 18:00:00', '2020-01-06 12:00:00', '2020-01-09 12:00:00', '2020-01-13 06:00:00',
                                                   '2020-01-09 12:00:00', '2020-01-09 18:00:00', '2020-01-14 12:00:00', '2020-01-14 19:00:00', '2020-01-17 00:00:00'
                          ],
            'deleteddate': ['2020-01-09 18:00:00', np.nan, np.nan, np.nan, '2020-01-10 00:00:00', '2020-01-21 12:00:00', np.nan, np.nan,
                            np.nan, '2020-01-18 12:00:00', np.nan, np.nan, np.nan, np.nan,
                            np.nan, '2020-01-14 00:00:00', np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, '2020-01-20 12:00:00', np.nan, np.nan, '2020-01-13 12:00:00', np.nan, np.nan, '2020-01-20 12:00:00', np.nan
                           ],
            'event_value': ['C', 'D', 'C', 'D', 'G', 'D', 'F', 'E',
                            'C', 'A', 'E', 'E', 'G', 'F',
                            'D', 'G', 'E', 'E', 'A',
                            'G', 'B', 'D', 'F', 'B', 'B', 'G', 'B', 'E', 'G', 'B'
                           ],
            'event_value2': [8,  4,  3, 10,  9, 10, 10,  2,
                             0,  1,  7,  9,  4,  0,
                             0,  5,  6,  7,  4,
                             9,  9, 10,  6, 10,  5, 10,  6,  4,  4,  7
                            ]        
        })

        df[['eventdate', 'createddate', 'deleteddate']] = df[['eventdate', 'createddate', 'deleteddate']].apply(pd.to_datetime, axis=1)

        def minmax_sum(s):
            """ Meaningless function for test purposes only
            """
            min_value = s.min()
            max_value = s.max()
            if pd.notnull(min_value):
                return min_value + max_value
            else:
                np.nan

        df_result = BaseFeaturizer().conditional_aggregator(
            census = census,
            df = df,
            value_headers = ['event_value', 'event_value2'],
            effectivedate_header = 'eventdate',
            createddate_header = 'createddate',
            deleteddate_header = 'deleteddate',
            result_headers = ['result1', 'result2'],
            window_size = 7,
            agg_func = minmax_sum
        )

        nan = np.nan
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2,
                                3, 3, 3, 3, 3, 3, 3, 3,
                                4, 4, 4, 4, 4, 4, 4, 4],
            'facilityid': [1] * 32,
            'censusdate': ['2020-01-01', '2020-01-02', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', 
                           '2020-01-02', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', 
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-10', '2020-01-11',
                           '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-08', '2020-01-09', '2020-01-14'
                           ],
            'result1': [None, 'CC', None, None,  'DD', 'CD', 'CD', 'CF',
                        None, 'CC', 'CC', 'CC', None, 'FF', 'FF', 'AF',
                        None, None, None, None,  'DD', 'DD', 'AA', 'AA',
                        None, None, 'DD', 'BD',  'BF', 'BF', 'BG', 'BG'
                       ],
            'result2': [None,   16, None, None,   20, 13, 13, 12,
                        None,    0,    0,    0, None,  0,  0,  9,
                        None, None, None, None,    0,  0,  8,  8,
                        None, None,   20,   19,   16, 16, 16, 10
                       ]
        })

        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)

if __name__ == '__main__':
    unittest.main()