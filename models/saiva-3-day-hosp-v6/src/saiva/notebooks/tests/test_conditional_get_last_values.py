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

class TestConditionalGetLastValues(unittest.TestCase):

    def test_conditional_get_last_values_job1(self):
        census = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'facilityid': [1] * 15,
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'
                           ],
        })
        census['censusdate'] = pd.to_datetime(census['censusdate'])

        # create another dataframe with event data containing a masterpatientid, eventdate, createddate, and event_type
        df = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 
                                3, 3,
                                4, 4, 4],
            'eventdate': ['2020-01-01 01:00:00', '2020-01-01 02:00:00', '2021-10-08 07:00:00', 
                          '2020-01-08 11:00:00', '2020-02-01 11:00:00', 
                          '2020-02-01 11:00:00', '2020-02-01 10:00:00',
                          '2020-01-01 07:00:00', '2020-01-02 07:00:00', '2020-01-03 07:00:00'],
            'createddate': ['2020-01-01 01:15:00', '2020-01-01 02:15:00', '2021-10-08 07:15:00', 
                          '2020-01-08 11:15:00', '2020-02-01 11:15:00', 
                          '2020-02-03 15:00:00', '2020-02-01 16:00:00',
                          '2020-01-01 07:15:00', '2020-01-05 07:00:00', '2020-01-04 07:00:00'
                          ],
            'event_value': ['A', 'A', 'A', 
                           'B', 'C',
                           'C', 'A',
                           'E', 'D', 'F']            
        })

        df[['eventdate', 'createddate']] = df[['eventdate', 'createddate']].apply(pd.to_datetime, axis=1)

        # run the function
        obj = AssessmentFeatures(
            census_df = census,
            assessments_df = None,
            config = None,
            training=False
        )
        df_result = obj.conditional_get_last_values(
            df = df,
            prefix = 'test',
            event_date_column = 'eventdate',
            event_reported_date_column = 'createddate',
            event_deleted_date_column = None,
            value_columns = ['event_value'],
            groupby_column = None,
            missing_event_dates = 'drop',
            n = 2,
            reset = None
        )

        # create df_expected 
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'],
            'facilityid': [1] * 15,
            'test_event_value_1st_previous_value': ['A', 'A', 'A',
                                                    np.nan, 'B', 'C',
                                                    np.nan, 'A', 'C',
                                                    'E', 'E', 'E', 'F', 'F', 'F'
                                                    ],
            'test_event_value_2nd_previous_value': ['A', 'A', 'A',
                                                    np.nan, np.nan, 'B',
                                                    np.nan, np.nan, 'A',
                                                    np.nan, np.nan, np.nan, 'E', 'D', 'D'
                                                    ]
        })
        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)

    def test_conditional_get_last_values_job2(self):
        census = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'facilityid': [1] * 15,
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'
                           ],
        })
        census['censusdate'] = pd.to_datetime(census['censusdate'])

        # create another dataframe with event data containing a masterpatientid, eventdate, createddate, and event_type
        df = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 
                                3, 3,
                                4, 4, 4],
            'eventdate': ['2020-01-01 01:00:00', '2020-01-01 02:00:00', '2021-10-08 07:00:00', 
                          '2020-01-08 11:00:00', '2020-02-01 11:00:00', 
                          '2020-02-01 11:00:00', '2020-02-01 10:00:00',
                          '2020-01-01 07:00:00', '2020-01-02 07:00:00', '2020-01-03 07:00:00'],
            'createddate': ['2020-01-01 01:15:00', '2020-01-01 02:15:00', '2021-10-08 07:15:00', 
                          '2020-01-08 11:15:00', '2020-02-01 11:15:00', 
                          '2020-02-01 15:00:00', '2020-02-01 16:00:00',
                          '2020-01-01 07:15:00', '2020-01-05 07:00:00', '2020-01-04 07:00:00'
                          ],
            'deleteddate': [np.nan, np.nan, np.nan,
                            '2020-02-01 18:03:03','2020-02-01 18:03:03',
                            '2020-03-01 10:00:00', np.nan,
                            np.nan, '2020-01-06 10:00:00', '2020-01-08 10:00:00'
                            ],
            'event_value': ['A', 'A', 'A', 
                           'B', 'C',
                           'C', 'A',
                           'E', 'D', 'F']            
        })

        df[['eventdate', 'createddate', 'deleteddate']] = df[['eventdate', 'createddate', 'deleteddate']].apply(pd.to_datetime, axis=1)

        # run the function
        obj = AssessmentFeatures(
            census_df = census,
            assessments_df = None,
            config = None,
            training=False
        )
        df_result = obj.conditional_get_last_values(
            df = df,
            prefix = 'test',
            event_date_column = 'eventdate',
            event_reported_date_column = 'createddate',
            event_deleted_date_column = 'deleteddate',
            value_columns = ['event_value'],
            groupby_column = None,
            missing_event_dates = 'drop',
            n = 2,
            reset = None
        )

        # create df_expected 
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'],
            'facilityid': [1] * 15,
            'test_event_value_1st_previous_value': ['A', 'A', 'A',
                                                    np.nan, 'B', np.nan,
                                                    np.nan, 'C', 'A',
                                                    'E', 'E', 'E', 'F', 'F', 'F'
                                                    ],
            'test_event_value_2nd_previous_value': ['A', 'A', 'A',
                                                    np.nan, np.nan, np.nan,
                                                    np.nan, 'A', np.nan,
                                                    np.nan, np.nan, np.nan, 'E', 'D', 'E'
                                                    ]
        })
        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)

    def test_conditional_get_last_values_job3(self):
        census = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'facilityid': [1] * 15,
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'
                           ],
        })
        census['censusdate'] = pd.to_datetime(census['censusdate'])

        # create another dataframe with event data containing a masterpatientid, eventdate, createddate, and event_type
        df = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 
                                3, 3,
                                4, 4, 4],
            'eventdate': ['2020-01-01 01:00:00', '2020-01-01 02:00:00', '2021-10-08 07:00:00', 
                          '2020-01-08 11:00:00', '2020-02-01 11:00:00', 
                          '2020-02-01 11:00:00', '2020-02-01 10:00:00',
                          '2020-01-01 07:00:00', '2020-01-02 07:00:00', '2020-01-03 07:00:00'],
            'createddate': ['2020-01-01 01:15:00', '2020-01-01 02:15:00', '2021-10-08 07:15:00', 
                          '2020-01-08 11:15:00', '2020-02-01 11:15:00', 
                          '2020-02-03 15:00:00', '2020-02-01 16:00:00',
                          '2020-01-01 07:15:00', '2020-01-05 07:00:00', '2020-01-04 07:00:00'
                          ],
            'event_value': ['A', 'A', 'A', 
                           'B', 'C',
                           'C', 'A',
                           'E', 'D', 'F'],
            'event_value2': [0, 1, 2,
                             3, 4,
                             5, 6,
                             7, 8, 9]  
        })

        df[['eventdate', 'createddate']] = df[['eventdate', 'createddate']].apply(pd.to_datetime, axis=1)

        # run the function
        obj = AssessmentFeatures(
            census_df = census,
            assessments_df = None,
            config = None,
            training=False
        )
        df_result = obj.conditional_get_last_values(
            df = df,
            prefix = 'test',
            event_date_column = 'eventdate',
            event_reported_date_column = 'createddate',
            event_deleted_date_column = None,
            value_columns = ['event_value', 'event_value2'],
            groupby_column = None,
            missing_event_dates = 'drop',
            n = 2,
            reset = None
        )

        # create df_expected 
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'],
            'facilityid': [1] * 15,
            'test_event_value_1st_previous_value': ['A', 'A', 'A',
                                                    np.nan, 'B', 'C',
                                                    np.nan, 'A', 'C',
                                                    'E', 'E', 'E', 'F', 'F', 'F'
                                                    ],
            'test_event_value_2nd_previous_value': ['A', 'A', 'A',
                                                    np.nan, np.nan, 'B',
                                                    np.nan, np.nan, 'A',
                                                    np.nan, np.nan, np.nan, 'E', 'D', 'D'
                                                    ],
            'test_event_value2_1st_previous_value': [1, 1, 2,
                                                     np.nan, 3, 4,
                                                     np.nan, 6, 5,
                                                     7, 7, 7, 9, 9, 9
                                                    ],
            'test_event_value2_2nd_previous_value': [0, 0, 1,
                                                     np.nan, np.nan, 3,
                                                     np.nan, np.nan, 6,
                                                     np.nan, np.nan, np.nan, 7, 8, 8
                                                    ]
        })
        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)


    def test_conditional_get_last_values_job4(self):
        census = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'facilityid': [1] * 15,
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'
                           ],
        })
        census['censusdate'] = pd.to_datetime(census['censusdate'])

        # create another dataframe with event data containing a masterpatientid, eventdate, createddate, and event_type
        df = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 
                                3, 3,
                                4, 4, 4],
            'eventdate': ['2020-01-01 01:00:00', '2020-01-01 02:00:00', '2021-10-08 07:00:00', 
                          '2020-01-08 11:00:00', '2020-02-01 11:00:00', 
                          '2020-02-01 11:00:00', '2020-02-01 10:00:00',
                          '2020-01-01 07:00:00', '2020-01-02 07:00:00', '2020-01-03 07:00:00'],
            'createddate': ['2020-01-01 01:15:00', '2020-01-01 02:15:00', '2021-10-08 07:15:00', 
                          '2020-01-08 11:15:00', '2020-02-01 11:15:00', 
                          '2020-02-03 15:00:00', '2020-02-01 16:00:00',
                          '2020-01-01 07:15:00', '2020-01-05 07:00:00', '2020-01-04 07:00:00'
                          ],
            'event_value': ['A', 'A', 'A', 
                           'B', 'C',
                           'C', 'A',
                           'E', 'D', 'F'],
            'group': ['X', 'Y', 'Y',
                      'X', 'X',
                      'Y', 'Y',
                      'Y', 'X', 'X'
                      ]          
        })

        df[['eventdate', 'createddate']] = df[['eventdate', 'createddate']].apply(pd.to_datetime, axis=1)

        # run the function
        obj = AssessmentFeatures(
            census_df = census,
            assessments_df = None,
            config = None,
            training=False
        )
        df_result = obj.conditional_get_last_values(
            df = df,
            prefix = 'test',
            event_date_column = 'eventdate',
            event_reported_date_column = 'createddate',
            event_deleted_date_column = None,
            value_columns = ['event_value'],
            groupby_column = 'group',
            missing_event_dates = 'drop',
            n = 2,
            reset = None
        )

        # create df_expected 
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'],
            'facilityid': [1] * 15,
            'test_X_event_value_1st_previous_value': ['A', 'A', 'A',
                                                      np.nan, 'B', 'C',
                                                      np.nan, np.nan, np.nan,
                                                      np.nan, np.nan, np.nan, 'F', 'F', 'F'
                                                    ],
            'test_Y_event_value_1st_previous_value': ['A', 'A', 'A',
                                                      np.nan, np.nan, np.nan,
                                                      np.nan, 'A', 'C',
                                                      'E', 'E', 'E', 'E', 'E', 'E'
                                                    ],
            'test_X_event_value_2nd_previous_value': [np.nan, np.nan, np.nan,
                                                      np.nan, np.nan, 'B',
                                                      np.nan, np.nan, np.nan,
                                                      np.nan, np.nan, np.nan, np.nan, 'D', 'D'
                                                    ],
            'test_Y_event_value_2nd_previous_value': [np.nan, np.nan, 'A',
                                                      np.nan, np.nan, np.nan,
                                                      np.nan, np.nan, 'A',
                                                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                                    ]
        })
        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)

    def test_conditional_get_last_values_job5(self):
        census = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'facilityid': [1] * 15,
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'
                           ],
        })
        census['censusdate'] = pd.to_datetime(census['censusdate'])

        # create another dataframe with event data containing a masterpatientid, eventdate, createddate, and event_type
        df = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 
                                3, 3,
                                4, 4, 4],
            'eventdate': ['2020-01-01 01:00:00', '2020-01-01 02:00:00', '2021-10-08 07:00:00', 
                          '2020-01-08 11:00:00', '2020-02-01 11:00:00', 
                          '2020-02-01 11:00:00', '2020-02-01 10:00:00',
                          '2020-01-01 07:00:00', '2020-01-02 07:00:00', '2020-01-03 07:00:00'],
            'createddate': ['2020-01-01 01:15:00', '2020-01-01 02:15:00', '2021-10-08 07:15:00', 
                          '2020-01-08 11:15:00', '2020-02-01 11:15:00', 
                          '2020-02-03 15:00:00', '2020-02-01 16:00:00',
                          '2020-01-01 07:15:00', '2020-01-05 07:00:00', '2020-01-04 07:00:00'
                          ],
            'event_value': ['A', 'A', 'A', 
                           'B', 'C',
                           'C', 'A',
                           'E', 'D', 'F']            
        })

        df[['eventdate', 'createddate']] = df[['eventdate', 'createddate']].apply(pd.to_datetime, axis=1)

        reset = pd.MultiIndex.from_arrays([
            [1, 3, 4], 
            pd.to_datetime(['2021-10-05 10:00:00', '2020-03-10 18:00:00', '2020-01-05 16:00:00']),
            pd.to_datetime(['2021-10-06 06:30:00', '2020-03-11 02:00:00', '2020-01-06 10:00:00'])
        ], names=['masterpatientid', 'eventdate', 'createddate'])

        # run the function
        obj = AssessmentFeatures(
            census_df = census,
            assessments_df = None,
            config = None,
            training=False
        )
        df_result = obj.conditional_get_last_values(
            df = df,
            prefix = 'test',
            event_date_column = 'eventdate',
            event_reported_date_column = 'createddate',
            event_deleted_date_column = None,
            value_columns = ['event_value'],
            groupby_column = None,
            missing_event_dates = 'drop',
            n = 2,
            reset = reset
        )

        # create df_expected 
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 
                                2, 2, 2, 
                                3, 3, 3,
                                4, 4, 4, 4, 4, 4],
            'censusdate': ['2020-01-01', '2020-01-02', '2021-10-09', 
                           '2020-01-01', '2020-01-08', '2020-02-01', 
                           '2020-01-01', '2020-02-01', '2020-03-10',
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'],
            'facilityid': [1] * 15,
            'test_event_value_1st_previous_value': ['A', 'A', 'A',
                                                    np.nan, 'B', 'C',
                                                    np.nan, 'A', 'C',
                                                    'E', 'E', 'E', 'F', 'F', np.nan
                                                    ],
            'test_event_value_2nd_previous_value': ['A', 'A', np.nan,
                                                    np.nan, np.nan, 'B',
                                                    np.nan, np.nan, 'A',
                                                    np.nan, np.nan, np.nan, 'E', 'D', np.nan
                                                    ]
        })
        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)

    def test_conditional_get_last_values_job6(self):
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
                            ],
            'group': ['X',  'X', 'XY', 'XY', 'XY',  'Y', 'XY', 'XY',
                      'X', 'XY',  'Y',  'X',  'X', 'XY',
                      'Y',  'Y',  'X', 'XY',  'X',
                      'Y',  'Y',  'Y',  'Y',  'Y', 'XY', 'XY',  'X', 'XY', 'X', 'Y'
                     ]          
        })

        df[['eventdate', 'createddate', 'deleteddate']] = df[['eventdate', 'createddate', 'deleteddate']].apply(pd.to_datetime, axis=1)

        reset = pd.MultiIndex.from_arrays([
            [1, 3, 4], 
            pd.to_datetime(['2020-01-09 10:00:00', '2020-01-10 18:00:00', '2020-01-14 16:00:00']),
            pd.to_datetime(['2020-01-11 06:30:00', '2020-01-11 02:00:00', '2020-01-14 18:00:00'])
        ], names=['masterpatientid', 'eventdate', 'createddate'])

        # run the function
        obj = AssessmentFeatures(
            census_df = census,
            assessments_df = None,
            config = None,
            training=False
        )
        df_result = obj.conditional_get_last_values(
            df = df,
            prefix = 'test',
            event_date_column = 'eventdate',
            event_reported_date_column = 'createddate',
            event_deleted_date_column = 'deleteddate',
            value_columns = ['event_value', 'event_value2'],
            groupby_column = 'group',
            missing_event_dates = 'drop',
            n = 2,
            reset = reset
        )

        # create df_expected 
        nan = np.nan
        n = np.nan
        df_expected = pd.DataFrame({
            'masterpatientid': [1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2,
                                3, 3, 3, 3, 3, 3, 3, 3,
                                4, 4, 4, 4, 4, 4, 4, 4],
            'censusdate': ['2020-01-01', '2020-01-02', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', '2020-01-13', '2020-01-14', 
                           '2020-01-02', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12', 
                           '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-10', '2020-01-11',
                           '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-08', '2020-01-09', '2020-01-14'
                           ],
            'facilityid': [1] * 32,
            'test_X_event_value_1st_previous_value': [nan, 'C', 'D', 'D', nan, nan, nan, nan,
                                                      nan, 'C', 'C', 'C', 'C', 'C', 'C', 'E',
                                                      nan, nan, nan, nan, nan, nan, 'A', nan,
                                                      nan, nan, nan, nan, nan, nan, 'B', 'G'
                                                     ],
            'test_XY_event_value_1st_previous_value': [nan, nan, nan, nan, 'D', 'D', 'D', 'E',
                                                       nan, nan, nan, nan, nan, 'F', 'F', 'F',
                                                       nan, nan, nan, nan, nan, nan, nan, nan, 
                                                       nan, nan, nan, nan, nan, nan, 'G', nan
                                                      ],
            'test_Y_event_value_1st_previous_value': [nan, nan, nan, nan, nan, nan, nan, 'D',
                                                      nan, nan, nan, nan, nan, nan, nan, 'E',
                                                      nan, nan, nan, nan, 'D', 'D', 'G', nan, 
                                                      nan, nan, 'D', 'D', 'F', 'F', 'B', nan
                                                     ],
            
            'test_X_event_value_2nd_previous_value': [nan, nan, nan, nan, nan, nan, nan, nan, 
                                                      nan, nan, nan, nan, nan, nan, nan, 'C',
                                                      nan, nan, nan, nan, nan, nan, 'E', nan,
                                                      nan, nan, nan, nan, nan, nan, nan, nan
                                                     ],
            'test_XY_event_value_2nd_previous_value': [nan, nan, nan, nan, nan, nan, nan, 'F',
                                                       nan, nan, nan, nan, nan, nan, nan, 'A',
                                                       nan, nan, nan, nan, nan, nan, nan, nan, 
                                                       nan, nan, nan, nan, nan, nan, nan, nan 
                                                      ],
            'test_Y_event_value_2nd_previous_value': [nan, nan, nan, nan, nan, nan, nan, nan, 
                                                      nan, nan, nan, nan, nan, nan, nan, nan, 
                                                      nan, nan, nan, nan, nan, nan, 'D', nan, 
                                                      nan, nan, nan, 'B', 'D', 'D', 'F', nan
                                                     ],
            
            'test_X_event_value2_1st_previous_value': [n, 8, 4, 4, n, n, n, n,
                                                       n, 0, 0, 0, 0, 0, 0, 9,
                                                       n, n, n, n, n, n, 4, n, 
                                                       n, n, n, n, n, n, 6, 4
                                                      ],
            'test_XY_event_value2_1st_previous_value': [n, n, n, n,10,10,10, 2,
                                                        n, n, n, n, n, 0, 0, 0,
                                                        n, n, n, n, n, n, n, n, 
                                                        n, n, n, n, n, n,10, n
                                                       ],
            'test_Y_event_value2_1st_previous_value': [n, n, n, n, n, n, n,10,
                                                       n, n, n, n, n, n, n, 7,
                                                       n, n, n, n, 0, 0, 5, n,
                                                       n, n,10,10, 6, 6,10, n
                                                      ],
            
            'test_X_event_value2_2nd_previous_value': [n, n, n, n, n, n, n, n, 
                                                       n, n, n, n, n, n, n, 0,
                                                       n, n, n, n, n, n, 6, n,
                                                       n, n, n, n, n, n, n, n
                                                      ],
            'test_XY_event_value2_2nd_previous_value': [n, n, n, n, n, n, n,10,
                                                        n, n, n, n, n, n, n, 1,
                                                        n, n, n, n, n, n, n, n,
                                                        n, n, n, n, n, n, n, n 
                                                       ],
            'test_Y_event_value2_2nd_previous_value': [n, n, n, n, n, n, n, n, 
                                                       n, n, n, n, n, n, n, n, 
                                                       n, n, n, n, n, n, 0, n,
                                                       n, n, n, 9,10,10, 6, n
                                                      ]
            
        })
        df_expected['censusdate'] = pd.to_datetime(df_expected['censusdate'])

        # assert that the two dataframes are equal
        assert_frame_equal(df_result, df_expected, check_dtype=False)

if __name__ == '__main__':
    unittest.main()