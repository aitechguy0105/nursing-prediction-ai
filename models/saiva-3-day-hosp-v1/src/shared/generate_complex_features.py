import gc
import os
import sys
import timeit
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from eliot import log_message, start_action

sys.path.insert(0, '/src')
from shared.utils import downcast_dtype, convert_to_int16


def column_segregation(combined):
    """
    - Group column names based on feature tables
    :return: lists
    """
    IDENTIFICATION_COLUMNS = ['censusdate', 'masterpatientid', 'facilityid', 'bedid',
                          'censusactioncode', 'payername', 'payercode']
    
    log_message(message_type='info', message='Group column names based on feature tables')
    vtl_cols = [col for col in combined.columns if col.startswith('vtl')]
    dx_cols = [col for col in combined.columns if col.startswith('dx')]
    med_cols = [col for col in combined.columns if col.startswith('med')]
    order_cols = [col for col in combined.columns if col.startswith('order')]
    alert_cols = [col for col in combined.columns if col.startswith('alert')]
    lab_cols = [col for col in combined.columns if col.startswith('labs__')]
    ignore_cols = [col for col in combined.columns if 'target' in col] + IDENTIFICATION_COLUMNS

    assert len(vtl_cols) != 0, 'vtl_cols list is non empty'
    assert len(dx_cols) != 0, 'dx_cols list is non empty'
    # if any of meds,order,alert,ignore cols are empty then it will not impact complex feature processing code.
    # assert len(med_cols) != 0, 'med_cols list is non empty'
    # assert len(order_cols) != 0, 'order_cols list is non empty'
    # assert len(alert_cols) != 0, 'alert_cols list is non empty'
    assert len(ignore_cols) != 0, 'ignore_cols list is non empty'

    return vtl_cols, dx_cols, med_cols, order_cols, alert_cols, lab_cols, ignore_cols


def add_datepart(df, fldname, drop=True, time=False, errors='raise'):
    """
    - making extra features from date. all the feature names are present in attr list.

    :param df: dataframe
    :param fldname: string
    :param drop: bool
    :param time: bool
    :param errors: string
    :return: dataframe
    """
    log_message(message_type='info', message=f'Extracting date features for {fldname}')
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time:
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        df[fldname + '_' + n] = getattr(fld.dt, n.lower())
    if drop:
        df.drop(fldname, axis=1, inplace=True)

    return df


def add_na_indicators(df, ignore_cols):
    """
    - removing the columns present in ignore list.
    - Add additional columns to indicate NaN indicators for each existing column.
    Later the NaN will be filled by either Median or Mean values
    """
    log_message(message_type='info', message=f'Add Na Indicators')
    # Boolean in each cell indicates whether its None
    missings = df.drop(columns=ignore_cols).isna()
    missings.columns = 'na_indictator_' + missings.columns
    missings_sums = missings.sum()

    return pd.concat([df, missings.loc[:, (missings_sums > 0)]], axis=1)


def get_derived_vitals_features(vtl_cols, df):
    """
    des:
        1. applying upper and lower threshold using quantiles to remove outliers of vital columns and make them nan.
        2. grouping the data by masterpatientid and forward filling the na values of vital columns
        and kept in separate dataframe
        3. difference between vital values is calculated between one and seven day period for each masterpatientid
        and kept in separate dtaframe.
        4. rolling average and standard deviation for a period of 7 and 14 days respectively is calculated
        for vital columns for each masterpatientid.
        5. concatenating the above dataframes using with axis=1 and returning the final dataframe.

    :param df:
    :param vtl_cols:
    :return: dataframe
    """
    log_message(message_type='info', message=f'Vitals - Find Rolling Avg..')

    # Forward fill only the vitals columns
    ffilled = df.groupby('masterpatientid')[vtl_cols].fillna(method='ffill')
    ffilled['masterpatientid'] = df.masterpatientid

    # difference between element in the same column of the previous row
    diff_1_day = ffilled.groupby('masterpatientid')[vtl_cols].diff()
    diff_1_day.columns = 'diff_1_day_' + diff_1_day.columns

    # Find the difference between current value and 7 rows prior value
    diff_7_day = ffilled.groupby('masterpatientid')[vtl_cols].diff(periods=7)
    diff_7_day.columns = 'diff_7_day_' + diff_7_day.columns

    # Mean value for window range of last 7 rows
    rolling_avg_7_day = (ffilled.groupby('masterpatientid')[vtl_cols].rolling(
        7, min_periods=1).mean().reset_index(0, drop=True))
    rolling_avg_7_day.columns = ('rol_avg_7_day_' + rolling_avg_7_day.columns)

    # Mean value for window range of last 14 rows
    rolling_avg_14_day = (
        ffilled.groupby('masterpatientid')[vtl_cols].rolling(14, min_periods=1).mean().reset_index(0, drop=True))
    rolling_avg_14_day.columns = ('rol_avg_14_day_' + rolling_avg_14_day.columns)

    rolling_std_7_day = (ffilled.groupby('masterpatientid')[vtl_cols].rolling(
        7, min_periods=1).std().reset_index(0, drop=True))
    rolling_std_7_day.columns = ('rol_std_7_day_' + rolling_std_7_day.columns)

    rolling_std_14_day = (ffilled.groupby('masterpatientid')[vtl_cols].rolling(
        14, min_periods=1).std().reset_index(0, drop=True))
    rolling_std_14_day.columns = ('rol_std_14_day_' + rolling_std_14_day.columns)

    df.loc[:, vtl_cols] = ffilled.loc[:, vtl_cols]

    # =============For Memory optimisation downcast to float32=============
    diff_1_day = downcast_dtype(diff_1_day)
    diff_7_day = downcast_dtype(diff_7_day)

    # diffs all indexed the same as original in the same order
    df = pd.concat([df, diff_1_day, diff_7_day], axis=1)

    rollings = pd.concat([rolling_avg_7_day, rolling_avg_14_day, rolling_std_7_day, rolling_std_14_day], axis=1)

    # =============For Memory optimisation downcast to float32=============
    rollings = downcast_dtype(rollings)

    # rollings were sorted so we explictly join via index
    df = df.merge(
        rollings,
        how='left',
        left_index=True,
        right_index=True
    )
    log_message(message_type='info', message='Feature generation for vital completed successfully')

    # =======================Trigger Garbage collector and free up memory============================
    del diff_1_day
    del diff_7_day
    del rolling_avg_7_day
    del rolling_std_7_day
    del rolling_avg_14_day
    del rolling_std_14_day
    gc.collect()

    return df


def get_cumsum_features(cols, df):
    """
    desc:
        1. filling diagnosis, meds, alert, order columns na values with 0 in a separate df - filled
        2. finding the cumulative sum for the above columns for all time, 7 days, 14 days and 30 days for each
        patientid and storing them in separate dataframes.
        3. concatenating the new columns with the input dataframe with axis =1 and returning it.
    :param df: dataframe
    :param dx_cols: list
    :param med_cols: list
    :param alert_cols: list
    :param order_cols: list
    :param lab_cols: list. This can be empty for few clients
    :return: dataframe
    """
    log_message(message_type='info', message=f'Meds Alerts Orders Rolling sum calculation..')

    # Rolling sum (Cumulative sum) for all time 
    log_message(message_type='info', message=f'Rolling window for all time..')
    # Filling all NaN with 0
    filled = df.groupby('masterpatientid')[cols].fillna(0)
    filled['masterpatientid'] = df.masterpatientid

    cumsum_all_time = filled.groupby('masterpatientid')[cols].cumsum()
    cumsum_all_time.columns = ('cumsum_all_' + cumsum_all_time.columns)

    # Drop existing dx_cols, med_cols, alert_cols, order_cols columns
    df = df.drop(columns=cols)

    # =============For Memory optimisation convert cumsum to int16=============
    cumsum_all_time = convert_to_int16(cumsum_all_time)

    df = pd.concat(
        [df, cumsum_all_time], axis=1
    )  # cumsum is indexed the same as original in the same order
    # =======================Trigger Garbage collector and free up memory============================
    del cumsum_all_time
    gc.collect()

    # Rolling sum (Cumulative sum) of last 7 days
    log_message(message_type='info', message=f'Rolling window for last 7 days..')
    cumsum_7_day = (
        filled.groupby('masterpatientid')[cols].rolling(7, min_periods=1).sum().reset_index(0, drop=True))
    cumsum_7_day.columns = 'cumsum_7_day_' + cumsum_7_day.columns

    log_message(message_type='info', message=f'Rolling window for last 14 days..')
    cumsum_14_day = (
        filled.groupby('masterpatientid')[cols]
            .rolling(14, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
    )
    cumsum_14_day.columns = 'cumsum_14_day_' + cumsum_14_day.columns

    log_message(message_type='info', message=f'Rolling window for last 30 days..')
    cumsum_30_day = (
        filled.groupby('masterpatientid')[cols]
            .rolling(30, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
    )
    cumsum_30_day.columns = 'cumsum_30_day_' + cumsum_30_day.columns

    rollings = pd.concat(
        [cumsum_7_day, cumsum_14_day, cumsum_30_day], axis=1
    )

    # =======================Trigger Garbage collector and free up memory============================
    log_message(message_type='info', message=f'cleaning up and reducing memory..')
    del cumsum_7_day
    del cumsum_14_day
    del cumsum_30_day
    gc.collect()

    # =============For Memory optimisation convert cumsum to int16=============
    rollings = convert_to_int16(rollings)

    df = df.merge(
        rollings, how='left', left_index=True, right_index=True
    )  # rollings were sorted so we explictly join via index

    # =======================Trigger Garbage collector and free up memory============================
    del rollings
    gc.collect()

    log_message(message_type='info', message='Feature generation for other columns completed successfully')

    return df


def process_demographic_features(df):
    """
    desc:
        1. converting the categorical columns into numerical columns.
        2. new column demo_age_in_days is calculated using difference between date of admission and birthdate.
        3. concatenating the above dataframes using with axis=1 and returning the final dataframe.

    :param df:dataframe
    :return: dataframe
    """
    log_message(message_type='info', message=f'Feature generation for demographics. One hot encoding done..')

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

    df = pd.concat([df, dummies], axis=1)

    del dummies
    gc.collect()

    return df


def parallelize_dataframe(target_func, df, cols=[]):
    # Get the total CPU cores for parallelising the jobs
    n_cores = os.cpu_count() - 2
    # Get all unique patients
    uq_patient = df.masterpatientid.unique()
    # Split the unique patients into groups as per CPU cores
    patient_grp = np.array_split(uq_patient, 35)
    df_split = []
    for grp in patient_grp:
        # Create dataframe chunks as per masterpatientid groups selected above
        # If check to handle the condition where unique masterpatientid is less than array_split value(35)        
        if len(grp) == 0:
            continue
        df_split.append(
            df[df.masterpatientid.isin(grp)]
        )

    pool = Pool(n_cores)
    # Pass static parameters (excluding iterative parameter) to the target function.
    func = partial(target_func, cols)
    # Spawn multiple process & call target function while passing iterative parameter
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def complex_feature_processing(combined, s3_location_path_prefix=None, save_outputs_in_s3=False,
                               local_folder=None, save_outputs_in_local=False):
    start_time = timeit.default_timer()

    log_message(message_type='info', message='Entering complex_feature_processing')
    assert combined.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    vtl_cols, dx_cols, med_cols, order_cols, alert_cols, lab_cols, ignore_cols = column_segregation(combined)

    # Drop rows where masterpatientid is NaN
    combined = combined[combined['masterpatientid'].notna()]

    combined = add_na_indicators(combined, ignore_cols)
    combined = add_datepart(combined, 'censusdate', drop=False)
    combined = add_datepart(combined, 'dateofbirth', drop=False)

    with start_action(action_type='generating_complex_vital_features'):
        combined = parallelize_dataframe(
            target_func=get_derived_vitals_features,
            df=combined,
            cols=vtl_cols
        )

    with start_action(action_type='generating_complex_dx_med_alert_order_lab_features'):
        cumsum_cols = dx_cols + med_cols + alert_cols + order_cols + lab_cols
        combined = parallelize_dataframe(
            target_func=get_cumsum_features,
            df=combined,
            cols=cumsum_cols
        )
    #     combined = get_derived_vitals_features(vtl_cols, combined)
    #     combined = get_cumsum_features(cumsum_cols, combined)

    with start_action(action_type='generating_demographic_features'):
        combined = process_demographic_features(combined)

    drop_cols = ['patientid', 'beddescription', 'roomratetypedescription',
                 'carelevelcode', 'gender', 'dateofbirth', 'education',
                 'citizenship', 'race', 'religion', 'state', 'primarylanguage']

    final = combined.drop(columns=drop_cols)
    final = final.reset_index(drop=True)

    save_intermediate_dfs(
        save_outputs_in_s3,
        s3_location_path_prefix,
        save_outputs_in_local,
        local_folder,
        final
    )

    assert final.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    log_message(message_type='info', Dataframe_shape=final.shape)
    log_message(message_type='info', Total_time_taken=(timeit.default_timer() - start_time))

    return final


def save_intermediate_dfs(save_outputs_in_s3, s3_location_path_prefix,
                          save_outputs_in_local, local_folder, final_df):
    """
    Save intermediate dataframes for testing and comparision
    :param save_outputs_in_s3:
    :param s3_location_path_prefix:
    :param save_outputs_in_local:
    :param local_folder:
    :param df_dict: dictionary of dataframes
    """
    if save_outputs_in_s3 and (s3_location_path_prefix is not None):
        path = s3_location_path_prefix + f'/complex_features_output.parquet'
        log_message(message_type='info', message='saving intermediate output', name='complex_features', path=path)
        final_df.to_parquet(path, index=False)

    if save_outputs_in_local and (local_folder is not None):
        path = local_folder + f'/complex_features_output.parquet'
        log_message(message_type='info', message='saving intermediate output', name='complex_features', path=path)
        final_df.to_parquet(path, index=False)
