import datetime
import gc
import logging
import sys
import typing
from dataclasses import asdict
from datetime import timedelta

import fire
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from eliot import to_file

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.saiva.model.shared.utils import url_encode_cols
from src.saiva.training import load_config
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ExperimentDates
from src.training_pipeline.shared.utils import setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.DATASETS_GENERATION


# Remove the Target values & identification columns
# Keep facilityid in idens and add a duplicate field as facility for featurisation
def prep(*, df: pd.DataFrame, iden_cols: typing.List[str], model_type: str, feature_names=None, category_columns=None, pandas_categorical=None):
    df.reset_index(drop=True, inplace=True)
    drop_cols = iden_cols + [col for col in df.columns if 'target' in col]
    drop_cols += [col for col in df.columns if 'positive_date_' in col]

    target_3_day = df[f'target_3_day_{model_type}'].astype('float32').values

    df['facility'] = df['facilityid']
    df['facility'] = df['facility'].astype('category')

    x = df.drop(columns=drop_cols).reset_index(drop=True)

    if feature_names is None:
        feature_names = x.columns.tolist()
    elif (len(x.columns) != len(feature_names)):
        raise ValueError("train and valid dataset feature names do not match")
    elif (x.columns != feature_names).any():
        x = x.reindex(columns=feature_names)

    if category_columns is None:
        category_columns = x.dtypes[x.dtypes == 'category'].index.tolist()

    if pandas_categorical is None:
        pandas_categorical = [list(x[col].cat.categories) for col in category_columns]

    else:
        if len(category_columns) != len(pandas_categorical):
            raise ValueError("train and valid dataset categorical_feature do not match")
        for col, category in zip(category_columns, pandas_categorical):
            if list(x[col].cat.categories) != list(category):
                x[col] = x[col].cat.set_categories(category)

    idens = df.loc[:, iden_cols]
    # add 'long_short_term' column to indens
    short_term_cond = ((x.payertype == 'Managed Care') | (x.payertype == 'Medicare A'))
    idens.loc[short_term_cond, 'long_short_term'] = 'short'

    nonType_cond = (x.payertype == 'no payer info')
    if nonType_cond.sum() > 0:
        print(f'{nonType_cond.sum()} patient days have no payer info')
        idens.loc[nonType_cond, 'long_short_term'] = 'no payer info'

    idens.loc[~(short_term_cond | nonType_cond), 'long_short_term'] = 'long'

    # converting to numpy array
    x[category_columns] = x[category_columns].apply(lambda col: col.cat.codes).replace({-1: np.nan})
    x = x.to_numpy(dtype=np.float32, na_value=np.nan)

    return x, target_3_day, idens, feature_names, category_columns, pandas_categorical


def drop_unwanted_columns(*, df: pd.DataFrame, model_type: str):
    positive_date = f'positive_date_{model_type}'
    target_3_day = f'target_3_day_{model_type}'
    drop_columns = ['dateofadmission']
    dates = list(df.columns[df.columns.str.contains('positive_date_')])
    targets = list(df.columns[df.columns.str.contains('target_3_day_')])
    for date in dates:
        if date != positive_date:
            drop_columns.append(date)
    for target in targets:
        if target != target_3_day:
            drop_columns.append(target)
    df = df.drop(columns=drop_columns, errors='ignore')
    return df


def update_experiment_dates(experiment_dates_config: ExperimentDates, hyper_parameter_tuning: bool):
    experiment_dates = asdict(experiment_dates_config)
    experiment_dates['train_start_date'] = (pd.to_datetime(experiment_dates_config.train_start_date) + pd.DateOffset(days=31)).date()

    if not hyper_parameter_tuning:
        experiment_dates['train_end_date'] = (experiment_dates.validation_end_date - timedelta(days=2)).strftime('%Y-%m-%d')
        experiment_dates['validation_start_date'] = (experiment_dates.validation_end_date - timedelta(days=1)).strftime('%Y-%m-%d')

    return experiment_dates


def generate_train_valid_test_sets(
    *,
    df: pd.DataFrame,
    experiment_dates: typing.Dict[str, datetime.date],
) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[(df.censusdate.dt.date >= experiment_dates['train_start_date']) & (df.censusdate.dt.date <= experiment_dates['train_end_date'])]
    valid = df.loc[(df.censusdate.dt.date >= experiment_dates['validation_start_date']) & (df.censusdate.dt.date <= experiment_dates['validation_end_date'])]
    test = df.loc[df.censusdate.dt.date >= experiment_dates['test_start_date']]

    def sort_group(group: pd.DataFrame) -> pd.DataFrame:
        return group.sort_values('masterpatientid')
    valid = valid.groupby(['facilityid', 'censusdate']).apply(sort_group).reset_index(drop=True)
    valid.reset_index(drop=True, inplace=True)
    test = test.groupby(['facilityid', 'censusdate']).apply(sort_group).reset_index(drop=True)
    test.reset_index(drop=True, inplace=True)
    return train, valid, test


def datasets_generation(
    *,
    run_id: str,
    model_type: typing.Optional[str] = 'MODEL_UPT',
    hyper_parameter_tuning: typing.Optional[bool] = True,
    force_regenerate: typing.Optional[bool] = False,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Generate datasets for training, validation and testing.

        :param run_id: the run id
        :param model_type: the model type
        :param hyper_parameter_tuning: whether to use hyper parameter tuning
        :param force_regenerate: force the regeneration of the data
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    model_type = model_type.lower()

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP), prefix=f"/{model_type}")
    config = load_config(LOCAL_TRAINING_CONFIG_PATH)
    training_metadata = config.training_config.training_metadata

    final_datasets = [
        f'final-train_x_{model_type}',
        f'final-train_target_3_day_{model_type}',
        f'final-valid_x_{model_type}',
        f'final-valid_target_3_day_{model_type}',
        f'final-test_x_{model_type}',
        f'final-test_target_3_day_{model_type}',
        'cate_columns',
        'pandas_categorical',
        'feature_names',
        f'final-train_idens_{model_type}',
        f'final-valid_idens_{model_type}',
        f'final-test_idens_{model_type}'
    ]

    if all(
        [dataset_provider.does_file_exist(
            filename=f"{model_type}/" + filename,
            step=CURRENT_STEP,
            file_format="pickle"
        ) for filename in final_datasets]
    ):
        log.info('All datasets exist - skipping datasets generation step.')
        return

    log.info(f'MODEL: {model_type}')

    experiment_dates_config = training_metadata.experiment_dates.dates_calculation

    # starting training from day 31 so that cumsum window 2,7,14,30 are all initial correct.
    # One day will be added to `censusdate` later in the code, so that the first date in
    # `train` will be `experiment_dates['train_start_date'] + 1 day`, that's why here we
    # add 31 days but not 30

    experiment_dates = {}
    experiment_dates['default'] = update_experiment_dates(ExperimentDates(
        train_start_date=experiment_dates_config.train_start_date,
        train_end_date=experiment_dates_config.train_end_date,
        validation_start_date=experiment_dates_config.validation_start_date,
        validation_end_date=experiment_dates_config.validation_end_date,
        test_start_date=experiment_dates_config.test_start_date,
        test_end_date=experiment_dates_config.test_end_date
    ), hyper_parameter_tuning)

    if experiment_dates_config.get('client_overrides', None):
        experiment_dates['client_overrides'] = {}
        for client, experiment_dates_client_override in experiment_dates_config['client_overrides'].items():
            experiment_dates['client_overrides'][client] = {}
            for key, experiment_dates_override in experiment_dates_client_override.items():
                modified_experiment_dates = update_experiment_dates(ExperimentDates(**experiment_dates_override), hyper_parameter_tuning)
                if key == 'default':
                    experiment_dates['client_overrides'][client]['default'] = modified_experiment_dates
                else:
                    facility_wise_overrides = experiment_dates['client_overrides'][client].get('facility_wise_overrides', {})
                    facility_wise_overrides[f"{client}_{key}"] = modified_experiment_dates
                    experiment_dates['client_overrides'][client]['facility_wise_overrides'] = facility_wise_overrides

    log.info(f'experiment_dates: {experiment_dates}')
    log.info(f'hyper_parameter_tuning: {hyper_parameter_tuning}')

    final = dataset_provider.get(dataset_name=f'{model_type}/final_cleaned_df', step=TrainingStep.FEATURE_SELECTION)

    final = url_encode_cols(final)

    assert f'target_3_day_{model_type}' in final.columns, f"There is no target for training `{model_type}` model"

    iden_cols = config.automatic_training.datasets_generation.iden_cols + [f'positive_date_{model_type}']

    # UPT model doesn't need the rows that with payername contains 'hospice'
    if model_type == 'model_upt':
        final = final[~(final['payername'].str.contains('hospice', case=False, regex=True, na=False))]

    # column processing
    final['client'] = final['masterpatientid'].apply(lambda z: z.split('_')[0])
    final["facilityid"] = final["client"] + "_" + final["facilityid"].astype(str)

    final['LFS'] = final['days_since_last_admission']

    """ We increment the census date by 1, since the prediction day always includes data upto last night.
    This means for every census date the data is upto previous night. 
    """
    log.info(f'final shape before prep: {final.shape}')

    # Increment censusdate by 1
    final['censusdate'] = (pd.to_datetime(final['censusdate']) + timedelta(days=1))

    log.info(f'final shape after prep: {final.shape}')

    # drop unwanted columns for this model
    final = drop_unwanted_columns(df=final, model_type=model_type)

    final[f'target_3_day_{model_type}'] = final[f'target_3_day_{model_type}'].fillna(False)

    x = PrettyTable()

    # manual check to make sure we're not including any columns that could leak data
    for col in final.columns:
        x.add_row([col])

    dataset_provider.store_txt(filename=f"{model_type}/columns_{model_type}", step=CURRENT_STEP, data=str(x))

    if 'client_overrides' in experiment_dates:
        train_dfs = []
        valid_dfs = []
        test_dfs = []

        for client_name in final.client.unique():
            client_df = final.loc[final.client == client_name]
            experiment_dates_client_override = experiment_dates['client_overrides'].get(client_name, experiment_dates)
            experiment_dates_override = experiment_dates_client_override.get('default', experiment_dates['default'])

            if 'facility_wise_overrides' in experiment_dates_client_override:
                for facilityid in client_df.facilityid.unique():
                    log.info(f'Generating train, valid, and test sets for client {client_name} and facility {facilityid} with experiment dates {experiment_dates_override}')

                    client_facility_df = client_df.loc[client_df.facilityid == facilityid]
                    experiment_dates_override = experiment_dates_client_override['facility_wise_overrides'].get(facilityid, experiment_dates_override)
                    train_facility_df, valid_facility_df, test_facility_df = generate_train_valid_test_sets(df=client_facility_df, experiment_dates=experiment_dates_override)
                    train_dfs.append(train_facility_df)
                    valid_dfs.append(valid_facility_df)
                    test_dfs.append(test_facility_df)
            else:
                log.info(f'Generating train, valid, and test sets for client {client_name} with experiment dates {experiment_dates_override}')

                train_client_df, valid_client_df, test_client_df = generate_train_valid_test_sets(df=client_df, experiment_dates=experiment_dates_override)
                train_dfs.append(train_client_df)
                valid_dfs.append(valid_client_df)
                test_dfs.append(test_client_df)

        train = pd.concat(train_dfs)
        valid = pd.concat(valid_dfs)
        test = pd.concat(test_dfs)
    else:
        train, valid, test = generate_train_valid_test_sets(df=final, experiment_dates=experiment_dates['default'])

    log.info(f'final shape: {final.shape}')
    log.info(f'train shape: {train.shape}')
    log.info(f'valid shape: {valid.shape}')
    log.info(f'test shape: {test.shape}')

    del final
    gc.collect()

    log.info('Target columns:')
    for col in train.columns:
        if 'target_3_day' in col:
            log.info(col)

    # start of basic tests - assert we have disjoint sets over time
    assert train.censusdate.max() < valid.censusdate.min()
    assert valid.censusdate.max() < test.censusdate.min()

    log.info(f'Train set covers {train.censusdate.min()} to {train.censusdate.max()} with 3_day_{model_type} percentage {train[f"target_3_day_{model_type}"].mean()}')
    log.info(f'Valid set covers {valid.censusdate.min()} to {valid.censusdate.max()} with 3_day_{model_type} percentage {valid[f"target_3_day_{model_type}"].mean()}')
    log.info(f'Test set covers {test.censusdate.min()} to {test.censusdate.max()} with 3_day_{model_type} percentage {test[f"target_3_day_{model_type}"].mean()}')

    log.info('Datetime columns that are not in IDEN_COLS:')
    for col in train.columns:
        if train[col].dtypes == 'datetime64[ns]':
            if col not in iden_cols:
                log.info(f"{col}, {train[col].dtypes}", )

    # Seperate target, x-frame and identification columns
    train_x, train_target_3_day, train_idens, feature_names, cate_columns, pandas_categorical = prep(
        df=train,
        iden_cols=iden_cols,
        model_type=model_type
    )
    del train

    valid_x, valid_target_3_day, valid_idens, _, _, _ = prep(
        df=valid,
        iden_cols=iden_cols,
        model_type=model_type,
        feature_names=feature_names,
        category_columns=cate_columns,
        pandas_categorical=pandas_categorical
    )
    del valid

    test_x, test_target_3_day, test_idens, _, _, _ = prep(
        df=test,
        iden_cols=iden_cols,
        model_type=model_type,
        feature_names=feature_names,
        category_columns=cate_columns,
        pandas_categorical=pandas_categorical
    )
    del test

    gc.collect()

    # make sure for that x's, targets, an idens all have the same # of rows
    assert train_x.shape[0] == train_target_3_day.shape[0] == train_idens.shape[0]
    assert valid_x.shape[0] == valid_target_3_day.shape[0] == valid_idens.shape[0]
    assert test_x.shape[0] == test_target_3_day.shape[0] == test_idens.shape[0]

    # make sure that train, valid, and test have the same # of columns
    assert train_x.shape[1] == valid_x.shape[1] == test_x.shape[1]

    # make sure that the idens all have the same # of columns
    assert train_idens.shape[1] == valid_idens.shape[1] == test_idens.shape[1]

    # Save train, test and validation datasets

    dataset_provider.store_pickle(filename=f'{model_type}/final-train_x_{model_type}', step=CURRENT_STEP, data=train_x)
    dataset_provider.store_pickle(filename=f'{model_type}/final-train_target_3_day_{model_type}', step=CURRENT_STEP, data=train_target_3_day)
    dataset_provider.store_pickle(filename=f'{model_type}/final-train_idens_{model_type}', step=CURRENT_STEP, data=train_idens)

    dataset_provider.store_pickle(filename=f'{model_type}/final-valid_x_{model_type}', step=CURRENT_STEP, data=valid_x)
    dataset_provider.store_pickle(filename=f'{model_type}/final-valid_target_3_day_{model_type}', step=CURRENT_STEP, data=valid_target_3_day)
    dataset_provider.store_pickle(filename=f'{model_type}/final-valid_idens_{model_type}', step=CURRENT_STEP, data=valid_idens)

    dataset_provider.store_pickle(filename=f'{model_type}/final-test_x_{model_type}', step=CURRENT_STEP, data=test_x)
    dataset_provider.store_pickle(filename=f'{model_type}/final-test_target_3_day_{model_type}', step=CURRENT_STEP, data=test_target_3_day)
    dataset_provider.store_pickle(filename=f'{model_type}/final-test_idens_{model_type}', step=CURRENT_STEP, data=test_idens)

    dataset_provider.store_pickle(filename=f'{model_type}/cate_columns', step=CURRENT_STEP, data=cate_columns)
    dataset_provider.store_pickle(filename=f'{model_type}/feature_names', step=CURRENT_STEP, data=feature_names)
    dataset_provider.store_pickle(filename=f'{model_type}/pandas_categorical', step=CURRENT_STEP, data=pandas_categorical)

    log.info('Datasets generated and stored in s3 bucket')

    log.info('Train_x shape: {}'.format(train_x.shape))
    log.info('Train_target_3_day shape: {}'.format(train_target_3_day.shape))
    log.info('Train_idens shape: {}'.format(train_idens.shape))
    log.info('Valid_x shape: {}'.format(valid_x.shape))
    log.info('Valid_target_3_day shape: {}'.format(valid_target_3_day.shape))
    log.info('Valid_idens shape: {}'.format(valid_idens.shape))
    log.info('Test_x shape: {}'.format(test_x.shape))
    log.info('Test_target_3_day shape: {}'.format(test_target_3_day.shape))
    log.info('Test_idens shape: {}'.format(test_idens.shape))

    dataset_provider.store_config(step=CURRENT_STEP, prefix=f"/{model_type}")


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(datasets_generation)
