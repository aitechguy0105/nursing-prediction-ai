import logging
from pathlib import Path
import pickle
import typing
import sys
import json
from dataclasses import asdict

import fire
import mlflow
from eliot import to_file
import pandas as pd

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from saiva.training import IdensDataset, load_config, train_optuna_integration
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ClientConfiguration, ExperimentDates
from src.training_pipeline.shared.utils import convert_input_params_decorator, load_x_y_idens_training_pipeline, setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.TRAIN_MODEL


@convert_input_params_decorator
def train_model(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    model_type: typing.Optional[str] = 'MODEL_UPT',
    hyper_parameter_tuning: typing.Optional[bool] = True,
    force_regenerate: typing.Optional[bool] = False,
    optuna_time_budget: typing.Optional[int] = None,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs,
):
    """Preprocess the data.

        :param run_id: the run id
        :param client_configurations: list of all client configurations
        :param model_type: the type of the model to be trained (MODEL_UPT or MODEL_FALL)
        :param hyper_parameter_tuning: whether to use hyper parameter tuning
        :param optuna_time_budget: the optuna time budget
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    model_type = model_type.lower()
    client = "+".join([c.client for c in client_configurations])

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP), prefix=f'/{model_type}')
    config = load_config(LOCAL_TRAINING_CONFIG_PATH)
    training_config = config.training_config

    if not optuna_time_budget:
        optuna_time_budget = training_config.training_metadata.optuna_time_budget

    # the time budget shoud be maximum 46 hours due to EC2 instance limit
    optuna_time_budget = min(optuna_time_budget, 46 * 60 * 60)
    log.info(f'optuna_time_budget (max 46 hours): {optuna_time_budget}')

    if dataset_provider.does_file_exist(filename=f'{model_type}/model_config', step=CURRENT_STEP, file_format='json'):
        log.info("Model already trained. Skipping step")
        return

    ### ========= Set the CONFIG & hyper_parameter_tuning in constants.py ==========

    experiment_dates = asdict(ExperimentDates(**training_config.training_metadata.experiment_dates.dates_calculation))

    TRAINING_DATA = client  # trained on which data? e.g. avante + champion
    SELECTED_MODEL_VERSION = f'saiva-3-day-{model_type}-v6'  # e.g. v3, v4 or v6 model

    # Name used to filter models in AWS quicksight & also used as ML Flow experiment name
    MODEL_DESCRIPTION = f'AUTOMATED-{run_id}-{client}-{SELECTED_MODEL_VERSION}'  # e.g. 'avante-upt-v6-model'

    log.info(f'model_type: {model_type}')
    log.info(f'hyper_parameter_tuning: {hyper_parameter_tuning}')
    log.info(f'client: {client}')

    ## ============ Initialise MLFlow Experiment =============

    # Create an ML-flow experiment
    mlflow.set_tracking_uri('http://mlflow.saiva-dev')

    # Experiment name which appears in ML flow
    mlflow.set_experiment(MODEL_DESCRIPTION)

    EXPERIMENT = mlflow.get_experiment_by_name(MODEL_DESCRIPTION)
    EXPERIMENT_ID = EXPERIMENT.experiment_id

    log.info(f'Experiment ID: {EXPERIMENT_ID}')

    ## =================== Loading data ======================

    log.info('Loading data...')

    train_x, train_target_3_day, train_idens = load_x_y_idens_training_pipeline(
        dataset_provider=dataset_provider,
        model_type=model_type,
        data_split='train',
    )

    valid_x, valid_target_3_day, valid_idens = load_x_y_idens_training_pipeline(
        dataset_provider=dataset_provider,
        model_type=model_type,
        data_split='valid',
    )

    test_x, test_target_3_day, test_idens = load_x_y_idens_training_pipeline(
        dataset_provider=dataset_provider,
        model_type=model_type,
        data_split='test',
    )

    cate_columns = dataset_provider.load_pickle(filename=f'{model_type}/cate_columns', step=TrainingStep.DATASETS_GENERATION)
    feature_names = dataset_provider.load_pickle(filename=f'{model_type}/feature_names', step=TrainingStep.DATASETS_GENERATION)
    pandas_categorical = dataset_provider.load_pickle(filename=f'{model_type}/pandas_categorical', step=TrainingStep.DATASETS_GENERATION)

    log.info('Train data shape: %s', train_x.shape)
    log.info('Train target shape: %s', train_target_3_day.shape)
    log.info('Train idens shape: %s', train_idens.shape)
    log.info('Valid data shape: %s', valid_x.shape)
    log.info('Valid target shape: %s', valid_target_3_day.shape)
    log.info('Valid idens shape: %s', valid_idens.shape)
    log.info('Test data shape: %s', test_x.shape)
    log.info('Test target shape: %s', test_target_3_day.shape)
    log.info('Test idens shape: %s', test_idens.shape)

    log.info('cat columns: %s', len(cate_columns))
    log.info('feature names: %s', len(feature_names))
    log.info('pandas_categorical: %s', len(pandas_categorical))

    info_cols = config.automatic_training.datasets_generation.iden_cols + [f'positive_date_{model_type}', 'long_short_term']

    train_data = IdensDataset(
        train_x,
        label=train_target_3_day,
        idens=train_idens.loc[:, info_cols],
        feature_name=feature_names,
        categorical_feature=cate_columns
    )
    valid_data = IdensDataset(
        valid_x,
        label=valid_target_3_day,
        idens=valid_idens.loc[:, info_cols],
        feature_name=feature_names,
        categorical_feature=cate_columns
    )
    test_data = IdensDataset(
        test_x,
        label=test_target_3_day,
        idens=test_idens.loc[:, info_cols],
        feature_name=feature_names,
        categorical_feature=cate_columns
    )

    ## =================== Model Training ===================

    # We have a new training method. After calling it, wait for 5 minutes, make sure everything is working properly. If there are no issues, you can start doing something else. Typically, this process takes around 12-24 hours (depending on the size of the dataset), and you can track the results through mlflow.

    params = {
        "seed": 1,
        "metric": "auc",
        "verbosity": 5,
        "boosting_type": "gbdt",
    }

    feature_cumulative_drop_stats = dataset_provider.load_txt(filename=f'{model_type}/feature_cumulative_drop_stats', step=TrainingStep.FEATURE_SELECTION)
    feature_group_drop_stats = dataset_provider.load_txt(filename=f'{model_type}/feature_group_drop_stats', step=TrainingStep.FEATURE_SELECTION)
    all_null_values_drop_list = dataset_provider.load_json(filename=f'{model_type}/all_null_dropped_col_names', step=TrainingStep.FEATURE_SELECTION)

    with open('feature_cumulative_drop_stats.txt', 'wb') as f:
        pickle.dump(feature_cumulative_drop_stats, f, protocol=4)

    with open('feature_group_drop_stats.txt', 'wb') as f:
        pickle.dump(feature_group_drop_stats, f, protocol=4)

    with open('cate_columns.pickle', 'wb') as f:
        pickle.dump(cate_columns, f, protocol=4)

    with open('all_null_dropped_col_names.json', 'w') as f:
        json.dump(all_null_values_drop_list, f)

    # we set the client to the first client in the list here (it doesn't matter which client we use)
    client = client_configurations[0].client

    base_model = train_optuna_integration(
        params,
        train_data,
        valid_data,
        test_data,
        training_config.training_metadata.vector_model,
        model_type,
        experiment_dates,
        hyper_parameter_tuning,
        TRAINING_DATA,
        SELECTED_MODEL_VERSION,
        MODEL_DESCRIPTION,
        EXPERIMENT_ID,
        optuna_time_budget,
        pandas_categorical,
        config
    )

    # Upload files generated by the training process to S3
    for dataset in ['train', 'valid', 'test']:
        performance_df = pd.read_csv(f'performance_{dataset}_base.csv')
        dataset_provider.store_df_csv(dataset_name=f'{model_type}/performance_{dataset}_base', step=CURRENT_STEP, df=performance_df)
        try:
            duplicate_rows_performance_df = pd.read_csv(f'duplicate_rows_performance_{dataset}_base.csv')
            dataset_provider.store_df_csv(dataset_name=f'{model_type}/duplicate_rows_performance_{dataset}_base', step=CURRENT_STEP, df=duplicate_rows_performance_df)
        except FileNotFoundError:
            log.info(f'No duplicate rows performance file found for {dataset} dataset')

    # Upload trial data to S3
    try:
        trial_data = pd.read_csv('trial_data.csv')
        dataset_provider.store_df_csv(dataset_name=f'{model_type}/trial_data', step=CURRENT_STEP, df=trial_data)
    except FileNotFoundError:
        log.info('No trial data file found')

    # Upload model pickle to S3
    with open(f'./{base_model.model_name}.pickle', 'rb') as f:
        model = pickle.load(f)
    dataset_provider.store_pickle(filename=f'{model_type}/{base_model.model_name}', step=CURRENT_STEP, data=model)

    dataset_provider.store_json(filename=f'{model_type}/model_config', step=CURRENT_STEP, data=base_model.config['metadata'])
    dataset_provider.store_config(step=CURRENT_STEP, prefix=f'/{model_type}')


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(train_model)
