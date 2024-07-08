import datetime
import logging
import typing
import sys

from omegaconf import OmegaConf
import fire
import pandas as pd
from eliot import to_file

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.saiva.training.utils import load_config
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ClientConfiguration, ExperimentDates
from src.training_pipeline.shared.utils import convert_dates_to_str, convert_input_params_decorator, setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.DATES_CALCULATION


def get_prior_date_as_str(date_as_str):
    prior_date = pd.to_datetime(date_as_str) - datetime.timedelta(days=1)
    prior_date_as_str = prior_date.date().strftime('%Y-%m-%d')
    return prior_date_as_str


@convert_input_params_decorator
def dates_calculation(*,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    force_regenerate: typing.Optional[bool] = False,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Calculate train_end_date, validation_start_date, validation_end_date and test_start_date.

        :param run_id: the run id
        :param client_configurations: list of all client configurations
        :param force_regenerate: force the regeneration of the data
    """
    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=CURRENT_STEP)

    config = load_config(LOCAL_TRAINING_CONFIG_PATH)
    dates_calculation_experiment_dates = config.training_config.training_metadata.experiment_dates.get('dates_calculation', {})

    if not force_regenerate and dates_calculation_experiment_dates:
        log.info("Experiment dates already calculated. Skipping experiment dates calculation step")
        return

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP))
    config = load_config(LOCAL_TRAINING_CONFIG_PATH)
    training_metadata = config.training_config.training_metadata

    date_range = training_metadata.experiment_dates.calculate_date_range

    if date_range.get('train_end_date') and date_range.get('validation_start_date') and date_range.get('validation_end_date') and date_range.get('test_start_date'):
        log.info("Experiment dates provided. Skipping experiment dates calculation step.")

        dates_calculation_experiment_dates = date_range

        # Assert that we have the required number of days in each date range
        if not (pd.to_datetime(date_range.get('validation_end_date')) - pd.to_datetime(date_range.get('validation_start_date'))).days >= 60:
            log.warning('Validation set should be at least 60 days')
        if not (pd.to_datetime(date_range.get('test_end_date')) - pd.to_datetime(date_range.get('test_start_date'))).days >= 60:
            log.warning('Test set should be at least 60 days')
    else:
        patient_census_df = dataset_provider.get(dataset_name='patient_census', step=TrainingStep.MERGE_DATA)

        patient_census_df.drop_duplicates(
            subset=['masterpatientid', 'censusdate'],
            keep='last',
            inplace=True
        )
        patient_census_df.sort_values(by=['censusdate'], inplace=True)

        total_count = patient_census_df.shape[0]
        test_count = int(total_count * 0.25)
        test_split_count = int(test_count * 0.5)  # split between validation & test set

        validation_start_idx = total_count - test_count
        test_start_idx = validation_start_idx + test_split_count

        train_start_date = patient_census_df.iloc[0].censusdate.date().strftime('%Y-%m-%d')
        validation_start_date = patient_census_df.iloc[validation_start_idx].censusdate.date().strftime('%Y-%m-%d')
        test_start_date = patient_census_df.iloc[test_start_idx].censusdate.date().strftime('%Y-%m-%d')
        test_end_date = patient_census_df.iloc[-1].censusdate.date().strftime('%Y-%m-%d')

        train_end_date = get_prior_date_as_str(validation_start_date)
        validation_end_date = get_prior_date_as_str(test_start_date)

        log.info(f'train_start_date: {train_start_date}')
        log.info(f'train_end_date: {train_end_date}')
        log.info(f'validation_start_date: {validation_start_date}')
        log.info(f'validation_end_date: {validation_end_date}')
        log.info(f'test_start_date: {test_start_date}')
        log.info(f'test_end_date: {test_end_date}')

        # # Assert that we have the required number of days in each date range
        assert (pd.to_datetime(validation_end_date) - pd.to_datetime(validation_start_date)).days >= 60, 'Validation set must be at least 60 days'
        assert (pd.to_datetime(test_end_date) - pd.to_datetime(test_start_date)).days >= 60, 'Test set must be at least 60 days'

        experiment_dates = ExperimentDates(
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date
        )

        dates_calculation_experiment_dates = convert_dates_to_str(experiment_dates=experiment_dates)

    for client_config in client_configurations:
        client = client_config.client
        if client_config.experiment_dates:
            if client_config.experiment_dates.train_end_date:
                client_overrides = dates_calculation_experiment_dates.get('client_overrides', {})
                client_override = client_overrides.get(client, {})
                client_override['default'] = convert_dates_to_str(experiment_dates=client_config.experiment_dates)
                client_overrides[client] = client_override
                dates_calculation_experiment_dates['client_overrides'] = client_overrides
        if client_config.experiment_dates_facility_wise_overrides:
            client_overrides = dates_calculation_experiment_dates.get('client_overrides', {})
            client_override = client_overrides.get(client, {})
            for facility, experiment_dates in client_config.experiment_dates_facility_wise_overrides.items():
                if experiment_dates.train_end_date:
                    client_override[facility] = convert_dates_to_str(experiment_dates=experiment_dates)
                    client_overrides[client] = client_override
                    dates_calculation_experiment_dates['client_overrides'] = client_overrides

    log.info(f'Experiment dates: {dates_calculation_experiment_dates}')
    training_metadata.experiment_dates.dates_calculation = dates_calculation_experiment_dates
        
    conf = OmegaConf.create({'training_config': {'training_metadata': training_metadata}})
    OmegaConf.save(conf, f'{LOCAL_TRAINING_CONFIG_PATH}generated/training_metadata.yaml')
    
    dataset_provider.store_config(step=CURRENT_STEP)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(dates_calculation)
