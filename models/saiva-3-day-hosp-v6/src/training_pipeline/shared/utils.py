from dataclasses import asdict
import datetime
import json
import typing
import logging

from omegaconf import OmegaConf
import boto3
import sentry_sdk
from sentry_sdk import set_tag

from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ClientConfiguration, ExperimentDates
from src.saiva.model.shared.constants import ENV, REGION_NAME


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_x_y_idens_training_pipeline(*, dataset_provider: DatasetProvider, model_type: str, data_split: str):
    x = dataset_provider.load_pickle(filename=f'{model_type}/final-{data_split}_x_{model_type}', step=TrainingStep.DATASETS_GENERATION)
    y = dataset_provider.load_pickle(filename=f'{model_type}/final-{data_split}_target_3_day_{model_type}', step=TrainingStep.DATASETS_GENERATION)
    idens = dataset_provider.load_pickle(filename=f'{model_type}/final-{data_split}_idens_{model_type}', step=TrainingStep.DATASETS_GENERATION)

    return x, y, idens


def convert_input_params_decorator(f):
    def wrapped(
        *args, 
        client_configurations: typing.List[ClientConfiguration],
        experiment_dates: typing.Optional[ExperimentDates] = None,
        **kwargs
    ):
        client_configurations = [ClientConfiguration(**c) for c in client_configurations]

        experiment_dates = ExperimentDates(**experiment_dates) if experiment_dates else ExperimentDates()

        return f(*args, client_configurations=client_configurations, experiment_dates=experiment_dates, **kwargs)
    return wrapped


def get_datasource(
    *,
    config: OmegaConf,
    datasource_id: str,
) -> OmegaConf:
    # Find the organization config in an array of organization configs
    for organization_config in config.training_config.organization_configs:
        if organization_config.datasource.id == datasource_id:
            return organization_config.datasource


def convert_dates_to_str(*, experiment_dates: typing.Union[typing.Dict[str, datetime.date], ExperimentDates]) -> typing.Dict[str, typing.Optional[str]]:
    if isinstance(experiment_dates, ExperimentDates):
        experiment_dates = asdict(experiment_dates)
    return {date_key: str(date_value) if date_value else None for date_key, date_value in experiment_dates.items()}


def get_date_range(
    *,
    config: OmegaConf,
    client_configuration: ClientConfiguration
) -> typing.Dict[str, datetime.date]:

    date_range = OmegaConf.to_container(config.training_config.training_metadata.experiment_dates.calculate_date_range, resolve=True)

    client = client_configuration.client

    if client_configuration.experiment_dates:
        if client_configuration.experiment_dates.train_start_date:
            log.info(f"Overriding train_start_date for client {client} with {client_configuration.experiment_dates.train_start_date}")
            date_range['train_start_date'] = client_configuration.experiment_dates.train_start_date
        if client_configuration.experiment_dates.test_end_date:
            log.info(f"Overriding test_end_date for client {client} with {client_configuration.experiment_dates.test_end_date}")
            date_range['test_end_date'] = client_configuration.experiment_dates.test_end_date

    log.info(f"Provided date range: {date_range['train_start_date']} - {date_range['test_end_date']}")

    return date_range


def setup_sentry(*, run_id: str, disable_sentry: typing.Optional[bool] = False, **kwargs):
    if disable_sentry:
        return

    session = boto3.session.Session()
    secrets_client = session.client(
        service_name="secretsmanager", region_name=REGION_NAME
    )
    sentry_info = json.loads(
        secrets_client.get_secret_value(SecretId="sentry")[
            "SecretString"
        ]
    )
    sentry_sdk.init(
        dsn=sentry_info['ml-dsn'],
        environment=ENV,
        traces_sample_rate=1.0
    )
    set_tag('run_id', run_id)

    for key, value in kwargs.items():
        try:
            set_tag(str(key), str(value))
        except Exception as e:
            log.error(f"Error setting tag {key} with value {value}: {e}")
            continue
