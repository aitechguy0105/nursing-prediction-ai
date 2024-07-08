import logging
import os
import shutil
import sys
import typing

import fire
from eliot import to_file
from omegaconf import OmegaConf

from src.saiva.model.shared.constants import saiva_api, LOCAL_TRAINING_CONFIG_PATH, ENV
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.shared.utils import convert_input_params_decorator, setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.SETUP


@convert_input_params_decorator
def setup(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Setup the configuration files for the training.

    :param run_id: the run id
    :param client_configurations: list of all client configurations
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    dataset_provider = DatasetProvider(run_id=run_id)

    organization_configs = []

    for client_configuration in client_configurations:
        datasource = saiva_api.data_sources.get(datasource_id=client_configuration.datasource_id)
        if ENV == 'dev':
            datasource.source_database_credentials_secret_id = 'dev-sqlserver'
        
        organization_configs.append({
            'organization_id': client_configuration.client,
            'datasource': datasource
        })

    # We upload the config to s3 bucket at this step
    shutil.rmtree(f'{LOCAL_TRAINING_CONFIG_PATH}/generated', ignore_errors=True)
    os.makedirs(f'{LOCAL_TRAINING_CONFIG_PATH}/generated/', exist_ok=True)

    conf = OmegaConf.create({'training_config': {'organization_configs': organization_configs}})
    OmegaConf.save(conf, f'{LOCAL_TRAINING_CONFIG_PATH}generated/organization_configs.yaml')

    dataset_provider.store_config(step=CURRENT_STEP)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(setup)
