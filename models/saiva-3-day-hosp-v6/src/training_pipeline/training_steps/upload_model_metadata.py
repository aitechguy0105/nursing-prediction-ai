import logging
import typing

import fire
from omegaconf import OmegaConf

from src.saiva.model.shared.constants import saiva_api, LOCAL_TRAINING_CONFIG_PATH
from src.saiva.training import load_config
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.utils import setup_sentry


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.UPLOAD_MODEL_METADATA


def upload_model_metadata(
    *,
    run_id: str,
    model_type: typing.Optional[str] = 'MODEL_UPT',
    disable_sentry: typing.Optional[bool] = False,
    **kwargs,
):
    """Upload model metadata to saivadb.

        :param run_id: the run id
        :param model_type: the type of the model to be trained (MODEL_UPT or MODEL_FALL)
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    model_type = model_type.lower()

    dataset_provider = DatasetProvider(run_id=run_id)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP), prefix=f'/{model_type}')

    config = load_config(LOCAL_TRAINING_CONFIG_PATH)
    training_config = config.training_config

    if not dataset_provider.does_file_exist(filename=f'{model_type}/model_config', step=TrainingStep.TRAIN_MODEL, file_format='json'):
        raise Exception("Model metadata does not exist. Cannot upload model metadata to database.")

    model_config = dataset_provider.load_json(filename=f'{model_type}/model_config', step=TrainingStep.TRAIN_MODEL)

    mlflow_model_id = model_config.get('modelid')

    model_exists = False

    try:
        model = saiva_api.trained_models.get_by_mlflow_model_id(
            mlflow_model_id=mlflow_model_id
        )
        if model:
            model_exists = True
    except Exception as e:
        if str(e) == f"Could not find Trained Model with mlflow_model_id={mlflow_model_id}":
            model_exists = False
        else:
            raise e

    if model_exists:
        log.info(f"Model with mlflow_model_id {mlflow_model_id} already exists in the database.")
        return
    
    model_config['validation_start_date'] = model_config['valid_start_date']
    model_config['validation_end_date'] = model_config['valid_end_date']

    training_metadata = OmegaConf.to_container(training_config.training_metadata, resolve=True)
    training_metadata['organization_configs'] = [{
        'organization_id': config.organization_id,
        'datasource_id': config.datasource.id

    } for config in training_config.organization_configs]

    datacards = dataset_provider.get_datacards()

    saiva_api.trained_models.create_with_dict(ml_model_org_config_id=None, training_metadata=training_metadata, model_config=model_config, datacards=datacards)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(upload_model_metadata)
