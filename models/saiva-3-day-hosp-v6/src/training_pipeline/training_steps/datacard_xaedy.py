import logging
import os
import typing

import fire

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.saiva.training.utils import load_config
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.shared.utils import convert_input_params_decorator, setup_sentry


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.DATACARD_X_AED_Y_CHECK


@convert_input_params_decorator
def datacard_x_aed_y_check(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    model_type: typing.Optional[str] = 'MODEL_UPT',
    force_regenerate: typing.Optional[bool] = False,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Generate the parameters for the datacard prediction probability.

    :param run_id: the run id
    :param client_configurations: list of all client configurations
    :param model_type: the model type
    :param force_regenerate: force the regeneration of the data
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    model_type = model_type.lower()

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP), prefix=f'/{model_type}')

    # Copy over feature_names.pickle from feature_selection step to dataset_generation step
    feature_names = dataset_provider.load_json(
        step=TrainingStep.FEATURE_SELECTION,
        filename=f'{model_type}/feature_drop_stats',
    )

    dataset_provider.store_json(
        step=TrainingStep.DATASETS_GENERATION,
        filename=f'{model_type}/feature_drop_stats',
        data=feature_names,
    )

    outfile_path = dataset_provider.make_datacard_s3_path()
    
    s3_path = dataset_provider.make_dataset_filepath(
        step=TrainingStep.PREPROCESS_DATA,
        dataset_name='',
    )

    s3_path_processed = dataset_provider.make_dataset_filepath(
        step=TrainingStep.DATASETS_GENERATION,
        dataset_name=f'{model_type}/',
    )

    all_datacard_params = {}

    for client_configuration in client_configurations:
        client = client_configuration.client

        if dataset_provider.does_datacard_exist(step=CURRENT_STEP, client=client):
            log.info(f"Datacard already generated for client {client}. Skipping {CURRENT_STEP.value} step.")
            continue

        command_parts = [
            f"--client {client}",
            f"--datasource_id {client_configuration.datasource_id}",
            f"--s3-folder-path {s3_path}",
            f"--outfile {outfile_path}",
            f"--s3-folder-path-processed {s3_path_processed}",
            f"--model-type {model_type}",
        ]
        datacard_params = " ".join(command_parts)
        all_datacard_params[client] = datacard_params

    if len(all_datacard_params) == 0:
        return

    if not os.path.exists('/data'):
        os.makedirs('/data')

    with open("/data/datacard_xaedy_params.txt", "w") as f:
        for datacard_params in all_datacard_params.values():
            f.write(f"{datacard_params}\n")

    print(all_datacard_params)

    dataset_provider.store_config(step=CURRENT_STEP, prefix=f'/{model_type}')


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(datacard_x_aed_y_check)
