import logging
import os
import typing

import fire

from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.utils import setup_sentry


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.DATACARD_SHAP_VALUES


def datacard_shap_values(
    *,
    run_id: str,
    model_type: typing.Optional[str] = 'MODEL_UPT',
    force_regenerate: typing.Optional[bool] = False,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Generate the parameters for the datacard shap values.

    :param run_id: the run id
    :param model_type: the model type
    :param force_regenerate: force the regeneration of the data
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    model_type = model_type.lower()

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP), prefix=f'/{model_type}')

    if dataset_provider.does_datacard_exist(step=CURRENT_STEP):
        log.info(f"Datacard already generated. Skipping {CURRENT_STEP.value} step.")
        return

    outfile_path = dataset_provider.make_datacard_s3_path()

    s3_path = dataset_provider.make_dataset_filepath(
        step=TrainingStep.TRAIN_MODEL,
        dataset_name=model_type,
    )

    s3_path_processed = dataset_provider.make_dataset_filepath(
        step=TrainingStep.DATASETS_GENERATION,
        dataset_name=f'{model_type}/',
    )

    command_parts = [
        f"--s3-folder-path {s3_path}",
        f"--outfile {outfile_path}",
        f"--s3-folder-path-processed {s3_path_processed}",
        f"--model-type {model_type}",
    ]
    datacard_params = " ".join(command_parts)

    if not os.path.exists('/data'):
        os.makedirs('/data')

    with open("/data/datacard_shap_values_params.txt", "w") as f:
        f.write(datacard_params)

    print("params: " + datacard_params)

    dataset_provider.store_config(step=CURRENT_STEP, prefix=f'/{model_type}')


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(datacard_shap_values)
