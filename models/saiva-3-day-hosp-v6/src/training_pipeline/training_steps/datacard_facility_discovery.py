import logging
import os
import typing

import fire

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.saiva.training.utils import load_config
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.shared.utils import convert_input_params_decorator, get_date_range, setup_sentry


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.DATACARD_FACILITY_DISCOVERY


@convert_input_params_decorator
def datacard_facility_discovery(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    force_regenerate: typing.Optional[bool] = False,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Generate the parameters for the datacard facility discovery.

    :param run_id: the run id
    :param client_configurations: list of all client configurations
    :param force_regenerate: force the regeneration of the data
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP))

    config = load_config(LOCAL_TRAINING_CONFIG_PATH)

    outfile_path = dataset_provider.make_datacard_s3_path()

    s3_path = dataset_provider.make_dataset_filepath(
        step=TrainingStep.PREPROCESS_DATA,
        dataset_name='',
    )

    all_datacard_params = {}

    for client_configuration in client_configurations:
        client = client_configuration.client

        if dataset_provider.does_datacard_exist(step=CURRENT_STEP, client=client):
            log.info(f"Datacard already generated for client {client}. Skipping {CURRENT_STEP.value} step.")
            continue

        date_range = get_date_range(
            config=config,
            client_configuration=client_configuration,
        )

        command_parts = [
            f"--client {client}",
            f"--datasource_id {client_configuration.datasource_id}",
            f"--start-date {date_range['train_start_date']}",
            f"--end-date {date_range['test_end_date']}",
            f"--outfile {outfile_path}",
            f"--s3-folder-path {s3_path}",
        ]
        datacard_params = " ".join(command_parts)
        all_datacard_params[client] = datacard_params

    if len(all_datacard_params) == 0:
        return

    if not os.path.exists('/data'):
        os.makedirs('/data')

    with open("/data/datacard_facility_discovery_params.txt", "w") as f:
        for datacard_params in all_datacard_params.values():
            f.write(f"{datacard_params}\n")

    print(all_datacard_params)

    dataset_provider.store_config(step=CURRENT_STEP)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(datacard_facility_discovery)
