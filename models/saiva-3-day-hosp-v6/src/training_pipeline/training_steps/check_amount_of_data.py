import logging
import sys
import typing

import fire
from eliot import to_file

from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.constants import AVERAGE_CENSUS, MAX_CENSUS, MIN_CENSUS
from src.training_pipeline.shared.utils import setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.CHECK_AMOUNT_OF_DATA


def check_amount_of_data(
    *,
    run_id: str,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Check the amount of data available for training.

    :param run_id: the run id
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=False)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP))

    df = dataset_provider.get(dataset_name='patient_census', step=TrainingStep.MERGE_DATA)

    num_rows = len(df)

    dataset_provider.store_config(step=CURRENT_STEP)

    if num_rows < MIN_CENSUS:
        raise Exception(f"Insufficient data. Only {num_rows} rows available. Minimum required is {MIN_CENSUS}.")
    elif num_rows < AVERAGE_CENSUS:
        log.warning(f"Insufficient data. Only {num_rows} rows available. Average is {AVERAGE_CENSUS}.")
    elif num_rows > MAX_CENSUS:
        log.warning(f"Too much data. {num_rows} rows available. Maximum allowed is {MAX_CENSUS}.")
    else:
        log.info(f"Data amount is sufficient. {num_rows} rows available.")



if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(check_amount_of_data)
