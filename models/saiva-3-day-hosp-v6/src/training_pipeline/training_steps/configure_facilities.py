import logging
import sys
import typing

import fire
from eliot import to_file

from src.training_pipeline.shared.helpers import TrainingStep
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.shared.enums import Strategy
from src.training_pipeline.strategies.strategy_multiple_clients.configure_facilities import ConfigureFacilitiesMultipleClients
from src.training_pipeline.strategies.strategy_single_client.configure_facilities import ConfigureFacilitiesSingleClient
from src.training_pipeline.strategies.strategy_large_plus_small_client.configure_facilities import ConfigureFacilitiesLargePlusSmallClient
from src.training_pipeline.shared.utils import convert_input_params_decorator, setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.CONFIGURE_FACILITIES


@convert_input_params_decorator
def configure_facilities(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    force_regenerate: typing.Optional[bool] = False,
    strategy: typing.Optional[Strategy] = Strategy.SINGLE_CLIENT,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Configure the facilities for the training based on provided facility ids.

    :param run_id: the run id
    :param client_configurations: list of all client configurations
    :param force_regenerate: force regeneration of the data
    :param strategy: the strategy to use (if not provided, the single client strategy is used - only the first client is used)
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    if strategy == Strategy.MULTIPLE_CLIENTS:
        configure_facilities_obj = ConfigureFacilitiesMultipleClients()
    elif strategy == Strategy.SINGLE_CLIENT:
        configure_facilities_obj = ConfigureFacilitiesSingleClient()
    elif strategy == Strategy.LARGE_PLUS_SMALL_CLIENT:
        configure_facilities_obj = ConfigureFacilitiesLargePlusSmallClient()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    configure_facilities_obj.execute(
        run_id=run_id,
        client_configurations=client_configurations,
        force_regenerate=force_regenerate,
    )


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(configure_facilities)
