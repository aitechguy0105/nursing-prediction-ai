import logging
import sys
import typing

import fire
from eliot import to_file

from src.training_pipeline.strategies.strategy_multiple_clients.fetch_data import FetchDataMultipleClients
from src.training_pipeline.strategies.strategy_single_client.fetch_data import FetchDataSingleClient
from src.training_pipeline.strategies.strategy_large_plus_small_client.fetch_data import FetchDataLargePlusSmallClient
from src.training_pipeline.shared.enums import Strategy
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.shared.utils import convert_input_params_decorator, setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@convert_input_params_decorator
def fetch_data(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    force_regenerate: typing.Optional[bool] = False,
    invalid_action_types: typing.Optional[typing.List[str]] = None,
    strategy: typing.Optional[Strategy] = Strategy.SINGLE_CLIENT,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Fetch data for the training.

    :param run_id: the run id
    :param client_configurations: list of all client configurations
    :param force_regenerate: force the regeneration of the data
    :param invalid_action_types: the list of invalid action types
    :param strategy: the strategy to use (if not provided, the single client strategy is used - only the first client is used)
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    if strategy == Strategy.MULTIPLE_CLIENTS:
        fetch_data_obj = FetchDataMultipleClients()
    elif strategy == Strategy.SINGLE_CLIENT:
        fetch_data_obj = FetchDataSingleClient()
    elif strategy == Strategy.LARGE_PLUS_SMALL_CLIENT:
        fetch_data_obj = FetchDataLargePlusSmallClient()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    fetch_data_obj.execute(
        run_id=run_id,
        client_configurations=client_configurations,
        force_regenerate=force_regenerate,
        invalid_action_types=invalid_action_types,
    )


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(fetch_data)