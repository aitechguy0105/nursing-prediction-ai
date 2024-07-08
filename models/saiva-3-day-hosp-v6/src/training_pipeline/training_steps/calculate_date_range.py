import logging
import sys
import typing

import fire
from eliot import to_file

from src.training_pipeline.strategies.strategy_multiple_clients.calculate_date_range import CalculateDateRangeMultipleClients
from src.training_pipeline.strategies.strategy_single_client.calculate_date_range import CalculateDateRangeSingleClient
from src.training_pipeline.strategies.strategy_large_plus_small_client.calculate_date_range import CalculateDateRangeLargePlusSmallClient
from src.training_pipeline.shared.enums import Strategy
from src.training_pipeline.shared.models import ClientConfiguration, ExperimentDates
from src.training_pipeline.shared.utils import convert_input_params_decorator, setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@convert_input_params_decorator
def calculate_date_range(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    experiment_dates: ExperimentDates,
    force_regenerate: typing.Optional[bool] = False,
    invalid_action_types: typing.Optional[typing.List[str]] = None,
    strategy: typing.Optional[Strategy] = Strategy.SINGLE_CLIENT,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Calculate the date range for the training.

    :param run_id: the run id
    :param client_configurations: list of all client configurations
    :param experiment_dates: dictionary of experiment dates
    :param force_regenerate: force the regeneration of the data
    :param invalid_action_types: the list of invalid action types
    :param strategy: the strategy to use (if not provided, the single client strategy is used - only the first client is used)
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    if strategy == Strategy.MULTIPLE_CLIENTS:
        calculate_date_range_obj = CalculateDateRangeMultipleClients(invalid_action_types=invalid_action_types)
    elif strategy == Strategy.SINGLE_CLIENT:
        calculate_date_range_obj = CalculateDateRangeSingleClient(invalid_action_types=invalid_action_types)
    elif strategy == Strategy.LARGE_PLUS_SMALL_CLIENT:
        calculate_date_range_obj = CalculateDateRangeLargePlusSmallClient()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    calculate_date_range_obj.execute(
        run_id=run_id,
        client_configurations=client_configurations,
        experiment_dates=experiment_dates,
        force_regenerate=force_regenerate,
    )


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(calculate_date_range)
