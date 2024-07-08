import logging
import sys
import typing

import fire
from eliot import to_file

from src.training_pipeline.strategies.strategy_multiple_clients.preprocess_data import PreprocessDataMultipleClients
from src.training_pipeline.strategies.strategy_single_client.preprocess_data import PreprocessDataSingleClient
from src.training_pipeline.strategies.strategy_large_plus_small_client.preprocess_data import PreprocessDataLargePlusSmallClient
from src.training_pipeline.shared.enums import Strategy
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.shared.utils import convert_input_params_decorator, setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@convert_input_params_decorator
def preprocess_data(
    *,
    run_id: str,
    client_configurations: typing.List[ClientConfiguration],
    required_features_preprocess: typing.Optional[typing.List[str]] = None,
    force_regenerate: typing.Optional[bool] = False,
    strategy: typing.Optional[Strategy] = Strategy.SINGLE_CLIENT,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Preprocess the data.

        :param run_id: the run id
        :param client_configurations: list of all client configurations
        :param required_features_preprocess: list of required features
        :param force_regenerate: force the regeneration of the data
        :param strategy: the strategy to use (if not provided, the single client strategy is used - only the first client is used)
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    if strategy == Strategy.MULTIPLE_CLIENTS:
        preprocess_data_obj = PreprocessDataMultipleClients()
    elif strategy == Strategy.SINGLE_CLIENT:
        preprocess_data_obj = PreprocessDataSingleClient()
    elif strategy == Strategy.LARGE_PLUS_SMALL_CLIENT:
        preprocess_data_obj = PreprocessDataLargePlusSmallClient()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    preprocess_data_obj.execute(
        run_id=run_id,
        client_configurations=client_configurations,
        force_regenerate=force_regenerate,
        required_features_preprocess=required_features_preprocess,
    )


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(preprocess_data)
