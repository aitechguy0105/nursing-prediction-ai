import logging
import typing

from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.strategies.common.preprocess_data import PreprocessDataBase


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PreprocessDataSingleClient(PreprocessDataBase):
    def execute(
        self,
        *,
        run_id: str,
        client_configurations: typing.List[ClientConfiguration],
        required_features_preprocess: typing.Optional[typing.List[str]] = None,
        force_regenerate: typing.Optional[bool] = False,
    ):
        dataset_provider, config = self.get_dataset_provider_and_config(run_id=run_id, force_regenerate=force_regenerate)

        client_configuration = client_configurations[0]

        client = client_configuration.client

        self.preprocess_data_for_client(
            dataset_provider=dataset_provider,
            config=config,
            client=client,
            required_features_preprocess=required_features_preprocess,
        )

        self.upload_config(
            dataset_provider=dataset_provider,
        )
