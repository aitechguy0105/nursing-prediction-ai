import logging
import typing

from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.strategies.common.fetch_data import FetchDataBase
from src.training_pipeline.shared.utils import get_date_range

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FetchDataMultipleClients(FetchDataBase):
    def execute(
        self,
        *,
        run_id: str,
        client_configurations: typing.List[ClientConfiguration],
        force_regenerate: typing.Optional[bool] = False,
        invalid_action_types: typing.Optional[typing.List[str]] = None,
    ):
    
        dataset_provider, config = self.get_dataset_provider_and_config(run_id=run_id, force_regenerate=force_regenerate)

        for client_configuration in client_configurations:
            client = client_configuration.client
            datasource_id = client_configuration.datasource_id
            facility_ids = config.client_configuration.facilities[client]

            date_range = get_date_range(
                config=config,
                client_configuration=client_configuration,
            )

            self.fetch_data_for_client(
                dataset_provider=dataset_provider,
                client=client,
                datasource_id=datasource_id,
                date_range=date_range,
                invalid_action_types=invalid_action_types,
                facility_ids=facility_ids,
                config=config,
                experiment_dates_facility_wise_overrides=client_configuration.experiment_dates_facility_wise_overrides,
            )

        self.upload_config(
            dataset_provider=dataset_provider,
        )
