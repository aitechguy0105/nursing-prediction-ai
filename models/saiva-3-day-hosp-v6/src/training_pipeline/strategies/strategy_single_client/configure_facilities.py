import logging
import typing

from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.strategies.common.configure_facilities import ConfigureFacilitiesBase
from src.training_pipeline.shared.utils import get_datasource


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ConfigureFacilitiesSingleClient(ConfigureFacilitiesBase):
    def execute(
        self,
        *,
        run_id: str,
        client_configurations: typing.List[ClientConfiguration],
        force_regenerate: typing.Optional[bool] = False,
    ):
        dataset_provider, config = self.get_dataset_provider_and_config_or_none(run_id=run_id, force_regenerate=force_regenerate)

        if not dataset_provider:
            return

        client_configuration = client_configurations[0]

        datasource_id = client_configuration.datasource_id
        facility_ids = client_configuration.facility_ids

        datasource = get_datasource(
            config=config,
            datasource_id=datasource_id,
        )

        client_facility_ids = self.get_facility_ids(
            facility_ids=facility_ids,
            datasource=datasource,
        )

        client_configuration = {
            'facilities': client_facility_ids,
        }

        self.upload_config(
            client_configuration=client_configuration,
            dataset_provider=dataset_provider,
        )
