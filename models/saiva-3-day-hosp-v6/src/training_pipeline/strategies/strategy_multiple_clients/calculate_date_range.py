from dataclasses import asdict
import datetime
import logging
import typing

from omegaconf import OmegaConf
import pandas as pd

from src.training_pipeline.shared.helpers import DatasetProvider
from src.training_pipeline.shared.models import ClientConfiguration, ExperimentDates
from src.training_pipeline.strategies.common.calculate_date_range import CalculateDateRangeBase
from src.training_pipeline.shared.utils import get_datasource


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CalculateDateRangeMultipleClients(CalculateDateRangeBase):
    def calculate_train_start_date(
        self,
        *,
        dataset_provider: DatasetProvider,
        config: OmegaConf,
        client_configurations: typing.List[ClientConfiguration],
        test_end_date: datetime.date,
    ) -> datetime.date:

        all_clients_data = []

        for client_configuration in client_configurations:
            client = client_configuration.client
            datasource_id = client_configuration.datasource_id
            facility_ids = config.client_configuration.facilities[client]

            client_test_end_date = client_configuration.experiment_dates.test_end_date if client_configuration.experiment_dates and client_configuration.experiment_dates.test_end_date else test_end_date

            datasource = get_datasource(
                config=config,
                datasource_id=datasource_id,
            )

            df = self.fetch_patient_census_data(
                dataset_provider=dataset_provider,
                client=client,
                datasource=datasource,
                facility_ids=facility_ids,
                test_end_date=client_test_end_date,
                experiment_dates_facility_wise_overrides=client_configuration.experiment_dates_facility_wise_overrides,
            )

            all_clients_data.append(df)

        df = pd.concat(all_clients_data, ignore_index=True)

        train_start_date = super().calculate_train_start_date(
            df=df,
            test_end_date=test_end_date,
        )

        return train_start_date

    def calculate_test_end_date(
        self,
        *,
        config: OmegaConf,
        client_configurations: typing.List[ClientConfiguration],
    ) -> datetime.date:

        test_end_dates = []

        for client_configuration in client_configurations:
            client = client_configuration.client

            if client_configuration.experiment_dates and client_configuration.experiment_dates.test_end_date:
                log.info(f"Test end date was provided for client {client_configuration.client}. Skipping calculate test end date step.")
                test_end_date = client_configuration.experiment_dates.test_end_date
            else:
                datasource_id = client_configuration.datasource_id
                facility_ids = config.client_configuration.facilities[client]

                datasource = get_datasource(
                    config=config,
                    datasource_id=datasource_id,
                )
                test_end_date = super().calculate_test_end_date(
                    client=client,
                    datasource=datasource,
                    facility_ids=facility_ids,
                )

            test_end_dates.append(test_end_date)

        return max(test_end_dates)
    
    def execute(
        self,
        *,
        run_id: str,
        experiment_dates: ExperimentDates,
        client_configurations: typing.List[ClientConfiguration],
        force_regenerate: typing.Optional[bool] = False,
    ):

        dataset_provider, config = self.get_dataset_provider_and_config_or_none(run_id=run_id, force_regenerate=force_regenerate)

        if not dataset_provider:
            return

        date_range = asdict(experiment_dates)

        if not date_range['test_end_date']:
            test_end_date = self.calculate_test_end_date(
                client_configurations=client_configurations,
                config=config,
            )

            log.info(f"Calculated test end date: {test_end_date}")

            date_range['test_end_date'] = test_end_date
        else:
            log.info(f"Using provided test end date: {date_range['test_end_date']}")

        if not date_range['train_start_date']:

            train_start_date = self.calculate_train_start_date(
                dataset_provider=dataset_provider,
                config=config,
                client_configurations=client_configurations,
                test_end_date=test_end_date,
            )

            log.info(f"Calculated train start date: {train_start_date}")

            date_range['train_start_date'] = train_start_date
        else:
            log.info(f"Using provided train start date: {date_range['train_start_date']}")

        self.upload_config(
            date_range=date_range,
            dataset_provider=dataset_provider,
            training_config=config.training_config,
        )
