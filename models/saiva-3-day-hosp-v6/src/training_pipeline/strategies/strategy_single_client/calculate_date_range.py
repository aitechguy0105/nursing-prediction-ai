import datetime
import logging
import typing

from dataclasses import asdict

from src.training_pipeline.shared.models import ClientConfiguration, ExperimentDates
from src.training_pipeline.strategies.common.calculate_date_range import CalculateDateRangeBase
from src.training_pipeline.shared.utils import get_datasource


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CalculateDateRangeSingleClient(CalculateDateRangeBase):
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

        client_configuration = client_configurations[0]

        datasource_id = client_configuration.datasource_id
        facility_ids = config.client_configuration.facilities

        datasource = get_datasource(
            config=config,
            datasource_id=datasource_id,
        )

        date_range = asdict(experiment_dates)

        if client_configuration.experiment_dates:
            if client_configuration.experiment_dates.test_end_date:
                date_range['test_end_date'] = client_configuration.experiment_dates.test_end_date
            if client_configuration.experiment_dates.train_start_date:
                date_range['train_start_date'] = client_configuration.experiment_dates.train_start_date

        if not date_range['test_end_date']:
            test_end_date = self.calculate_test_end_date(
                client=client_configuration.client,
                datasource=datasource,
                facility_ids=facility_ids,
            )

            log.info(f"Calculated test end date: {test_end_date}")

            date_range['test_end_date'] = test_end_date
        else:
            log.info(f"Using provided test end date: {date_range['test_end_date']}")

        if not date_range['train_start_date']:
            df = self.fetch_patient_census_data(
                dataset_provider=dataset_provider,
                client=client_configuration.client,
                datasource=datasource,
                facility_ids=facility_ids,
                test_end_date=test_end_date,
                experiment_dates_facility_wise_overrides=client_configuration.experiment_dates_facility_wise_overrides,
            )

            train_start_date = self.calculate_train_start_date(
                df=df,
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
