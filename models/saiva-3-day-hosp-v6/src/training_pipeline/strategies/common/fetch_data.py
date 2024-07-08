import datetime
import logging
import typing

import pandas as pd
from eliot import start_action
from omegaconf import OmegaConf

from src.saiva.model.shared.database import DbEngine
from src.saiva.model.shared.load_raw_data import postprocess_data, unroll_patient_census, validate_dataset
from src.saiva.model.shared.utils import get_client_class
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.strategies.common.base import BaseStrategy
from src.training_pipeline.shared.utils import get_datasource
from src.training_pipeline.shared.models import ExperimentDates


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.FETCH_DATA


class FetchDataBase(BaseStrategy):
    def __init__(self) -> None:
        super().__init__(current_step=CURRENT_STEP)

    def fetch_data_for_client(
        self,
        *,
        dataset_provider: DatasetProvider,
        client: str,
        config: OmegaConf,
        datasource_id: str,
        facility_ids: typing.List[int],
        date_range: typing.Dict[str, datetime.date],
        invalid_action_types: typing.Optional[typing.List[str]] = None,
        experiment_dates_facility_wise_overrides: typing.Optional[typing.Dict[int, ExperimentDates]] = None,
    ):

        datasource = get_datasource(
            config=config,
            datasource_id=datasource_id,
        )

        # Connect to DB and fetch data
        engine = DbEngine()
        client_sql_engine = engine.get_sqldb_engine(
            db_name=datasource.source_database_name,
            credentials_secret_id=datasource.source_database_credentials_secret_id,
            query={"driver": "ODBC Driver 17 for SQL Server"}
        )

        # verify connectivity
        engine.verify_connectivity(client_sql_engine)

        ### ======================== Fetch Data ============================
        client_obj = get_client_class(client)(
            facilities=facility_ids,
            engine=client_sql_engine,
        )
        queries = getattr(client_obj, 'get_training_queries')(
            train_start_date=date_range['train_start_date'],
            test_end_date=date_range['test_end_date'],
        )

        # fetch data for training and dump it to disk
        for feature_group, query in queries.items():
            dataset_name = f"{client}_{feature_group}"

            if dataset_provider.does_file_exist(filename=dataset_name, step=CURRENT_STEP):
                log.info(f"Dataset {dataset_name} already exists. Skipping fetch data step")
                continue

            with start_action(action_type=f'pull_{dataset_name}'):
                # Execute the query using pandas
                log.info(f"Fetching data for dataset {dataset_name}")
                df = pd.read_sql(query, con=client_sql_engine)

                log.info(f"Total records fetched for dataset {dataset_name} is {len(df)}")

                validate_dataset(feature_group, df)
                df = postprocess_data(df, dataset_name)
                if feature_group == 'patient_census':
                    df = unroll_patient_census(
                        census=df,
                        first_date=date_range['train_start_date'],
                        last_date=date_range['test_end_date'],
                        training=True,
                        invalid_action_types=invalid_action_types,
                        experiment_dates_facility_wise_overrides=experiment_dates_facility_wise_overrides,
                    )

                # TODO: should do client transformation here, but no client has it implemented yet, so skipping for now

                dataset_provider.set(dataset_name=dataset_name, step=CURRENT_STEP, df=df)
