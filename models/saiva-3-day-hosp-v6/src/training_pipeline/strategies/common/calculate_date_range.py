import datetime
import logging
import typing

from omegaconf import OmegaConf
import pandas as pd

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.strategies.common.base import BaseStrategy
from src.training_pipeline.shared.helpers import DatasetProvider
from src.saiva.model.shared.database import DbEngine
from src.saiva.model.shared.load_raw_data import postprocess_data, unroll_patient_census, validate_dataset
from src.training_pipeline.shared.constants import AVERAGE_CENSUS
from src.saiva.model.shared.utils import get_client_class
from src.training_pipeline.shared.models import ExperimentDates
from src.training_pipeline.shared.utils import convert_dates_to_str


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.CALCULATE_DATE_RANGE


class CalculateDateRangeBase(BaseStrategy):
    def __init__(self, *, invalid_action_types: typing.Optional[typing.List[str]] = None) -> None:
        self.invalid_action_types = invalid_action_types
        super().__init__(current_step=CURRENT_STEP)

    def get_dataset_provider_and_config_or_none(
        self,
        *,
        run_id: str,
        force_regenerate: typing.Optional[bool] = False,
    ) -> typing.Tuple[typing.Optional[DatasetProvider], typing.Optional[OmegaConf]]:
        dataset_provider, config = self.get_dataset_provider_and_config(
            run_id=run_id,
            force_regenerate=force_regenerate,
        )

        if not force_regenerate and config.training_config.training_metadata.experiment_dates.get('calculate_date_range', None):
            log.info("Experiment dates already calculated. Skipping experiment dates calculation step")
            return None, None

        return dataset_provider, config

    def calculate_test_end_date(
        self,
        *,
        client: str,
        datasource: OmegaConf,
        facility_ids: typing.List[int],
    ) -> datetime.date:

        engine = DbEngine()
        client_sql_engine = engine.get_sqldb_engine(
            db_name=datasource.source_database_name,
            credentials_secret_id=datasource.source_database_credentials_secret_id,
            query={"driver": "ODBC Driver 17 for SQL Server"}
        )
            
        result = client_sql_engine.execute(
            f"""
                WITH unique_census AS (
                    SELECT
                        DISTINCT clientid, facilityid, CAST(censusdate AS date) AS censusdate
                    FROM
                        view_ods_daily_census_v2
                    WHERE 
                        censusdate <= CURRENT_TIMESTAMP
                    AND
                        facilityid IN ({",".join(str(facility_id) for facility_id in facility_ids)})
                )
                SELECT 
                    DATEADD(day, -14, MAX(censusdate)) as max_censusdate
                FROM unique_census;
            """
        ).fetchone()

        test_end_date = result[0]

        log.info(f"Test end date for client {client} is {test_end_date}")

        return test_end_date

    def fetch_patient_census_data(
        self,
        *,
        dataset_provider: DatasetProvider,
        client: str,
        datasource: OmegaConf,
        facility_ids: typing.List[int],
        test_end_date: datetime.date,
        experiment_dates_facility_wise_overrides: typing.Optional[typing.Dict[int, ExperimentDates]] = None,
    ) -> pd.DataFrame:

        test_end_date_str = datetime.datetime.strftime(test_end_date, '%Y-%m-%d')

        date_range = {
            "test_end_date": test_end_date_str,
            "train_start_date": '2016-01-01'
        }

        feature_group = 'patient_census'
        dataset_name = f"{client}_{feature_group}"

        if dataset_provider.does_file_exist(filename=dataset_name, step=self.CURRENT_STEP):
            log.info(f"Dataset {dataset_name} already exists. Skipping fetch data step")
            return dataset_provider.get(dataset_name=dataset_name, step=self.CURRENT_STEP)

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
        clientClass = get_client_class(client)
        queries = getattr(clientClass(facilities=facility_ids), 'get_training_queries')(
            train_start_date=date_range['train_start_date'],
            test_end_date=date_range['test_end_date']
        )
        
        query = queries[feature_group]

        log.info(f"Fetching data for dataset patient_census to calculate train start date")
        df = pd.read_sql(query, con=client_sql_engine)

        log.info(f"Total records fetched for dataset {dataset_name} is {len(df)}")

        validate_dataset(feature_group, df)
        df = postprocess_data(df, dataset_name)

        df = unroll_patient_census(
            census=df,
            first_date=date_range['train_start_date'],
            last_date=date_range['test_end_date'],
            training=True,
            invalid_action_types=self.invalid_action_types,
            experiment_dates_facility_wise_overrides=experiment_dates_facility_wise_overrides,
        )
        df['client'] = client
        dataset_provider.set(dataset_name=dataset_name, step=self.CURRENT_STEP, df=df)

        return df

    def calculate_train_start_date(
        self,
        *,
        df: pd.DataFrame,
        test_end_date: datetime.date,
    ) -> datetime.date:

        if len(df) == 0:
            raise Exception("No data found for patient_census. Unable to calculate train start date.")
        
        test_end_date_str = datetime.datetime.strftime(test_end_date, '%Y-%m-%d')

        minimum_date = (test_end_date - datetime.timedelta(days=516))
        minimum_date = datetime.datetime.strftime(minimum_date, '%Y-%m-%d')

        df.sort_values(['censusdate'], inplace=True, ascending=False)

        df = df[df['censusdate'] <= pd.to_datetime(test_end_date_str)]
        
        idx = -1 if AVERAGE_CENSUS > len(df) else AVERAGE_CENSUS 
        train_start_date = min(df.iloc[idx]['censusdate'], pd.to_datetime(minimum_date)).date()

        return train_start_date

    def upload_config(
        self,
        *,
        date_range: typing.Dict[str, datetime.date],
        dataset_provider: DatasetProvider,
        training_config: OmegaConf,
    ):
        training_metadata = training_config.training_metadata

        assert date_range['train_start_date'] and date_range['test_end_date'], "Not enough data to calculate date range."

        if pd.to_datetime(date_range['train_start_date']) <= pd.to_datetime('2016-01-01'):
            log.warning(f"Train start date {date_range['train_start_date']} is too early. Consult with ML team before you proceed.")

        log.info(f"Date range: {date_range['train_start_date']} - {date_range['test_end_date']}")

        # Convert to string to avoid yaml serialization error
        training_metadata.experiment_dates.calculate_date_range = convert_dates_to_str(experiment_dates=date_range)

        conf = OmegaConf.create({'training_config': {'training_metadata': training_metadata}})
        OmegaConf.save(conf, f'{LOCAL_TRAINING_CONFIG_PATH}generated/training_metadata.yaml')

        super().upload_config(dataset_provider=dataset_provider)
