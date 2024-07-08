import logging
import os
import shutil
import typing

from omegaconf import OmegaConf

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.saiva.model.shared.database import DbEngine
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.strategies.common.base import BaseStrategy


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.CONFIGURE_FACILITIES


class ConfigureFacilitiesBase(BaseStrategy):
    def __init__(self) -> None:
        super().__init__(current_step=CURRENT_STEP)

    def get_dataset_provider_and_config_or_none(
        self,
        *,
        run_id: str,
        force_regenerate: typing.Optional[bool] = False,
    ) -> typing.Tuple[typing.Optional[DatasetProvider], typing.Optional[OmegaConf]]:
        
        dataset_provider, config = super().get_dataset_provider_and_config(run_id=run_id, force_regenerate=force_regenerate)

        if dataset_provider.does_file_exist(step=CURRENT_STEP, filename='conf/training/generated/facilities', file_format='yaml'):
            log.info("Facilities already configured. Skipping configure facilities step.")
            return None, None

        return dataset_provider, config

    def get_facility_ids(
        self,
        *,
        datasource: OmegaConf,
        facility_ids: typing.Optional[typing.Union[typing.List[int], str]] = None,
    ) -> typing.List[int]:

        engine = DbEngine()
        client_sql_engine = engine.get_sqldb_engine(
            db_name=datasource.source_database_name,
            credentials_secret_id=datasource.source_database_credentials_secret_id,
            query={"driver": "ODBC Driver 17 for SQL Server"}
        )

        query = "SELECT FacilityID FROM view_ods_facility WHERE"
        apply_filter = "LineOfBusiness = 'SNF' AND Deleted='N'"

        if facility_ids:
            if isinstance(facility_ids, str):
                query = facility_ids
                apply_filter = ""
                log.info(f"Using custom query to select facilities: {query}")
            else:
                log.info(f"Using facility ids: {facility_ids}")
                apply_filter = f"facilityid IN ({','.join(str(facility_id) for facility_id in facility_ids)})"
        else:
            log.info(f"No facility ids provided. Using all SNF facilities.")
            
        result = client_sql_engine.execute(f"{query} {apply_filter}").fetchall()

        result = [row[0] for row in result]

        log.info(f"Found {len(result)} SNF facilities: {result}")

        if facility_ids:
            assert len(result) == len(facility_ids), f"Number of SNF facilities found ({len(result)}) does not match number of facilities provided ({len(facility_ids)})"

        return result

    def upload_config(
        self,
        *,
        client_configuration: typing.Dict[str, typing.Any],
        dataset_provider: DatasetProvider
    ):
        conf = OmegaConf.create({'client_configuration': client_configuration})

        OmegaConf.save(conf, f'{LOCAL_TRAINING_CONFIG_PATH}/generated/facilities.yaml')

        super().upload_config(dataset_provider=dataset_provider)