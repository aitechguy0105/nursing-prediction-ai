import logging
import typing

import pandas as pd
from omegaconf import OmegaConf

from src.training_pipeline.shared.models import JoinMasterPatientLookupConfig
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.strategies.common.base import BaseStrategy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.PREPROCESS_DATA


class PreprocessDataBase(BaseStrategy):
    def __init__(self) -> None:
        super().__init__(current_step=CURRENT_STEP)

    def preprocess_data_for_client(
        self,
        *,
        config: OmegaConf,
        dataset_provider: DatasetProvider,
        client: str,
        required_features_preprocess: typing.Optional[typing.List[str]] = None,
    ):

        join_config = {
            'patient_demographics': JoinMasterPatientLookupConfig(
                merge_on=[]
            ),
            'patient_census': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"]
            ),
            'patient_rehosps': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_admissions': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_diagnosis': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_vitals': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_meds': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_orders': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_alerts': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_immunizations': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_risks': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_assessments': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_adt': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_progress_notes': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
            'patient_lab_results': JoinMasterPatientLookupConfig(
                merge_on=["patientid", "facilityid"],
                column_subset=['facilityid', 'patientid', 'masterpatientid']
            ),
        }

        if required_features_preprocess is not None:
            log.info("List of required features was provided")

        required_features_preprocess = required_features_preprocess or config.automatic_training.required_features_preprocess

        log.info(f"Required features are: {required_features_preprocess}")

        for feature_name in required_features_preprocess:
            join_config[feature_name].required = True

        master_patient_lookup = dataset_provider.get(dataset_name=f'{client}_master_patient_lookup', step=TrainingStep.FETCH_DATA)
        
        dataset_provider.set(dataset_name=f'{client}_master_patient_lookup', step=TrainingStep.PREPROCESS_DATA, df=master_patient_lookup)

        for feature_group, dataset_join_config in join_config.items():
            dataset_name = f"{client}_{feature_group}"

            if dataset_provider.does_file_exist(filename=dataset_name, step=TrainingStep.PREPROCESS_DATA):
                log.info(f"Dataset {dataset_name} already exists. Skipping preprocessing step")
                continue

            if not dataset_provider.does_file_exist(
                filename=dataset_name,
                step=TrainingStep.FETCH_DATA,
                ignore_force_regenerate=True
            ):
                if not dataset_join_config.required:
                    log.info(f"Skipping join for dataset {dataset_name} as it does not exist and is optional")
                    continue
                else:
                    raise Exception(f"Dataset {dataset_name} does not exist and is not optional")

            df = dataset_provider.get(dataset_name=dataset_name, step=TrainingStep.FETCH_DATA)

            if df.empty:
                if not dataset_join_config.required:
                    log.info(f"Skipping join for dataset {dataset_name} as it does not exist and is optional")
                    continue
                else:
                    raise Exception(f"Dataset {dataset_name} does not exist and is not optional")
                
            log.info(f"Joining dataset {dataset_name} with master_patient_lookup")

            if "patientid" in df.columns:
                df.dropna(subset=["patientid"], axis=0, inplace=True)

            if feature_group == "patient_demographics":
                df['dateofbirth'] = pd.to_datetime(df['dateofbirth']).astype('datetime64[ms]')
            elif dataset_join_config.column_subset:
                df = df.merge(master_patient_lookup[dataset_join_config.column_subset], on=dataset_join_config.merge_on)
            else:
                df = df.merge(master_patient_lookup, on=dataset_join_config.merge_on)

            dataset_provider.set(dataset_name=dataset_name, step=TrainingStep.PREPROCESS_DATA, df=df)
            del (df)

        del (master_patient_lookup)

        for feature_group in join_config.keys():
            dataset_name = f"{client}_{feature_group}"

            if dataset_provider.does_file_exist(
                filename=dataset_name,
                step=TrainingStep.PREPROCESS_DATA,
                ignore_force_regenerate=True
            ):
                df = dataset_provider.get(dataset_name=dataset_name, step=TrainingStep.PREPROCESS_DATA)
                if df is None:
                    log.error(f"Dataset {dataset_name} is missing")
                    continue

                log.info(f"Dataset {dataset_name}, shape: {df.shape}")
