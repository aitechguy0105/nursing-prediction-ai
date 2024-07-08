import logging
import sys
import time
import typing

from omegaconf import OmegaConf
import fire
import pandas as pd
from eliot import to_file

from src.saiva.model.shared.constants import saiva_api, LOCAL_TRAINING_CONFIG_PATH
from src.saiva.model.shared.patient_census import PatientCensus
from src.saiva.model.shared.notes import NoteFeatures
from src.saiva.model.shared.assessments import AssessmentFeatures
from src.saiva.model.shared.adt import AdtFeatures
from src.saiva.model.shared.risks import RiskFeatures
from src.saiva.model.shared.admissions import AdmissionFeatures
from src.saiva.model.shared.immunizations import ImmunizationFeatures
from src.saiva.model.shared.diagnosis import DiagnosisFeatures
from src.saiva.model.shared.rehosp import RehospFeatures
from src.saiva.model.shared.labs import LabFeatures
from src.saiva.model.shared.alerts import AlertFeatures
from src.saiva.model.shared.meds import MedFeatures
from src.saiva.model.shared.orders import OrderFeatures
from src.saiva.model.shared.vitals import VitalFeatures
from src.saiva.model.shared.demographics import DemographicFeatures
from src.saiva.model.shared.mds import MDSFeatures
from src.saiva.training import load_config
from src.training_pipeline.shared.models import ExperimentDates
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.shared.utils import convert_input_params_decorator, setup_sentry


to_file(sys.stdout)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.FEATURE_ENGINEERING


@convert_input_params_decorator
def feature_engineering(
    *,
    run_id: str,
    feature: str,
    client_configurations: typing.List[ClientConfiguration],
    force_regenerate: typing.Optional[bool] = False,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Calculate the date range for the training.

    :param run_id: the run id
    :param feature: the feature to be engineered
    :param client_configurations: list of all client configurations
    :param force_regenerate: whether to force regeneration of the dataset
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    # we set the client to the first client in the list (it doesn't matter which client we use)
    client = client_configurations[0].client

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP))
    
    config = load_config(LOCAL_TRAINING_CONFIG_PATH)

    training_config = config.training_config

    model_version = saiva_api.model_types.get_by_model_type_id(model_type_id=training_config.model_type, version=training_config.model_version)

    training_metadata = training_config.training_metadata
    training_metadata['model_type_version_id'] = model_version.id

    conf = OmegaConf.create({'training_config': {'training_metadata': training_metadata}})
    OmegaConf.save(conf, f'{LOCAL_TRAINING_CONFIG_PATH}generated/training_metadata.yaml')

    experiment_dates = ExperimentDates(**training_metadata.experiment_dates.dates_calculation)

    feature_groups_spec = {
        'patient_census': {
            'feature_group_class': PatientCensus,
            'params': {'train_start_date': experiment_dates.train_start_date, 'test_end_date': experiment_dates.test_end_date},
            'in_dataset_name': 'patient_census',
            'param_dataset_name': 'census_df',
            'out_dataset_name': 'patient_census',
        },
        'demographics': {
            'feature_group_class': DemographicFeatures,
            'in_dataset_name': 'patient_demographics',
            'param_dataset_name': 'demo_df',
            'out_dataset_name': 'demo_df',
        },
        'vitals': {
            'feature_group_class': VitalFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_vitals',
            'param_dataset_name': 'vitals',
            'out_dataset_name': 'vitals_df',
        },
        'orders': {
            'feature_group_class': OrderFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_orders',
            'param_dataset_name': 'orders',
            'out_dataset_name': 'orders_df',
        },
        'meds': {
            'feature_group_class': MedFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_meds',
            'param_dataset_name': 'meds',
            'out_dataset_name': 'meds_df',
        },
        'alerts': {
            'feature_group_class': AlertFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_alerts',
            'param_dataset_name': 'alerts',
            'out_dataset_name': 'alerts_df',
        },
        'lab_results': {
            'feature_group_class': LabFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_lab_results',
            'param_dataset_name': 'labs',
            'out_dataset_name': 'labs_df',
        },
        'patient_rehosps': {
            'feature_group_class': RehospFeatures,
            'params': {'config': config, 'train_start_date': experiment_dates.train_start_date},
            'in_dataset_name': 'patient_rehosps',
            'param_dataset_name': 'rehosps',
            'out_dataset_name': 'rehosp_df',
        },
        'patient_admissions': {
            'feature_group_class': AdmissionFeatures,
            'in_dataset_name': 'patient_admissions',
            'param_dataset_name': 'admissions',
            'out_dataset_name': 'admissions_df',
        },
        'patient_diagnosis': {
            'feature_group_class': DiagnosisFeatures,
            'params': {'diagnosis_lookup_ccs_s3_file_path': model_version.diagnosis_lookup_ccs_s3_uri, 'config': config},
            'in_dataset_name': 'patient_diagnosis',
            'param_dataset_name': 'diagnosis',
            'out_dataset_name': 'diagnosis_df',
        },
        'patient_progress_notes': {
            'feature_group_class': NoteFeatures,
            'params': {'client': client, 'vector_model': training_metadata.vector_model},
            'in_dataset_name': 'patient_progress_notes',
            'param_dataset_name': 'notes',
            'out_dataset_name': 'notes_df',
        },
        'patient_immunizations': {
            'feature_group_class': ImmunizationFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_immunizations',
            'param_dataset_name': 'immuns_df',
            'out_dataset_name': 'immuns_df',
        },
        'patient_risks': {
            'feature_group_class': RiskFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_risks',
            'param_dataset_name': 'risks_df',
            'out_dataset_name': 'risks_df',
        },
        'patient_assessments': {
            'feature_group_class': AssessmentFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_assessments',
            'param_dataset_name': 'assessments_df',
            'out_dataset_name': 'assessments_df',
        },
        'patient_adt': {
            'feature_group_class': AdtFeatures,
            'params': {'config': config},
            'in_dataset_name': 'patient_adt',
            'param_dataset_name': 'adt_df',
            'out_dataset_name': 'adt_df',
        },
        'patient_mds': {
            'feature_group_class': MDSFeatures, 
            'params': {'config': config},
            'in_dataset_name': 'patient_mds',
            'param_dataset_name': 'mds_df',
            'out_dataset_name': 'mds_df'
        }
    }

    feature_group_spec = feature_groups_spec[feature]

    if dataset_provider.does_file_exist(filename=feature_group_spec['out_dataset_name'], step=CURRENT_STEP):
        log.info(f"Dataset {feature} already exists. Skipping featurization step")
        return
    
    if not dataset_provider.does_file_exist(filename=feature_group_spec['in_dataset_name'], step=TrainingStep.MERGE_DATA, ignore_force_regenerate=True, file_format='parquet'):
        if feature == 'patient_census':
            raise Exception(f"Dataset patient_census does not exist. Stopping training pipeline.")
        else:
            log.info(f"Dataset {feature} does not exist. Skipping featurization step")
            return

    feature_group_kwargs = feature_group_spec.get('params', {})
    feature_group_class = feature_group_spec['feature_group_class']
    
    feature_group_kwargs[feature_group_spec['param_dataset_name']] = dataset_provider.get(dataset_name=feature_group_spec['in_dataset_name'], step=TrainingStep.MERGE_DATA)
    
    if feature != 'patient_census':
        feature_group_kwargs['census_df'] = dataset_provider.get(dataset_name='patient_census', step=CURRENT_STEP)
        feature_group_kwargs['training'] = True

    if feature in ['patient_rehosps', 'patient_mds']:
        if not dataset_provider.does_file_exist(filename='patient_adt', step=TrainingStep.MERGE_DATA, ignore_force_regenerate=True):
            log.info(f"Dataset patient_adt does not exist. Skipping featurization step for {feature}")

        feature_group_kwargs['adt_df'] = dataset_provider.get(dataset_name='patient_adt', step=TrainingStep.MERGE_DATA)
    
    start_time = time.time()

    feature_group = feature_group_class(**feature_group_kwargs)
    df = feature_group.generate_features()

    # Write to new parquet file
    if not isinstance(df, pd.DataFrame):
        df = df[0]

    dataset_provider.set(dataset_name=feature_group_spec['out_dataset_name'], step=CURRENT_STEP, df=df)

    duration = time.time() - start_time
    log.info(f'Wall time for {feature_group_class.__name__}: {time.strftime("%H:%M:%S", time.gmtime(duration))}')

    dataset_provider.store_config(step=CURRENT_STEP)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(feature_engineering)
