import logging
import sys
import typing

import fire
from eliot import to_file

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.saiva.model.shared.utils import get_memory_usage
from src.saiva.training.utils import load_config
from src.training_pipeline.shared.models import JoinFeaturesConfig
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.shared.utils import setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.POST_FEATURE_ENGINEERING_MERGE


def post_feature_engineering_merge(
    *,
    run_id: str,
    required_features: typing.List[str] = [],
    force_regenerate: typing.Optional[bool] = False,
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Merge data after feature engineering.

        :param run_id: the run id
        :param required_features: list of required features
        :param force_regenerate: force the regeneration of the data
    """

    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP))

    config = load_config(LOCAL_TRAINING_CONFIG_PATH)

    if dataset_provider.does_file_exist(filename='final_df', step=CURRENT_STEP):
        log.info("Dataset final_df already exists. Skipping step")
        return

    join_features_config = {
        'demo_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Demographics'
        ),
        'vitals_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Vitals'
        ),
        'orders_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Orders'
        ),
        'alerts_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Alerts'
        ),
        'meds_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Medications'
        ),
        'rehosp_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Transfers'
        ),
        'admissions_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Admissions'
        ),
        'diagnosis_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Diagnoses'
        ),
        'labs_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Labs'
        ),
        'notes_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='ProgressNotes'
        ),
        'immuns_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Immunizations'
        ),
        'risks_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Risks'
        ),
        'assessments_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Assessments'
        ),
        'adt_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Adt'
        ),
        'mds_df': JoinFeaturesConfig(
            merge_on=['masterpatientid', 'facilityid', 'censusdate'],
            feature_group='Mds'
        )
    }

    if len(required_features):
        log.info(f"List of required features was provided")
        required_features = required_features or config.automatic_training.required_features_post_feature_engineering_merge

    for feature_name in required_features:
        join_features_config[feature_name].required = True

    exclude_columns = ['masterpatientid', 'facilityid', 'censusdate', 'client', 'date_of_transfer', 'na_indictator_date_of_transfer']
    feature_groups = {}

    final_df = dataset_provider.get(dataset_name='demo_df', step=TrainingStep.FEATURE_ENGINEERING)

    for feature_name, feature_join_config in join_features_config.items():
        if dataset_provider.does_file_exist(filename=feature_name, step=CURRENT_STEP):
            log.info(f"Dataset {feature_name} already exists. Skipping step")
            final_df = dataset_provider.get(dataset_name=feature_name, step=CURRENT_STEP)
            continue

        if feature_name == 'demo_df':
            df = final_df
        else:
            if not dataset_provider.does_file_exist(
                filename=feature_name,
                step=TrainingStep.FEATURE_ENGINEERING,
                ignore_force_regenerate=True
            ):
                if not feature_join_config.required:
                    log.info(f"Skipping join for dataset {feature_name} as it does not exist and is optional")
                    continue
                else:
                    raise Exception(f"Dataset {feature_name} does not exist and is not optional")

            log.info(f"Joining dataset {feature_name} to final_df")
            df = dataset_provider.get(dataset_name=feature_name, step=TrainingStep.FEATURE_ENGINEERING)

            if df.empty:
                if not feature_join_config.required:
                    log.info(f"Skipping join for dataset {feature_name} as it does not exist and is optional")
                    continue
                else:
                    raise Exception(f"Dataset {feature_name} does not exist and is not optional")
                
            reordered_cols = list(df.columns.sort_values())
            df = df[reordered_cols]    
            final_df = final_df.merge(df, how='left', on=feature_join_config.merge_on)

        feature_groups[feature_join_config.feature_group] = [x for x in df.columns if x not in exclude_columns]

        dataset_provider.set(dataset_name=feature_name, step=CURRENT_STEP, df=final_df)
        del df

    columns_to_drop = final_df.columns[
        (final_df.columns.str.contains('_masterpatientid|_facilityid|_x$|_y$|^patientid')) | (final_df.columns.duplicated())
    ].tolist()
    if len(columns_to_drop) > 0:
        final_df.drop(columns_to_drop, axis=1, inplace=True)

    log.info(f'Number of columns in the dataframe: {final_df.shape[1]}')

    # Write to new parquet file
    dataset_provider.set(dataset_name='final_df', step=CURRENT_STEP, df=final_df)

    log.info(f'Memory usage of final dataframe: {get_memory_usage(final_df)}')
    log.info(f'Final dataframe shape: {final_df.shape}')

    nan_cols = [i for i in final_df.columns if final_df[i].isna().any() and 'e_' not in i]

    log.info(f'Number of columns with NaNs: {len(nan_cols)}')

    del final_df

    dataset_provider.store_json(filename='feature_groups', step=CURRENT_STEP, data=feature_groups)
    
    dataset_provider.store_config(step=CURRENT_STEP)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(post_feature_engineering_merge)