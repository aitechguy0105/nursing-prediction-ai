# """
# Run Command :
# pipenv run python /src/facility_setup.py execute --organization=trio --facility_id=1
# """

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
import datetime
import pickle
import subprocess
import sys

from sklearn.metrics import roc_auc_score
import fire
import pandas as pd

from data_models import BaseModel
from saiva_internal_sdk import TrainedModel, OrganizationMlModelConfig, FacilityMlModelConfigCreate
from shared.featurizer import BaseFeaturizer
from shared.constants import saiva_api
import clients.base
import shared.database as database
import shared.utils as utils
import shared.load_raw_data as load_raw_data


from eliot import to_file, start_action, log_message
to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


def get_unique_models_info(
    *,
    ml_model_org_config: OrganizationMlModelConfig,
) -> List[TrainedModel]:
    facilities = ml_model_org_config.facilities

    unique_model_ids = set()
    unique_models = []

    for facility in facilities:
        model = facility.trained_model
        if model is not None and model.id not in unique_model_ids:
            unique_model_ids.add(model.id)
            unique_models.append(model)

    return unique_models


class SetupFacility(object):
    def __init__(
        self,
        *,
        organization: str,
        facility_id: int,
        ml_model_org_config_id: str,
        data_max_days: int = 90,
        data_maturity_days: int = 7,
    ) -> None:

        self.org_id: str = organization
        self.org_class: clients.base.Base = utils.get_client_class(self.org_id)()
        self.facility_id: int = facility_id
        self.ml_model_org_config_id: str = ml_model_org_config_id
        self.ml_model_org_config: OrganizationMlModelConfig = saiva_api.organization_ml_model_configs.get(
            org_id=self.org_id,
            ml_model_org_config_id=self.ml_model_org_config_id,
        )

        engine = database.DbEngine()
        self.saiva_engine = engine.get_postgresdb_engine()
        self.org_sql_engine = engine.get_sqldb_engine(
            db_name=self.ml_model_org_config.source_database_name,
            credentials_secret_id=self.ml_model_org_config.source_database_credentials_secret_id,
            query={"driver": "ODBC Driver 17 for SQL Server"},
        )

        self.processed_path = Path('/data/processed')
        self.raw_path = Path('/data/raw')

        self.start_date, self.end_date = self.get_test_dates(
            data_max_days=data_max_days,
            data_maturity_days=data_maturity_days,
        )

    def get_test_dates(
        self,
        *,
        data_max_days,
        data_maturity_days,
    ) -> Tuple[str, str]:
        """
        Returns date in the past no older than 3 month but also not earlier that test_start_date from organization class
        """
        minimum_start = datetime.date.today() - datetime.timedelta(days=data_max_days + data_maturity_days)
        end = datetime.date.today() - datetime.timedelta(days=data_maturity_days)

        latest_test_start_date: datetime.date = self.ml_model_org_config.facilities[0].trained_model.test_start_date
        for facility_config in self.ml_model_org_config.facilities:
            if facility_config.trained_model.test_start_date > latest_test_start_date:
                latest_test_start_date = facility_config.trained_model.test_start_date

        assert minimum_start > latest_test_start_date, "Data is not mature enough"

        return latest_test_start_date.isoformat(), end.isoformat()

    def filter_for_facility(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return
        if 'facilityid' not in df.columns:
            return df

        return df.loc[df['facilityid'] == self.facility_id]

    def feature_engineering(self):
        from shared.demographics import DemographicFeatures
        from shared.vitals import VitalFeatures
        from shared.rehosp import RehospFeatures
        from shared.notes import NoteFeatures
        from shared.diagnosis import DiagnosisFeatures
        from shared.patient_census import PatientCensus
        from shared.admissions import AdmissionFeatures

        self.processed_path.mkdir(parents=True, exist_ok=True)

        training_data_dict = load_raw_data.fetch_training_cache_data(client=self.org_id, generic=True)

        patient_census = PatientCensus(
            census_df=self.filter_for_facility(training_data_dict.get('patient_census', None)),
            train_start_date=self.start_date,
            test_end_date=self.end_date,
        )
        census_df = patient_census.generate_features()
        census_df.to_parquet(self.processed_path/'census_df.parquet')

        featurizer_dict: Dict[Type[BaseFeaturizer], Dict[str, str]] = {
            DemographicFeatures: {
                'key_in': 'patient_demographics',
                'name_out': 'demo_df.parquet',
            },
            VitalFeatures: {
                'key_in': 'patient_vitals',
                'name_out': 'vitals_df.parquet',
            },
            RehospFeatures: {
                'key_in': 'patient_rehosps',
                'name_out': 'rehosp_df.parquet',
            },
            AdmissionFeatures: {
                'key_in': 'patient_admissions',
                'name_out': 'admissions_df.parquet',
            },
            DiagnosisFeatures: {
                'key_in': 'patient_diagnosis',
                'name_out': 'diagnosis_df.parquet',
                'kwargs': {
                    'diagnosis_lookup_ccs_s3_file_path': self.ml_model_org_config.facilities[0].trained_model.model_type_version.diagnosis_lookup_ccs_s3_uri
                }
            },
        }

        for featurizer_class, data in featurizer_dict.items():
            featurizer = featurizer_class(
                census_df.copy(),
                self.filter_for_facility(training_data_dict.get(data['key_in'], None)),
                training=True,
                **data.get('kwargs', dict()),
            )
            features = featurizer.generate_features()

            if isinstance(features, pd.DataFrame):
                features.to_parquet(self.processed_path/data['name_out'])
            else:
                features[0].to_parquet(self.processed_path/data['name_out'])

        if not training_data_dict.get('patient_progress_notes', pd.DataFrame()).empty:
            notes = NoteFeatures(
                census_df=census_df.copy(),
                notes=self.filter_for_facility(training_data_dict.get('patient_progress_notes', None)),
                client=self.org_id,
                training=True
            )

            notes.generate_features().to_parquet(self.processed_path/'notes_df.parquet')

    def merge_data(self) -> pd.DataFrame:
        with start_action(action_type="merge_data"):
            demo_df = pd.read_parquet(self.processed_path/'demo_df.parquet')
            vitals_df = pd.read_parquet(self.processed_path/'vitals_df.parquet')
            rehosp_df = pd.read_parquet(self.processed_path/'rehosp_df.parquet')
            admissions_df = pd.read_parquet(self.processed_path/'admissions_df.parquet')
            diagnosis_df = pd.read_parquet(self.processed_path/'diagnosis_df.parquet')

            merge_on = ['masterpatientid', 'facilityid', 'censusdate']
            final_df = demo_df.merge(vitals_df, how='left', left_on=merge_on, right_on=merge_on)\
                .merge(rehosp_df, how='left', left_on=merge_on, right_on=merge_on)\
                .merge(admissions_df, how='left', left_on=merge_on, right_on=merge_on)\
                .merge(diagnosis_df, how='left', left_on=merge_on, right_on=merge_on)

            if Path.exists(self.processed_path/'notes_df.parquet'):
                final_df = final_df.merge(
                    pd.read_parquet(self.processed_path/'notes_df.parquet'),
                    how='left', left_on=merge_on, right_on=merge_on
                )

            final_df.drop(
                final_df.columns[final_df.columns.str.contains('_masterpatientid|_facilityid|_x$|_y$')].tolist(),
                axis=1,
                inplace=True
            )

            if 'patientid' in final_df.columns:
                # If patientid included in the above regex pattern it drops masterpatientid column even
                final_df.drop(['patientid'], axis=1, inplace=True)

            return final_df

    def load_unique_models(self) -> Dict[str, BaseModel]:
        with start_action(action_type="load_unique_models"):
            result = {}

            for model in get_unique_models_info(
                ml_model_org_config=self.ml_model_org_config,
            ):
                model_id = model.mlflow_model_id.strip()
                model_s3_folder = model.model_s3_folder.strip()

                with start_action(action_type="loading model", model_path=model_s3_folder, model_id=model_id):
                    subprocess.run(
                        f'aws s3 sync s3://saiva-models/{model_s3_folder}/{model_id} /data/models/{model_id}',
                        shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    )

                    with open(f'/data/models/{model_id}/artifacts/{model_id}.pickle', 'rb') as f:
                        result[model.id] = pickle.load(f)

            return result

    def fill_na(self, *, model: BaseModel, df: pd.DataFrame) -> pd.DataFrame:
        with open(f'/data/models/{model.model_name}/artifacts/na_filler.pickle', 'rb') as f:
            na_filler = pickle.load(f)

        return df.fillna(na_filler)

    def prep(self, *, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        iden_cols = ['censusdate', 'masterpatientid', 'facilityid', 'bedid',
                     'censusactioncode', 'payername', 'payercode', 'dateofadmission']

        drop_cols = iden_cols + [col for col in df.columns if 'target' in col]

        y = df['hosp_target_3_day_hosp'].astype('float32').values

        x = df.drop(columns=drop_cols).reset_index(drop=True).astype('float32')

        idens = df.loc[:, iden_cols]

        return x, y, idens

    def execute(self):
        with start_action(action_type="fetch_test_data"):
            load_raw_data.fetch_training_data(
                client=self.org_id,
                client_sql_engine=self.org_sql_engine,
                train_start_date=self.start_date,
                test_end_date=self.end_date,
                excluded_censusactioncodes=self.ml_model_org_config.excluded_censusactioncodes,
            )

        for ft in load_raw_data.get_genric_file_names(data_path=self.raw_path, client=self.org_id):
            org_df = pd.read_parquet(self.raw_path / f'{self.org_id}_{ft}.parquet')
            org_df['masterpatientid'] = org_df['masterpatientid'].apply(lambda x: self.org_id + '_' + str(x))

            if ft == 'patient_demographics':
                org_df['dateofbirth'] = org_df['dateofbirth'].astype('datetime64[ms]')

            if ft == 'patient_diagnosis':
                org_df['onsetdate'] = org_df['onsetdate'].astype('datetime64[ms]')

            org_df.to_parquet(self.raw_path/f'{ft}.parquet')

        self.feature_engineering()

        model_input_data = self.merge_data()

        models = self.load_unique_models()
        assert len(models) >= 1, "There were no existing models found"

        best_roc_auc = 0
        best_model_id = None

        for model_id, model in models.items():
            try:
                with start_action(action_type=f'run_model_{model.model_name}'):
                    filled_data = self.fill_na(
                        model=model,
                        df=model_input_data
                    )

                    x, y, idens = self.prep(
                        df=self.filter_for_facility(filled_data)
                    )

                    preds = model.predict(x)
                    roc_auc = roc_auc_score(y, preds)

                    log_message(message_type='info', model_id=model.model_name, roc_auc=roc_auc)
                    if roc_auc >= best_roc_auc:
                        best_roc_auc = roc_auc
                        best_model_id = model_id
            except Exception:
                pass

        assert best_model_id is not None, "Best model not foound"
        log_message(
            message_type='info',
            best_model_id=best_model_id,
            best_roc_auc=best_roc_auc,
            best_mlflow_model_id=models[best_model_id].model_name,
        )

        facility_ml_model_config = FacilityMlModelConfigCreate(
            missing_datasets=[],
            group_level='facility',
            golive_date=datetime.date.today(),
            trained_model_id=best_model_id,
            customers_facility_identifier=str(self.facility_id),
        )

        facility_ml_model_config = saiva_api.organization_ml_model_configs.create_facility_ml_model(
            org_id=self.org_id,
            ml_model_org_config_id=self.ml_model_org_config_id,
            facility_ml_model_config=facility_ml_model_config,
        )


if __name__ == '__main__':
    setup_facility = fire.Fire(SetupFacility)
