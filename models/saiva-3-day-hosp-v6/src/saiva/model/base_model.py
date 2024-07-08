"""
Run Command :
python /src/run_model.py --client infinity-benchmark --facilityids '[37]' --prediction-date 2020-05-02
--s3-bucket saiva-dev-data-bucket --replace_existing_predictions True
--save-outputs-in-local True --local-folder /data/test
--sub_clients : Comma seperated sub-client names for the same data feed
--group_level : Indicate whether the ranking will be done on facility/unit/floor
"""

import os
from pathlib import Path
import re
import pickle
import shutil
import subprocess
import sys
import typing

import datetime
from omegaconf import OmegaConf
import pandas as pd
from saiva_internal_sdk import TrainedModel
from eliot import start_action, start_task, log_message
import sqlalchemy


from .explanations.config import FEATURE_TYPE_MAPPING
from .ranking import Ranking
from .shared.constants import saiva_api
from .shared.constants import CACHE_STRATEGY_OPTIONS
from .shared.constants import CHECK_CACHE_FIRST
from .shared.data_manager import DataManager
from .shared.database import DbEngine
from .shared.utils import url_encode_cols, get_client_class
from saiva.training.core import BaseModel
from saiva.model.shared.load_raw_data import fetch_prediction_data
from saiva.model.run_explanation import generate_explanations


def handle_delete_error(func, path, exc_info):
    """
    handler if you get exceptions when deleting model files from local disk
    """
    log_message(message_type='warning', file_not_deleted=path, exception=exc_info)


class Inference:
    def __init__(
            self,
            client: str,
            prediction_date: str,
            facilityid: int,
            ml_model_org_config_id: str,
            mlflow_model_id: typing.Optional[str] = None,
            sub_clients: typing.Optional[typing.List[str]] = None,
            test: bool = False,
            replace_existing_predictions: bool = False,
            test_group_percentage: float = 1.0,
            cache_s3_folder: str = 'raw',
            cache_strategy: str = CHECK_CACHE_FIRST,
            save_outputs_in_s3: bool = False,
            save_outputs_in_local: bool = False,
            local_folder=None,
            cleanup_downloaded_models=True,
            data_facilities=None,
    ):

        self.client = client
        self.prediction_date = prediction_date
        self.facility_id = facilityid
        self.ml_model_org_config_id = ml_model_org_config_id
        self.data_facilities = data_facilities

        self._model = None
        self._config = None

        self.facility_ml_model = (
            saiva_api.organization_ml_model_configs.get_facility_ml_model(
                org_id=self.client,
                ml_model_org_config_id=self.ml_model_org_config_id,
                customers_facility_identifier=self.facility_id,
            )
        )

        # Override the trained model if mlflow_model_id is provided
        if mlflow_model_id:
            self.facility_ml_model.trained_model = saiva_api.trained_models.get_by_mlflow_model_id(
                mlflow_model_id=mlflow_model_id
            )

        self.organization_ml_model_config = saiva_api.organization_ml_model_configs.get(
            org_id=client,
            ml_model_org_config_id=ml_model_org_config_id
        )

        self.model_type = self.organization_ml_model_config.quality_measure
        self.test = test
        self.replace_existing_predictions = replace_existing_predictions
        self.test_group_percentage = test_group_percentage
        self.cache_s3_folder = cache_s3_folder
        self.cache_strategy = cache_strategy
        self.save_outputs_in_s3 = save_outputs_in_s3
        self.save_outputs_in_local = save_outputs_in_local
        self.local_folder = local_folder

        if sub_clients is not None:
            raise NotImplementedError("The subclient functionality is not supported in v6. "
                                      "For more information see SAIV-2993")
        self.sub_clients = []

        self.group_level = self.facility_ml_model.group_level
        self.cleanup_downloaded_models = cleanup_downloaded_models

        self._db_engine = None
        self._saiva_engine = None
        self._client_sql_engine = None

        self._run_asserts()
        self._create_local_directory()

    @property
    def db_engine(self) -> DbEngine:
        with start_action(action_type="Connecting with database"):
            if self._db_engine is None:
                self._db_engine = DbEngine()
            return self._db_engine

    @property
    def saiva_engine(self) -> sqlalchemy.engine.Engine:
        if self._saiva_engine is None:
            self._saiva_engine = self.db_engine.get_postgresdb_engine()
        return self._saiva_engine

    @property
    def client_sql_engine(self) -> sqlalchemy.engine.Engine:
        if self._client_sql_engine is None:
            self._client_sql_engine = self.db_engine.get_sqldb_engine(
                db_name=self.facility_ml_model.ml_model.source_database_name,
                credentials_secret_id=self.facility_ml_model.ml_model.source_database_credentials_secret_id,
                query={"driver": "ODBC Driver 17 for SQL Server"},
            )
        return self._client_sql_engine

    def _run_asserts(self):
        assert self.cache_strategy in CACHE_STRATEGY_OPTIONS, (
            f"\"{self.cache_strategy}\" option for cache_strategy is not valid - "
            f"it should be one of {CACHE_STRATEGY_OPTIONS}"
        )
        if self.save_outputs_in_local and self.local_folder is None:
            assert False, 'local_folder cannot be empty if save_outputs_in_local is True'

        clientClass = get_client_class(self.client)
        facilities = getattr(clientClass(), 'facilities')
        db_facilities = list(pd.read_sql(
            f"""
            SELECT FacilityID FROM view_ods_facility
            WHERE FacilityID in ({facilities})
            """,
            self.client_sql_engine)['FacilityID'])
        assert len(db_facilities) > 0, (
            "The facilities specified in the client class don't overlap with facilities available in DB, "
            "check `self.facilities` in the client class and access to DB"""
        )
        assert self.facility_id in db_facilities, (
            f"facilityid={self.facility_id} is not in the list of the facilities we are using the data from. "
            f"Such behavior is prohibited. Change `self.facilities` in the client class or `facilityids` parameter"
        )

    def _create_local_directory(self):
        """
        For test purpose save intermediate outputs in local folders.
        Create the directory if it does not exists
        """
        if self.save_outputs_in_local and self.local_folder is not None:
            if not os.path.exists(self.local_folder):
                os.makedirs(self.local_folder)

    def download_prediction_models(self):
        """
        Downlaod the relevant prediction models required for facility's from S3
        """
        with start_action(action_type="download_prediction_models"):
            subprocess.run(
                f"aws s3 sync "
                f"s3://saiva-models/{Path(self.trained_model.model_s3_folder) / self.trained_model.mlflow_model_id} "
                f"/data/models/{self.trained_model.mlflow_model_id}",
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def delete_downloaded_models(self):
        """
        Delete prediction models downloaded to the disk
        :param facilityid: string
        """
        with start_action(action_type='DELETE_DOWNLOADED_MODELS'):
            dir_path = "/data/models"
            log_message(message_type='info', dir_path_to_delete=dir_path)
            # Delete all contents of a directory and handle errors
            shutil.rmtree(dir_path, onerror=handle_delete_error)

    def _get_s3_path(self, *, prediction_date: str):
        ml_model = self.facility_ml_model.ml_model
        s3_path = (
            Path(ml_model.output_s3_bucket) /
            ml_model.output_s3_prefix /
            prediction_date /
            str(self.facility_id) /
            self.cache_s3_folder
        )
        return f"s3://{s3_path}"

    def _fill_na(self, df: pd.DataFrame):
        """
        Aligns the dataframe `df` with the required columns order for the model.
        It drops columns that the model doesn't use and adds ones that don't exist in `df`.
        The values of the added columns are NaNs by default but are filled based on the
        configuration parameter `postprocessing.missing_column_fill_values`.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to be aligned with the model requirements.

        Returns:
        --------
        x_frame : pd.DataFrame
            The modified dataframe that meets the model's column order requirement.
        """

        # Get all features for this model and reindex the x_frame
        all_feats = self.model.feature_name()
        x_frame = df.reindex(columns=all_feats)

        for item in self.config.postprocessing.missing_column_fill_values:
            pattern, fill_value = item.get('pattern'), item.get('value')
            mask = x_frame.columns.str.contains(pattern, regex=True)
            x_frame.loc[:, mask] = x_frame.loc[:, mask].fillna(fill_value)

        return x_frame

    def feature_logging(self, facilityid, x_frame, model_feats):
        """
        Map feature names to groups
        Eg:
        TOTAL_FEATURES_BEFORE_DROP: 3500
        TOTAL_FEATURES_AFTER_DROP: 2500
        TOTAL_FEATURES_DROPPED: 1500
        TOTAL_FEATURES_MISSING: 2500 - (3500 - 1500)
        """
        training_feats = pd.DataFrame({'feature': list(x_frame.columns)})
        training_feats['feature_group'] = training_feats.feature.replace(
            FEATURE_TYPE_MAPPING,
            regex=True
        )
        features = training_feats.groupby('feature_group')['feature_group'].count().to_dict()
        log_message(message_type='info',
                    title='MODEL_FEATURES: TOTAL_FEATURES_BEFORE_DROP',
                    feature_count=len(training_feats.feature),
                    facilityid=facilityid,
                    client=self.client,
                    )
        # ====================================================================================
        dropped_columns = set(training_feats.feature).difference(set(model_feats.feature))
        log_message(message_type='info',
                    title='MODEL_FEATURES: TOTAL_FEATURES_DROPPED',
                    feature_count=len(dropped_columns),
                    facilityid=facilityid,
                    client=self.client,
                    )

        # ====================================================================================
        missing_columns = set(model_feats.feature).difference(set(training_feats.feature))
        missing_feats = pd.DataFrame({'feature': list(missing_columns)})
        missing_feats['feature_group'] = missing_feats.feature.replace(
            FEATURE_TYPE_MAPPING,
            regex=True
        )
        features = missing_feats.groupby('feature_group')['feature_group'].count().to_dict()
        log_message(message_type='info',
                    title='MODEL_FEATURES: TOTAL_FEATURES_MISSING',
                    feature_count=len(missing_columns),
                    feature_group=features,
                    features=f'MISSING_COLUMNS: {missing_columns}',
                    facilityid=facilityid,
                    client=self.client,
                    )
        # ====================================================================================
        model_feats['feature_group'] = model_feats.feature.replace(
            FEATURE_TYPE_MAPPING,
            regex=True
        )
        features = model_feats.groupby('feature_group')['feature_group'].count().to_dict()
        log_message(message_type='info',
                    title='MODEL_FEATURES: TOTAL_FEATURES_AFTER_DROP',
                    feature_count=len(model_feats.feature),
                    facilityid=facilityid,
                    client=self.client,
                    )

    def filter_hospice_patients(self, *, df: pd.DataFrame):
        """
        Filter out hospice patients.
        Filter out pre-configured censusactioncodes which needs to be excluded.
        """
        hospice_condition = (df['payername'].str.contains("hospice", case=False, na=False))
        # TODO: when we have a communication channel between prediction and report generator
        # we can send the warning to the report that if we have nulls in `payername` column
        # that we ranked a patient that can be on hospice
        hospice_df = df[hospice_condition]
        final_df = df[~hospice_condition]

        if not hospice_df.empty:
            hospice_df = hospice_df[self.config.iden_cols]  # Retain only iden_cols for hospice_df

        return final_df, hospice_df

    def _prep(self, *, df: pd.DataFrame, prediction_date: str):
        mlflow_model_id = self.trained_model.mlflow_model_id

        # Required for latest LGBM
        df = url_encode_cols(df)
        idens = pd.DataFrame()
        for col in self.config.iden_cols:
            if col in df.columns:
                idens[col] = df[col]
            else:
                idens[col] = ''
        idens = idens.reset_index(drop=True)

        # -----------------------------Handle empty values------------------------------
        x_frame = self._fill_na(df)
        numeric_col = x_frame.select_dtypes(include=['number']).columns
        x_frame.loc[:, numeric_col] = x_frame.loc[:, numeric_col].astype('float32')

        # add client name to facility id
        x_frame['facility'] = re.sub('_*v6.*', '', self.client) + "_" + df['facilityid'].astype(str)
        x_frame['facility'] = x_frame['facility'].astype('category')

        # make sure all categorial columns are are in category datatype
        with open(f'/data/models/{mlflow_model_id}/artifacts/cate_columns.pickle', 'rb') as f:
            cate_clos = pickle.load(f)
        x_frame[cate_clos] = x_frame[cate_clos].astype('category')

        # reindex the x_frame again to make sure the order of the columns is the same as the training set
        all_feats = pd.read_csv(
            f'/data/models/{mlflow_model_id}/artifacts/input_features.csv'
        )
        self.feature_logging(self.facility_id, x_frame, all_feats)
        x_frame = x_frame.reindex(columns=all_feats.feature)
        x_frame.reset_index(drop=True, inplace=True)
        self._save_dataframe(df=x_frame, filename='finalx_output', prediction_date=prediction_date)
        self._save_dataframe(df=idens, filename='idens_output', prediction_date=prediction_date)

        return x_frame, idens

    def _save_dataframe(self, *, df: pd.DataFrame, filename: str, prediction_date: str):
        """
        Save the final input dataframe & output predictions for testing
        :param df: Can be either Final X-frame or generated predictions
        :param filename: Filename to save the dataframe
        """
        if self.save_outputs_in_s3:
            final_x_s3_path = (self._get_s3_path(prediction_date=prediction_date) + f"/{filename}.parquet")
            df.to_parquet(final_x_s3_path, index=False)

        if self.save_outputs_in_local:
            final_x_local_path = self.local_folder + f'/{filename}.parquet'
            df.to_parquet(final_x_local_path, index=False)

    def fetch_prediction_data(self):
        with start_action(
            action_type="fetching_prediction_data",
            facilityid=self.facility_id,
            client=self.client,
        ):
            # ------------Fetch the data from client SQL db and store in S3 buckets----------
            result_dict = fetch_prediction_data(
                client_sql_engine=self.client_sql_engine,
                prediction_date=self.prediction_date,
                facilityid=self.facility_id,
                train_start_date=self.trained_model.train_start_date,
                client=self.client,
                cache_strategy=self.cache_strategy,
                s3_location_path_prefix=self._get_s3_path(prediction_date=self.prediction_date),
                local_folder=self.local_folder,
                save_outputs_in_local=self.save_outputs_in_local,
                save_outputs_in_s3=self.save_outputs_in_s3,
                config=self.config,
            )
            return result_dict

    def feature_engineering(self, *, result_dict, prediction_date):
        with start_task(action_type='feature-engineering', client=self.client, facilityid=self.facility_id):
            dm = DataManager(
                result_dict=result_dict,
                facilityid=self.facility_id,
                client=self.client,
                prediction_date=prediction_date,
                train_start_date=self.trained_model.train_start_date,
                save_outputs_in_s3=self.save_outputs_in_s3,
                s3_base_path=self._get_s3_path(prediction_date=prediction_date),
                save_outputs_in_local=self.save_outputs_in_local,
                local_folder=self.local_folder,
                model_config=self.config,
                diagnosis_lookup_ccs_s3_file_path=self.trained_model.model_type_version.diagnosis_lookup_ccs_s3_uri
            )
            final = dm.get_features()

            result_dict = dm.get_modified_result_dict()

            drop_cols = [col for col in final.columns if ('target' in col) or ('positive_date' in col)]
            final = final.drop(columns=drop_cols)

            log_message(message_type='info', action_type='x_frame_rows_before_dropna', x_frame_shape=final.shape)

        return final, result_dict

    def prediction(
        self,
        *,
        final_x,
        final_idens,
        prediction_date,
    ):
        with start_action(
            action_type="generate_predictions",
            facilityid=self.facility_id,
            client=self.client,
        ):
            preds = self.model.predict(final_x)
            predictions = final_idens
            predictions['facilityid'] = self.facility_id
            predictions['modelid'] = self.facility_ml_model.trained_model.mlflow_model_id
            predictions['predictionvalue'] = preds
            predictions['censusdate'] = prediction_date
            predictions['client'] = self.client
            predictions["ml_model_org_config_id"] = self.facility_ml_model.ml_model.id

            log_message(message_type='info', num_patients=len(pd.unique(predictions['masterpatientid'])))
            log_message(message_type='info', min_predictions=predictions['predictionvalue'].min())
            log_message(message_type='info', max_prediction=predictions['predictionvalue'].max())

            predictions = predictions.reindex(
                columns=[
                    'masterpatientid',
                    'facilityid',
                    'bedid',
                    'censusdate',
                    'predictionvalue',
                    'modelid',
                    'client',
                    'admissionstatus',
                    'censusactioncode',
                    'payername',
                    'payercode',
                    'to_from_type'
                ]
            )

        ranking = Ranking(
            predictions=predictions,
            facilityid=self.facility_id,
            modeltype=self.model_type,
            client=self.client,
            prediction_date=prediction_date,
            saiva_engine=self.saiva_engine,
            client_engine=self.client_sql_engine,
            group_level=self.group_level,
            subclients=self.sub_clients,
            replace_existing_predictions=self.replace_existing_predictions,
            test=self.test,
            facility_ml_model=self.facility_ml_model,
        )
        predictions = ranking.execute()

        duplicate_mask = predictions.loc[predictions['show_in_report']].duplicated(subset='predictionvalue', keep=False)

        if sum(duplicate_mask) > 0:
            log_message(
                message_type='info',
                message=f"Client {self.client} FacilityId {self.facility_id} {sum(duplicate_mask)} "
                        f"duplicate prediction probabilities detected..."
            )

        self._save_dataframe(
            df=predictions,
            filename="predictions_output",
            prediction_date=prediction_date
        )
        return predictions

    def insert_hospice_patients(self, *, hospice_df: pd.DataFrame, prediction_date: str):
        """
        Insert Hospice & other excluded censusactioncode patients into daily_prediction table
        """

        if not hospice_df.empty:
            hospice_df['censusdate'] = prediction_date
            hospice_df['client'] = self.client
            hospice_df['ml_model_org_config_id'] = self.facility_ml_model.ml_model.id

            log_message(
                message_type='info',
                message=f'Delete all old facility hospice rows for the given date: {prediction_date}'
            )
            self.saiva_engine.execute(
                f"""delete from daily_predictions
                    where censusdate = '{prediction_date}'
                    and facilityid = {self.facility_id}
                    and client = '{self.client}'
                    and ml_model_org_config_id = '{self.facility_ml_model.ml_model.id}'
                    and predictionrank IS NULL
                """
            )

            log_message(
                message_type="info",
                message=f"Save facility hospice data for the given date: "
                        f"{prediction_date}, {self.client}, {self.facility_id}"
            )
            hospice_df.drop('primaryphysicianid', inplace=True, axis=1)
            hospice_df['is_exposed'] = self.organization_ml_model_config.is_exposed
            hospice_df.to_sql(
                'daily_predictions',
                self.saiva_engine,
                method='multi',
                if_exists='append',
                index=False
            )

    def select_prediction_census(self, final: pd.DataFrame):
        """
        this function -
        1. stores all the residents present at 2am in the snf for prediction date 'D'
        2. for live run keeps only the patients who present on midnight and at the prediction time;
           for retrospective keeps only 'D-1' date data for prediction and keeps residents present at the midnight
        3. Converts date 'D-1' to 'D' so that the code doesnot break at explanation and reporting section.
        """

        last_day = (
            (pd.to_datetime(self.prediction_date) - pd.Timedelta(1, 'd'))
            .date()
            .strftime('%Y-%m-%d')  # day before prediction date
        )
        # the patient MUST be present at the `last_day` (i.e. midnight):
        predictiondate_mpid_list = final.loc[final.censusdate == last_day, 'masterpatientid'].unique().tolist()
        # for the live predictions patient also have to be present at the `self.prediction_date` (i.e. ~2am):
        census_day = self.prediction_date if self.is_live() else last_day
        final = final[(final.censusdate == census_day) & (final['masterpatientid'].isin(predictiondate_mpid_list))]
        final['censusdate'] = self.prediction_date
        final = final.loc[final['facilityid']==self.facility_id] # for more info see MLOP-919
        log_message(message_type='info',
                    title='Number of residents present on the facility on the prediction date',
                    dataframe_shape=final.shape[0]
                    )
        assert final.shape[0] > 0, "No residents present on the facility on the prediction date."
        return final

    def is_live(self):
        """
        Check if the prediction is run in a live mode, or restrospectively.
        The criteria - the prediction takes place before 11am UTC
        """
        delta_from_midnight = datetime.datetime.utcnow() - datetime.datetime.strptime(self.prediction_date, '%Y-%m-%d')
        return delta_from_midnight <= datetime.timedelta(hours=11)

    def _load_model(self) -> BaseModel:  # todo return type
        """
        Load the model from s3
        """
        self.download_prediction_models()

        with start_action(action_type='load_model',
                          facilityid=self.facility_id,
                          modelid=self.trained_model.mlflow_model_id,
                          client=self.client):
            with open(f"/data/models/{self.trained_model.mlflow_model_id}"
                      f"/artifacts/{self.trained_model.mlflow_model_id}.pickle", "rb") as f:
                try:
                    model = pickle.load(f)
                except ModuleNotFoundError:
                    import saiva.training as training_module

                    sys.modules['training'] = training_module
                    model = pickle.load(f)
            model.truncate_v6_suffix()
            return model

    @property
    def model(self) -> BaseModel:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_config(self) -> OmegaConf:
        default_config = OmegaConf.load('/src/saiva/conf/prediction/defaults.yaml')
        if getattr(self.model, 'config', None):
            config = OmegaConf.create(self.model.config)
            config = OmegaConf.merge(default_config, config)
            if self.data_facilities:
                if config.client_configuration.multiple_clients:
                    config.client_configuration.facilities[re.sub('_*v6.*', '', self.client)] = self.data_facilities
                else:
                    config.client_configuration.facilities = self.data_facilities
        else:
            config = default_config
            
        return config

    @property
    def config(self) -> OmegaConf:
        """
        Load the config from the disk
        """
        if self._config is None:
            self._config = self._load_config()
        return self._config

    @property
    def trained_model(self) -> TrainedModel:
        return self.facility_ml_model.trained_model

    def _filter_show_in_report_patients(
        self,
        *,
        predictions_df: pd.DataFrame,
        pd_final_df: pd.DataFrame,
        pd_final_idens: pd.DataFrame
    ):
        # Adding a new column 'calculate_shap' to the final dataframe
        # This columns is True for all the masterpatientids that are ranked <= cutoff.
        pd_final_idens['calculate_shap'] = False
        masterpatientids = predictions_df.query('show_in_report == True')['masterpatientid'].tolist()
        pd_final_idens['calculate_shap'] = pd_final_idens['masterpatientid'].isin(masterpatientids)
        pd_final_df = pd_final_df.reset_index(drop=True)
        pd_final_idens = pd_final_idens.reset_index(drop=True)

        return pd_final_df, pd_final_idens

    def generate_explanations(
        self,
        *,
        predictions_df: pd.DataFrame,
        final_df: pd.DataFrame,
        final_idens: pd.DataFrame,
        result_dict: dict
    ):
        with start_action(action_type='generate_explanations', facilityid=self.facility_id, client=self.client):
            final_df, final_idens = self._filter_show_in_report_patients(
                predictions_df=predictions_df,
                pd_final_df=final_df,
                pd_final_idens=final_idens
            )
            generate_explanations(
                model=self.model,
                final_x=final_df,
                final_idens=final_idens,
                raw_data_dict=result_dict,
                client=self.client,
                s3_location_path_prefix=self._get_s3_path(prediction_date=self.prediction_date),
                save_outputs_in_local=self.save_outputs_in_local,
                local_folder=self.local_folder,
                ml_model=self.facility_ml_model.ml_model,
            )
