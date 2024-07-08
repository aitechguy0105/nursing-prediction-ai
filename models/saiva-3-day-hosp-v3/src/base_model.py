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
import pickle
import shutil
import subprocess
from typing import Optional

import pandas as pd
from eliot import start_action, start_task, log_message
from saiva_internal_sdk import OrganizationMlModelConfig, FacilityMlModelConfig

from explanations.config import FEATURE_TYPE_MAPPING
from ranking import Ranking
from shared.constants import CACHE_STRATEGY_OPTIONS
from shared.constants import CHECK_CACHE_FIRST
from shared.data_manager import DataManager
from shared.database import DbEngine

def handle_delete_error(func, path, exc_info):
    """
    handler if you get exceptions when deleting model files from local disk
    """
    log_message(message_type="warning", file_not_deleted=path, exception=exc_info)


class BasePredictions:
    def __init__(self):
        self.client = None
        self.facilityid = None
        self.training_start_date = None
        self.prediction_date = None
        self.sub_clients = []
        self.subclient_group_level = None
        self.test = False
        self.replace_existing_predictions = False
        self.test_group_percentage = 1.0
        self.cache_s3_folder = "raw"
        self.cache_strategy = CHECK_CACHE_FIRST
        self.save_outputs_in_s3 = False
        self.save_outputs_in_local = False
        self.local_folder = None
        self.saiva_engine = None
        self.modelid = None
        self.iden_cols = [
            "censusdate",
            "masterpatientid",
            "facilityid",
            "bedid",
            "admissionstatus",
            "censusactioncode",
            "payername",
            "payercode",
            "to_from_type",
        ]
        self.org_ml_model_config: Optional[OrganizationMlModelConfig] = None
        self.facility_ml_model: Optional[FacilityMlModelConfig] = None
        self.group_level = None

    def _run_asserts(self):
        assert (
            self.cache_strategy in CACHE_STRATEGY_OPTIONS
        ), f'"{self.cache_strategy}" option for cache_strategy is not valid - it should be one of {CACHE_STRATEGY_OPTIONS}'

        if self.save_outputs_in_local and self.local_folder is None:
            assert (
                False
            ), "local_folder cannot be empty if save_outputs_in_local is True"

    def _create_local_directory(self):
        """
        For test purpose save intermediate outputs in local folders.
        Create the directory if it does not exists
        """
        if self.save_outputs_in_local and self.local_folder is not None:
            if not os.path.exists(self.local_folder):
                os.makedirs(self.local_folder)

    def _load_db_engines(self):
        engine = DbEngine()
        self.saiva_engine = engine.get_postgresdb_engine()
        self.client_sql_engine = engine.get_sqldb_engine(
            db_name=self.facility_ml_model.ml_model.source_database_name,
            credentials_secret_id=self.facility_ml_model.ml_model.source_database_credentials_secret_id,
            query={"driver": "ODBC Driver 17 for SQL Server"},
        )

    def download_prediction_models(self):
        """
        Downlaod the relevant prediction models required for facility's from S3
        """
        trained_model = self.facility_ml_model.trained_model
        with start_action(action_type="download_prediction_models"):
            subprocess.run(
                f"aws s3 sync s3://saiva-models/{Path(trained_model.model_s3_folder) / trained_model.mlflow_model_id} /data/models/{trained_model.mlflow_model_id}",
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def delete_downloaded_models(self):
        """
        Delete prediction models downloaded to the disk
        :param facilityid: string
        """
        with start_action(action_type="DELETE_DOWNLOADED_MODELS"):
            dir_path = "/data/models"
            log_message(message_type="info", dir_path_to_delete=dir_path)
            # Delete all contents of a directory and handle errors
            shutil.rmtree(dir_path, onerror=handle_delete_error)

    def _get_s3_path(self, *, prediction_date):
        ml_model = self.facility_ml_model.ml_model
        return f"s3://{Path(ml_model.output_s3_bucket) / ml_model.output_s3_prefix / prediction_date / str(self.facilityid) / self.cache_s3_folder}"

    def _fill_na(self, *, df):
        mlflow_model_id = self.facility_ml_model.trained_model.mlflow_model_id
        with open(
            f"/data/models/{mlflow_model_id}/artifacts/na_filler.pickle", "rb"
        ) as f:
            na_filler = pickle.load(f)

        return df.fillna(na_filler)

    def feature_logging(self, facilityid, x_frame, model_feats):
        """
        Map feature names to groups
        Eg:
        TOTAL_FEATURES_BEFORE_DROP: 3500
        TOTAL_FEATURES_AFTER_DROP: 2500
        TOTAL_FEATURES_DROPPED: 1500
        TOTAL_FEATURES_MISSING: 2500 - (3500 - 1500)
        """
        training_feats = pd.DataFrame({"feature": list(x_frame.columns)})
        training_feats["feature_group"] = training_feats.feature.replace(
            FEATURE_TYPE_MAPPING, regex=True
        )
        features = (
            training_feats.groupby("feature_group")["feature_group"].count().to_dict()
        )
        log_message(
            message_type="info",
            title="MODEL_FEATURES: TOTAL_FEATURES_BEFORE_DROP",
            feature_count=len(training_feats.feature),
            feature_group=features,
            facilityid=facilityid,
            client=self.client,
        )
        # ====================================================================================
        dropped_columns = set(training_feats.feature).difference(
            set(model_feats.feature)
        )
        log_message(
            message_type="info",
            title="MODEL_FEATURES: TOTAL_FEATURES_DROPPED",
            feature_count=len(dropped_columns),
            features=f"DROPPED_COLUMNS: {dropped_columns}",
            facilityid=facilityid,
            client=self.client,
        )
        # ====================================================================================
        missing_columns = set(model_feats.feature).difference(
            set(training_feats.feature)
        )
        missing_feats = pd.DataFrame({"feature": list(missing_columns)})
        missing_feats["feature_group"] = missing_feats.feature.replace(
            FEATURE_TYPE_MAPPING, regex=True
        )
        features = (
            missing_feats.groupby("feature_group")["feature_group"].count().to_dict()
        )
        log_message(
            message_type="info",
            title="MODEL_FEATURES: TOTAL_FEATURES_MISSING",
            feature_count=len(missing_columns),
            feature_group=features,
            features=f"MISSING_COLUMNS: {missing_columns}",
            facilityid=facilityid,
            client=self.client,
        )
        # ====================================================================================
        model_feats["feature_group"] = model_feats.feature.replace(
            FEATURE_TYPE_MAPPING, regex=True
        )
        features = (
            model_feats.groupby("feature_group")["feature_group"].count().to_dict()
        )
        log_message(
            message_type="info",
            title="MODEL_FEATURES: TOTAL_FEATURES_AFTER_DROP",
            feature_count=len(model_feats.feature),
            feature_group=features,
            facilityid=facilityid,
            client=self.client,
        )

    def filter_hospice_patients(self, df):
        """
        Filter out hospice patients.
        Filter out pre-configured censusactioncodes which needs to be excluded.
        """
        censusactioncodes = self.facility_ml_model.ml_model.excluded_censusactioncodes
        hospice_condition = df["censusactioncode"].isin(censusactioncodes) | (
            df["payername"].str.contains("hospice", case=False)
        )
        hospice_df = df[hospice_condition]
        final_df = df[~hospice_condition]

        if not hospice_df.empty:
            hospice_df = hospice_df[
                self.iden_cols
            ]  # Retain only iden_cols for hospice_df

        return final_df, hospice_df

    def _prep(self, *, df, prediction_date):
        mlflow_model_id = self.facility_ml_model.trained_model.mlflow_model_id

        drop_cols = self.iden_cols + [col for col in df.columns if "target" in col]
        x_frame = df.drop(columns=drop_cols).reset_index(drop=True).astype("float32")
        idens = df.loc[:, self.iden_cols]
        idens = idens.reset_index(drop=True)

        # Get all features for this model and reindex the x_frame
        all_feats = pd.read_csv(
            f"/data/models/{mlflow_model_id}/artifacts/input_features.csv"
        )
        self.feature_logging(self.facilityid, x_frame, all_feats)
        x_frame = x_frame.reindex(columns=all_feats.feature, fill_value=0)

        self._save_dataframe(
            df=x_frame,
            filename="finalx_output",
            prediction_date=prediction_date
        )

        return x_frame, idens

    def _save_dataframe(self, *, df, filename, prediction_date):
        """
        Save the final input dataframe & output predictions for testing
        :param df: Can be either Final X-frame or generated predictions
        :param filename: Filename to save the dataframe
        """
        if self.save_outputs_in_s3:
            final_x_s3_path = (self._get_s3_path(prediction_date=prediction_date) + f"/{filename}.parquet")
            df.to_parquet(final_x_s3_path, index=False)

        if self.save_outputs_in_local:
            final_x_local_path = self.local_folder + f"/{filename}.parquet"
            df.to_parquet(final_x_local_path, index=False)

    def feature_engineering(self, *, result_dict, prediction_date):
        trained_model = self.facility_ml_model.trained_model
        with start_task(
            action_type="feature-engineering", client=self.client, facilityid=self.facilityid
        ):
            dm = DataManager(
                result_dict=result_dict,
                facilityid=self.facilityid,
                client=self.client,
                prediction_date=prediction_date,
                train_start_date=trained_model.train_start_date,
                save_outputs_in_s3=self.save_outputs_in_s3,
                s3_base_path=self._get_s3_path(prediction_date=prediction_date),
                save_outputs_in_local=self.save_outputs_in_local,
                local_folder=self.local_folder,
                diagnosis_lookup_ccs_s3_file_path=trained_model.model_type_version.diagnosis_lookup_ccs_s3_uri
            )
            final = dm.get_features()
            result_dict = dm.get_modified_result_dict()

            drop_cols = [col for col in final.columns if "target" in col]
            final = final.drop(columns=drop_cols)

            # -----------------------------Handle empty values------------------------------
            final = self._fill_na(df=final)

            # TODO: clean this code which drops NaN. This causes issues when ever we introduce new columns
            exclude_cols = [
                "bedid",
                "admissionstatus",
                "censusactioncode",
                "payername",
                "payercode",
                "to_from_type",
            ]
            include_cols = final.columns.difference(exclude_cols)
            log_message(
                message_type="info",
                action_type="x_frame_rows_before_dropna",
                x_frame_shape=final.shape,
            )
            final = final.dropna(subset=include_cols)
            log_message(
                message_type="info",
                action_type="x_frame_rows_after_dropna",
                x_frame_shape=final.shape,
            )

        return final, result_dict

    def prediction(
        self,
        *,
        model,
        final_x,
        final_idens,
        prediction_date,
    ):
        with start_action(
            action_type="generate_predictions",
            facilityid=self.facilityid,
            client=self.client,
        ):
            preds = model.predict(final_x)
            predictions = final_idens
            predictions["facilityid"] = self.facilityid
            predictions["modelid"] = self.facility_ml_model.trained_model.mlflow_model_id
            predictions["predictionvalue"] = preds
            predictions["censusdate"] = prediction_date
            predictions["client"] = self.client
            predictions["ml_model_org_config_id"] = self.facility_ml_model.ml_model.id

            log_message(
                message_type="info",
                num_patients=len(pd.unique(predictions["masterpatientid"])),
            )
            log_message(
                message_type="info",
                min_predictions=predictions["predictionvalue"].min(),
            )
            log_message(
                message_type="info", max_prediction=predictions["predictionvalue"].max()
            )

            predictions = predictions.reindex(
                columns=[
                    "masterpatientid",
                    "facilityid",
                    "bedid",
                    "censusdate",
                    "predictionvalue",
                    "modelid",
                    "ml_model_org_config_id",
                    "client",
                    "admissionstatus",
                    "censusactioncode",
                    "payername",
                    "payercode",
                    "to_from_type",
                ]
            )

        ranking = Ranking(
            predictions=predictions,
            facilityid=self.facilityid,
            client=self.client,
            prediction_date=prediction_date,
            saiva_engine=self.saiva_engine,
            client_engine=self.client_sql_engine,
            group_level=self.group_level,
            subclients=self.sub_clients,
            replace_existing_predictions=self.replace_existing_predictions,
            test=self.test,
            facility_ml_model=self.facility_ml_model,
            subclient_group_level=self.subclient_group_level,
        )
        predictions = ranking.execute()
        self._save_dataframe(
            df=predictions,
            filename="predictions_output",
            prediction_date=prediction_date
        )
        return predictions

    def insert_hospice_patients(self, *, hospice_df, prediction_date):
        """
        Insert Hospice & other excluded censusactioncode patients into daily_prediction table
        """

        if not hospice_df.empty:
            hospice_df["censusdate"] = prediction_date
            hospice_df["client"] = self.client

            log_message(
                message_type="info",
                message=f"Delete all old facility hospice rows for the given date: {prediction_date}",
            )
            self.saiva_engine.execute(
                f"""delete from daily_predictions 
                    where censusdate = '{prediction_date}' 
                    and facilityid = {self.facilityid} and 
                    client = '{self.client}' and predictionrank IS NULL
                """
            )

            log_message(
                message_type="info",
                message=f"Save facility hospice data for the given date: {prediction_date}, {self.client}, {self.facilityid}",
            )
            hospice_df['is_exposed'] = self.org_ml_model_config.is_exposed
            hospice_df.to_sql(
                "daily_predictions",
                self.saiva_engine,
                method="multi",
                if_exists="append",
                index=False,
            )
