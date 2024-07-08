"""
Run Command :
python /src/run_model.py 
--client infinity-benchmark 
--facilityid '37'
--prediction-date 2020-05-02
--ml_model_org_config_id infinity-benchmark
--mlflow-model-id : Optional override for the trained model
--test False
--replace_existing_predictions True
--test_group_percentage 1.0
--cache-s3-folder raw
--cache-strategy CHECK_CACHE_FIRST
--save-outputs-in-s3 False
--save-outputs-in-local True 
--local-folder /data/test
--cleanup-downloaded-models True
--disable-instrumentation True
"""

import json
import pickle
import sys

import boto3
import fire
from eliot import start_action, start_task, to_file, log_message
import sentry_sdk

from base_model import BasePredictions
from data_models import BaseModel  # noqa: F401
from run_explanation import generate_explanations
from shared.constants import CHECK_CACHE_FIRST, ENV, REGION_NAME, saiva_api
from shared.load_raw_data import fetch_prediction_data
from decorators import instrumented

to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


class RunPredictions(BasePredictions):

    def _filter_show_in_report_patients(self, predictions_df, pd_final_df, pd_final_idens):
        masterpatientids = predictions_df.query('show_in_report == True')['masterpatientid'].tolist()
        pd_final_idens = pd_final_idens.query('masterpatientid in @masterpatientids')
        pd_final_df = pd_final_df[pd_final_df.index.isin(list(pd_final_idens.index))]
        
        pd_final_df = pd_final_df.reset_index(drop=True)
        pd_final_idens = pd_final_idens.reset_index(drop=True)

        return pd_final_df, pd_final_idens

    def execute(self):
        try:
            self._run_asserts()
            self._create_local_directory()
            self._load_db_engines()

            with start_task(action_type='run_model', client=self.client):
                trained_model = self.facility_ml_model.trained_model
                self.download_prediction_models()

                with start_action(
                    action_type="fetching_prediction_data",
                    facilityid=self.facilityid,
                    client=self.client,
                ):
                    # ------------Fetch the data from client SQL db and store in S3 buckets----------
                    result_dict = fetch_prediction_data(
                        client_sql_engine=self.client_sql_engine,
                        prediction_date=self.prediction_date,
                        facilityid=self.facilityid,
                        train_start_date=trained_model.train_start_date,
                        client=self.client,
                        cache_strategy=self.cache_strategy,
                        s3_location_path_prefix=self._get_s3_path(prediction_date=self.prediction_date),
                        local_folder=self.local_folder,
                        save_outputs_in_local=self.save_outputs_in_local,
                        save_outputs_in_s3=self.save_outputs_in_s3,
                        model_missing_datasets=trained_model.missing_datasets,
                        facility_missing_datasets=self.facility_ml_model.missing_datasets,
                    )

                final, result_dict = self.feature_engineering(result_dict=result_dict, prediction_date=self.prediction_date)

                with start_action(
                    action_type="load_model",
                    facilityid=self.facilityid,
                    modelid=trained_model.mlflow_model_id,
                    client=self.client,
                ):
                    with open(
                        f"/data/models/{trained_model.mlflow_model_id}/artifacts/{trained_model.mlflow_model_id}.pickle",
                        "rb",
                    ) as f:
                        model = pickle.load(f)

                # Filter records for the given prediction_date
                final = final[final.censusdate == self.prediction_date]
                num_null_patients = final.isnull().any(axis=1).sum()
                log_message(message_type='info', num_patients_with_null_values=num_null_patients)

                # --------------------Create X-frame and Identifier columns------------------------
#                     final, hospice_final = self.filter_hospice_patients(final)
                pd_final_df, pd_final_idens = self._prep(df=final, prediction_date=self.prediction_date)
                # ----------- Predict for the given input features & Rank those predictions -------------
                predictions_df = self.prediction(
                    model=model,
                    final_x=pd_final_df,
                    final_idens=pd_final_idens,
                    prediction_date=self.prediction_date,
                )
                # ----------- Insert Hospice & other excluded patients into daily_prediction table -------------
#                     self.insert_hospice_patients(hospice_final, self.prediction_date, facilityid)

                if not self.test:
                    with start_action(
                        action_type="generate_explanations",
                        facilityid=self.facilityid,
                        client=self.client,
                    ):
                        pd_final_df, pd_final_idens = self._filter_show_in_report_patients(
                            predictions_df=predictions_df,
                            pd_final_df=pd_final_df,
                            pd_final_idens=pd_final_idens,
                        )
                        generate_explanations(
                            model=model,
                            final_x=pd_final_df,
                            final_idens=pd_final_idens,
                            raw_data_dict=result_dict,
                            client=self.client,
                            s3_location_path_prefix=self._get_s3_path(prediction_date=self.prediction_date),
                            save_outputs_in_local=self.save_outputs_in_local,
                            local_folder=self.local_folder,
                            ml_model=self.facility_ml_model.ml_model,
                        )
        finally:
            # Since same instance can be used for other tasks, clean up dowloaded models
            if self.cleanup_downloaded_models:
                self.delete_downloaded_models()

    @instrumented
    def run_model(
        self,
        *,
        client,
        prediction_date,
        ml_model_org_config_id,
        facilityid,
        mlflow_model_id=None,
        test=False,
        replace_existing_predictions=False,
        test_group_percentage=1.0,
        cache_s3_folder="raw",
        cache_strategy=CHECK_CACHE_FIRST,
        save_outputs_in_s3=False,
        save_outputs_in_local=False,
        local_folder=None,
        cleanup_downloaded_models=True,
        disable_instrumentation=None,
    ):

        self.facility_ml_model = (
            saiva_api.organization_ml_model_configs.get_facility_ml_model(
                org_id=client,
                ml_model_org_config_id=ml_model_org_config_id,
                customers_facility_identifier=facilityid,
            )
        )

        # Override the trained model if mlflow_model_id is provided
        if mlflow_model_id:
            self.facility_ml_model.trained_model = saiva_api.trained_models.get_by_mlflow_model_id(mlflow_model_id=mlflow_model_id)

        self.org_ml_model_config = saiva_api.organization_ml_model_configs.get(
            org_id=client,
            ml_model_org_config_id=ml_model_org_config_id
        )
        self.client = client
        self.training_start_date = (
            None  # Use preconfigured training start date fetched from db
        )
        self.prediction_date = prediction_date
        self.facilityid = facilityid
        self.test = test
        self.replace_existing_predictions = replace_existing_predictions
        self.test_group_percentage = test_group_percentage
        self.cache_s3_folder = cache_s3_folder
        self.cache_strategy = cache_strategy
        self.save_outputs_in_s3 = save_outputs_in_s3
        self.save_outputs_in_local = save_outputs_in_local
        self.local_folder = local_folder
        self.cleanup_downloaded_models = cleanup_downloaded_models
        self.group_level = self.facility_ml_model.group_level

        log_message(
            message_type="info",
            client=client,
            facilityid=facilityid,
            ml_model_org_config_id=ml_model_org_config_id,
            prediction_date=prediction_date,
            test=test,
            replace_existing_predictions=replace_existing_predictions,
            trainset_start_date=self.training_start_date,
            test_group_percentage=test_group_percentage,
            cache_s3_folder=cache_s3_folder,
            cache_strategy=cache_strategy,
            save_outputs_in_s3=save_outputs_in_s3,
            save_outputs_in_local=save_outputs_in_local,
            local_folder=local_folder,
            group_level=self.group_level,
        )

        self.execute()


if __name__ == "__main__":
    prediction = RunPredictions()
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(prediction.run_model)
