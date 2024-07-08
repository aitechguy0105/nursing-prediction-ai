"""
Run Command :
python /src/run_model.py --client infinity-benchmark --facilityids '[37]' --prediction-date 2020-05-02
--s3-bucket saiva-dev-data-bucket --replace_existing_predictions True
--save-outputs-in-local True --local-folder /data/test
--sub_clients : Comma seperated sub-client names for the same data feed
--group_level : Indicate whether the ranking will be done on facility/unit/floor
"""

import pickle
import sys

import fire
import pandas as pd
from eliot import start_action, start_task, to_file, log_message

from base_model import BasePredictions
# from run_explanation import generate_explanations
from shared.constants import CHECK_CACHE_FIRST
from shared.constants import saiva_api
from shared.load_raw_data import fetch_backfill_data
from data_models import BaseModel # noqa: F401

to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


class RunPredictions(BasePredictions):
    def __init__(self):
        self.prediction_start_date = None
        super().__init__()

    def execute(self):
        self._run_asserts()
        self._create_local_directory()
        self._load_db_engines()

        with start_task(action_type='run_model', client=self.client):
            trained_model = self.facility_ml_model.trained_model
            self.download_prediction_models()

            with start_action(
                    action_type='fetching_prediction_data', facilityid=facilityid, client=self.client
            ):
                # ------------Fetch the data from client SQL db and store in S3 buckets----------
                result_dict = fetch_backfill_data(
                    client=self.client,
                    client_sql_engine=self.client_sql_engine,
                    train_start_date=trained_model.train_start_date,
                    prediction_start_date=self.prediction_start_date,
                    prediction_end_date=self.prediction_date,
                    facilityid=self.facilityid,
                    model_missing_datasets=self.org_ml_model_config.missing_datasets,
                    facility_missing_datasets=self.facility_ml_model.missing_datasets,
                )

            final, result_dict = self.feature_engineering(
                prediction_date=self.prediction_date,
                result_dict=result_dict,
            )

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

            # Loop through the given date range and do the prediction per date
            for current_prediction_date in pd.date_range(start=self.prediction_start_date,
                                                            end=self.prediction_date):
                log_message(
                    message_type='info',
                    message=f'***************** Run Prediction for facility: {facilityid}, {current_prediction_date.date()} *****************'
                )

                log_message(
                    message_type='info',
                    message=f'***************** Overall shape: {final.shape} *****************'
                )

                # Filter records for the given prediction_date
                pd_final = final[
                    final['censusdate'].dt.strftime('%Y-%m-%d') == current_prediction_date.date().strftime("%Y-%m-%d")
                    ].copy()
                log_message(
                    message_type='info',
                    message=f'***************** Filtered shape: {pd_final.shape}  *****************'
                )
                num_null_patients = pd_final.isnull().any(axis=1).sum()
                log_message(message_type='info', num_patients_with_null_values=num_null_patients)

                # --------------------Create X-frame and Identifier columns------------------------
                pd_final, hospice_final = self.filter_hospice_patients(df=pd_final)
                pd_final_df, pd_final_idens = self._prep(df=final, prediction_date=current_prediction_date)
                # ----------- Predict for the given input features & Rank those predictions -------------
                predictions_df = self.prediction(
                    model=model,
                    final_x=pd_final_df,
                    final_idens=pd_final_idens,
                    prediction_date=current_prediction_date,
                )
                # ----------- Insert Hospice & other excluded patients into daily_prediction table -------------
                self.insert_hospice_patients(
                    hospice_df=hospice_final,
                    prediction_date=current_prediction_date
                )

                # TODO: Explanations take a lot of time, hence commenting it for time being
                # if not self.test:
                #     with start_action(action_type='generate_explanations', facilityid=facilityid,
                #                       client=self.client):
                #         generate_explanations(
                #             current_prediction_date,
                #             model,
                #             pd_final_df,
                #             pd_final_idens,
                #             result_dict,
                #             self.client,
                #             facilityid,
                #             self.saiva_engine,
                #             self.save_outputs_in_s3,
                #             self._get_s3_path(facilityid),
                #             self.save_outputs_in_local,
                #             self.local_folder
                #         )

            # Since same instance can be used for other tasks, clean up dowloaded models
            if self.cleanup_downloaded_models:
                self.delete_downloaded_models()

    def run_model(
            self,
            client,
            s3_bucket,
            prediction_start_date,
            prediction_end_date,
            facilityids=None,
            sub_clients=None,
            group_level='facility',
            test=False,
            replace_existing_predictions=False,
            test_group_percentage=1.0,
            cache_s3_folder='raw',
            cache_strategy=CHECK_CACHE_FIRST,
            save_outputs_in_s3=False,
            save_outputs_in_local=False,
            local_folder=None,
            cleanup_downloaded_models=True):

        self.facility_ml_model = (
            saiva_api.organization_ml_model_configs.get_facility_ml_model(
                org_id=client,
                ml_model_org_config_id=ml_model_org_config_id,
                customers_facility_identifier=facilityid,
            )
        )
        self.org_ml_model_config = saiva_api.organization_ml_model_configs.get(
            org_id=client,
            ml_model_org_config_id=ml_model_org_config_id
        )

        self.client = client
        self.training_start_date = None  # Use preconfigured training start date fetched from db
        self.s3_bucket = s3_bucket
        self.prediction_start_date = prediction_start_date
        self.prediction_date = prediction_end_date
        self.facilityids = facilityids
        self.test = test
        self.replace_existing_predictions = replace_existing_predictions
        self.test_group_percentage = test_group_percentage
        self.cache_s3_folder = cache_s3_folder
        self.cache_strategy = cache_strategy
        self.save_outputs_in_s3 = save_outputs_in_s3
        self.save_outputs_in_local = save_outputs_in_local
        self.local_folder = local_folder
        self.sub_clients = sub_clients.split(',') if sub_clients else []
        self.group_level = group_level
        self.cleanup_downloaded_models = cleanup_downloaded_models

        log_message(
            message_type='info', client=client, s3_bucket=s3_bucket, prediction_start_date=prediction_start_date,
            prediction_end_date=prediction_end_date, facilityids=facilityids, test=test,
            replace_existing_predictions=replace_existing_predictions,
            trainset_start_date=self.training_start_date, test_group_percentage=test_group_percentage,
            cache_s3_folder=cache_s3_folder, cache_strategy=cache_strategy,
            save_outputs_in_s3=save_outputs_in_s3,
            save_outputs_in_local=save_outputs_in_local, local_folder=local_folder
        )

        self.execute()


if __name__ == '__main__':
    prediction = RunPredictions()
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(prediction.run_model)
