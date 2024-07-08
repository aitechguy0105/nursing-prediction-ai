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
import typing

import fire
import pandas as pd
from eliot import start_action, start_task, to_file, log_message
from omegaconf import OmegaConf

from saiva.model.base_model import Inference
from saiva.model.shared.constants import CHECK_CACHE_FIRST, saiva_api
from saiva.model.shared.load_raw_data import fetch_backfill_data
from saiva.training.core import BaseModel  # noqa: F401 (pickle)


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


class RunBackfillInference(Inference):
    def __init__(
            self,
            prediction_start_date: str,
            **kwargs: typing.Dict[str, typing.Any],
    ):
        super().__init__(**kwargs)
        self.prediction_start_date = prediction_start_date

    def fetch_prediction_data(self):
        with start_action(action_type='fetching_prediction_data', facilityid=self.facility_id, client=self.client):
            # ------------Fetch the data from client SQL db and store in S3 buckets----------
            result_dict = fetch_backfill_data(
                client=self.client,
                client_sql_engine=self.client_sql_engine,
                train_start_date=self.trained_model.train_start_date,
                prediction_start_date=self.prediction_start_date,
                prediction_end_date=self.prediction_date,
                facilityid=self.facility_id,
                config=self.config,
            )
        return result_dict

    def execute(self):
        with start_task(action_type='run_model', client=self.client):
            result_dict = self.fetch_prediction_data()

            final, result_dict = self.feature_engineering(
                prediction_date=self.prediction_date,
                result_dict=result_dict,
            )

            # Loop through the given date range and do the prediction per date
            for current_prediction_date in pd.date_range(
                start=self.prediction_start_date,
                end=self.prediction_date
            ):
                log_message(
                    message_type="info",
                    message=(
                        f"***************** "
                        f"Run Prediction for facility: {self.facilityid}, {current_prediction_date.date()} "
                        f"*****************")
                )

                log_message(
                    message_type='info',
                    message=(
                        f"***************** "
                        f"Overall shape: {final.shape} "
                        f"*****************"
                    )
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
                pd_final_df, pd_final_idens = self._prep(df=pd_final, prediction_date=current_prediction_date)
                # ----------- Predict for the given input features & Rank those predictions -------------
                self.prediction(
                    model=self.model,
                    final_x=pd_final_df,
                    final_idens=pd_final_idens,
                    prediction_date=current_prediction_date
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
    ):
        self.execute()


if __name__ == '__main__':
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(RunBackfillInference)
