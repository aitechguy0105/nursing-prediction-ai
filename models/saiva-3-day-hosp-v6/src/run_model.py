"""
Run Command :
python /src/run_model.py run_model
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

import sys
import typing
import os

import fire
from eliot import start_task, start_action, to_file, log_message

from saiva.model.base_model import Inference
from saiva.decorators import instrumented
from saiva.training.core import BaseModel  # noqa: F401 (pickle)

to_file(sys.stdout)  # ECS containers log stdout to CloudWatch
to_file(open("src/run_model.log", "w"))

class RunInference(Inference):
    def execute(self):
        try:
            with start_task(action_type="run_model executing ", client=self.client, ml_model_org_config_id=self.ml_model_org_config_id,
            prediction_date=self.prediction_date, facilityid=self.facility_id):
                
                with start_action(action_type="fetch_prediction_data."):
                    
                    result_dict = self.fetch_prediction_data()
                    for key in result_dict.keys():
                        log_message(
                            message_type='info', 
                            message=f"Prediction data for {key}", dataframe_shape = result_dict[key].shape
                        )
                
                with start_action(action_type="feature engineering process"):
                    final, result_dict = self.feature_engineering(
                        result_dict=result_dict,
                        prediction_date=self.prediction_date
                    )
                    log_message(
                        message_type='info', 
                        message=f"final df shape after feature generation & merging.",
                        final_shape= final.shape
                    )

                # Filter records for the given prediction_date
                with start_action(action_type="selecting final data"):
                    final = self.select_prediction_census(final)
                    num_null_patients = final.isnull().any(axis=1).sum()
                    log_message(message_type='info', num_patients_with_null_values=num_null_patients)

                # --------------------Create X-frame and Identifier columns------------------------
                with start_action(action_type="creating prediction dataset"):
                    if self.model_type == 'model_upt':
                        final, hospice_final = self.filter_hospice_patients(df=final)

                    pd_final_df, pd_final_idens = self._prep(df=final, prediction_date=self.prediction_date)

                # store the final_df and idens into s3 for future use
                s3_base_path = self._get_s3_path(prediction_date=self.prediction_date) 
                pd_final_df.to_parquet(s3_base_path + '/final_x.parquet', index=False)
                pd_final_idens.to_parquet(s3_base_path + '/final_idens.parquet', index=False)

                # ----------- Predict for the given input features & Rank those predictions -------------
                with start_action(action_type="running predictions", final_x_shape = pd_final_df.shape, 
                                 final_idens_shape = pd_final_idens.shape,
                                 prediction_date = self.prediction_date):
                    predictions_df = self.prediction(
                        final_x=pd_final_df,
                        final_idens=pd_final_idens,
                        prediction_date=self.prediction_date,
                    )

                # ----------- Insert Hospice & other excluded patients into daily_prediction table -------------
                if self.model_type == 'model_upt':
                    with start_action(action_type="inserting hospice and other excluded residents into daily_precition table", 
                                      hospice_df_shape = hospice_final.shape,
                                      prediction_date = self.prediction_date):
                        self.insert_hospice_patients(
                            hospice_df=hospice_final,
                            prediction_date=self.prediction_date
                        )
                
                if not self.test:
                    with start_action(action_type="executing generate_explanations function", 
                                      predictions_df=predictions_df.shape,
                                      final_df_shape=pd_final_df.shape,
                                      final_idens_shape=pd_final_idens.shape):
                        self.generate_explanations(
                            predictions_df=predictions_df,
                            final_df=pd_final_df,
                            final_idens=pd_final_idens,
                            result_dict=result_dict,
                        )
        finally:
            # Since same instance can be used for other tasks, clean up dowloaded models
            if self.cleanup_downloaded_models:
                self.delete_downloaded_models()

    @instrumented
    def run_model(self, disable_instrumentation: typing.Optional[bool] = None):
        log_message(
            message_type='info',
            client=self.client, ml_model_org_config_id=self.ml_model_org_config_id,
            prediction_date=self.prediction_date, facilityid=self.facility_id, test=self.test,
            replace_existing_predictions=self.replace_existing_predictions,
            trainset_start_date=self.trained_model.train_start_date, test_group_percentage=self.test_group_percentage,
            cache_s3_folder=self.cache_s3_folder, cache_strategy=self.cache_strategy,
            save_outputs_in_s3=self.save_outputs_in_s3, save_outputs_in_local=self.save_outputs_in_local,
            local_folder=self.local_folder
        )

        self.execute()


if __name__ == '__main__':
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(RunInference)
