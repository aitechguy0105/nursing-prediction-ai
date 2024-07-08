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
from explanations import generate_explanations
from shared.constants import CHECK_CACHE_FIRST
from shared.constants import MODELS
from shared.load_raw_data import fetch_backfill_data
from data_models import BaseModel

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
            for facilityid in self.facilityids:
                self.modelid = MODELS[self.client][facilityid]
                self.download_prediction_models(facilityid)

                with start_action(
                        action_type='fetching_prediction_data', facilityid=facilityid, client=self.client
                ):
                    # ------------Fetch the data from client SQL db and store in S3 buckets----------
                    result_dict = fetch_backfill_data(
                        client=self.client,
                        client_sql_engine=self.client_engine,
                        train_start_date=self.training_start_date,
                        prediction_start_date=self.prediction_start_date,
                        prediction_end_date=self.prediction_date,
                        facilityid=facilityid
                    )

                final, result_dict = self.feature_engineering(
                    facilityid=facilityid,
                    result_dict=result_dict,
                    prediction_start_date=self.prediction_start_date
                )

                with start_action(action_type='load_model', facilityid=facilityid, modelid=self.modelid,
                                  client=self.client):
                    with open(f'/data/models/{self.modelid}/artifacts/{self.modelid}.pickle', 'rb') as f:
                        model = pickle.load(f)

                # Loop through the given date range and do the prediction per date
                for current_prediction_date in pd.date_range(start=self.prediction_start_date,
                                                             end=self.prediction_date):
                    log_message(
                        message_type='info', 
                        message=f'***************** Run Prediction for facility: {facilityid}, {current_prediction_date.date()} *****************'
                    )
                    
                    # Filter records for the given prediction_date
                    pd_final = final[final.censusdate == current_prediction_date.date()].copy()
                    pd_final = pd_final.dropna()
                    # --------------------Create X-frame and Identifier columns------------------------
                    pd_final_df, pd_final_idens = self._prep(pd_final, facilityid)
                    # ----------- Predict for the given input features & Rank those predictions -------------
                    self.prediction(model, pd_final_df, pd_final_idens, facilityid, current_prediction_date)

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
            local_folder=None):

        self.client = client
        # Use preconfigured training start date
        self.training_start_date = MODELS[client]['training_start_date']
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
