"""
Run Command :
python /src/run_model.py --client trio --facilityids '[42]' --prediction_start_date 2021-01-01 --prediction_end_date 2021-01-31 --s3-bucket saiva-dev-data-bucket --replace_existing_predictions True
python /src/run_model.py --client trio --facilityids '[7,42,52,55,265,186,21,1,194,273,274,275,276,277,278,279]' --prediction_start_date 2021-01-01 --prediction_end_date 2021-02-28 --s3-bucket saiva-dev-data-bucket --replace_existing_predictions True
"""

import sys

import fire
import pandas as pd
from eliot import start_action, start_task, to_file, log_message

from base_model import BasePredictions
from shared.constants import MODELS
from shared.load_raw_data import fetch_prediction_data

to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


class RunPredictions(BasePredictions):
    def execute(self):
        self._load_db_engines()
        modelid = 'CARD-MODEL-V1'

        with start_task(action_type='run_model', client=self.client):
            for facilityid in self.facilityids:

                with start_action(
                        action_type='fetching_prediction_data', facilityid=facilityid, client=self.client
                ):
                    # ------------Fetch the data from client SQL db and store in S3 buckets----------
                    result_dict = fetch_prediction_data(
                        client=self.client,
                        client_sql_engine=self.client_engine,
                        start_date=self.prediction_start_date,
                        end_date=self.prediction_end_date,
                        facilityid=facilityid,
                    )

                final_df = self.feature_engineering(facilityid, result_dict)
                final_df['modelid'] = modelid
                final_df['client'] = self.client
                final_df = final_df.rename(columns={'total_score': 'predictionvalue'})
                final_df.drop(
                    ['hosp_target_3_day_hosp','alert_score','admissions_proximity_score','readmissions_score','dx_sum'], 
                    axis=1, 
                    inplace = True
                ) 
                for prediction_date in pd.date_range(start=self.prediction_start_date,
                                                     end=self.prediction_end_date):
                    # Filter records for the given prediction_date
                    pd_final = final_df[final_df.censusdate == prediction_date.date()].copy()

                    if self.replace_existing_predictions:
                        log_message(
                            message_type='info',
                            message=f'Delete all old facility data for the given date: {prediction_date}'
                        )
                        self.saiva_engine.execute(
                            f"""delete from daily_predictions where censusdate = '{prediction_date}' 
                        and facilityid = '{facilityid}' and client = '{self.client}' and modelid = '{modelid}'"""
                        )

                    log_message(
                        message_type='info',
                        message=f'Save facility data for the given date: {prediction_date}, {self.client}, {facilityid}'
                    )
                    pd_final.to_sql(
                        'daily_predictions',
                        self.saiva_engine,
                        method='multi',
                        if_exists='append',
                        index=False
                    )

    def run_model(
            self,
            client,
            prediction_start_date,
            prediction_end_date,
            s3_bucket,
            facilityids=None,
            test=False,
            replace_existing_predictions=False,
    ):

        self.client = client
        # Use preconfigured training start date
        self.training_start_date = MODELS[client]['training_start_date']
        self.prediction_start_date = prediction_start_date
        self.prediction_end_date = prediction_end_date
        self.facilityids = facilityids
        self.test = test
        self.s3_bucket = s3_bucket
        self.replace_existing_predictions = replace_existing_predictions

        log_message(
            message_type='info', client=client, prediction_end_date=prediction_end_date,
            prediction_start_date=prediction_start_date, facilityids=facilityids, test=test,
            replace_existing_predictions=replace_existing_predictions,
            trainset_start_date=self.training_start_date
        )

        self.execute()


if __name__ == '__main__':
    prediction = RunPredictions()
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(prediction.run_model)
