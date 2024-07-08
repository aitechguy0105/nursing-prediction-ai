"""
Run Command :
python /src/run_model.py --client infinity-benchmark --facilityids '[37]' --prediction-date 2020-05-02
--s3-bucket saiva-dev-data-bucket --replace_existing_predictions True
--save-outputs-in-local True --local-folder /data/test
--sub_clients : Comma seperated sub-client names for the same data feed
--group_level : Indicate whether the ranking will be done on facility/unit/floor
"""

from eliot import start_task, log_message

from shared.data_manager import DataManager
from shared.database import DbEngine


def handle_delete_error(func, path, exc_info):
    """
    handler if you get exceptions when deleting model files from local disk
    """
    log_message(message_type='warning', file_not_deleted=path, exception=exc_info)


class BasePredictions(object):
    def __init__(self):
        self.client = None
        self.prediction_start_date = None
        self.prediction_end_date = None
        self.facilityids = None
        self.test = False
        self.replace_existing_predictions = False
        self.saiva_engine = None
        self.client_engine = None
        self.s3_bucket = None

    def _run_asserts(self):
        pass

    def _load_db_engines(self):
        engine = DbEngine()
        self.saiva_engine = engine.get_postgresdb_engine()
        self.client_engine = engine.get_sqldb_engine(clientdb_name=self.client)

    def feature_engineering(self, facilityid, result_dict):
        with start_task(action_type='feature-engineering', client=self.client, facilityid=facilityid):
            dm = DataManager(
                result_dict=result_dict,
                facilityid=facilityid,
                client=self.client,
                start_date=self.prediction_start_date,
                end_date=self.prediction_end_date,
                s3_bucket=self.s3_bucket,
            )
            alerts_df, admissions_df, diagnosis_df, rehosp_df = dm.get_features()
            final_df = dm.merge_features(
                alerts_df,
                admissions_df,
                diagnosis_df,
                rehosp_df
            )
            score_df = dm.generate_total_score(final_df)
            result_df = dm.generate_ranks(score_df)

            return result_df
