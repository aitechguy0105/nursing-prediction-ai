"""
Run Command :
python /src/run_load_data.py run_load_data
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

import fire
from eliot import to_file, log_message

from saiva.model.base_model import Inference
from saiva.decorators import instrumented
from saiva.training.core import BaseModel  # noqa: F401 (pickle)

to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


class RunLoadData(Inference):

    def execute(self):
        try:
            self.fetch_prediction_data()
        finally:
            # Since same instance can be used for other tasks, clean up dowloaded models
            if self.cleanup_downloaded_models:
                self.delete_downloaded_models()

    @instrumented
    def run_load_data(self, disable_instrumentation: typing.Optional[bool] = None):
        log_message(
            message_type='info', client=self.client, ml_model_org_config_id=self.ml_model_org_config_id,
            prediction_date=self.prediction_date, facilityid=self.facility_id,
            replace_existing_predictions=self.replace_existing_predictions,
            cache_s3_folder=self.cache_s3_folder,
            save_outputs_in_local=self.save_outputs_in_local, local_folder=self.local_folder
        )

        self.execute()


if __name__ == '__main__':
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(RunLoadData)
