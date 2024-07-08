import io
import json
import logging
import pickle
import subprocess
import typing
from enum import Enum
from pathlib import Path

import boto3
import pandas as pd

from src.saiva.model.shared.constants import ENV, LOCAL_TRAINING_CONFIG_PATH
from src.training_pipeline.shared.constants import AUTOMATIC_TRAINING_S3_BUCKET, DATACARDS_S3_BUCKET


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrainingStep(str, Enum):
    SETUP = "setup"
    CONFIGURE_FACILITIES = "configure_facilities"
    CALCULATE_DATE_RANGE = "calculate_date_range"
    FETCH_DATA = "fetch_data"
    PREPROCESS_DATA = "preprocess_data"
    DATACARD_FACILITY_DISCOVERY = "datacard_facility_discovery"
    DATACARD_DATA_AVAILABILITY = "datacard_data_availability"
    MERGE_DATA = "merge_data"
    CHECK_AMOUNT_OF_DATA = "check_amount_of_data"
    DATES_CALCULATION = "dates_calculation"
    FEATURE_ENGINEERING = "feature_engineering"
    POST_FEATURE_ENGINEERING_MERGE = "post_feature_engineering_merge"
    FEATURE_SELECTION = "feature_selection"
    DATASETS_GENERATION = "datasets_generation"
    DATACARD_X_AED_Y_CHECK = "datacard_x_aed_y_check"
    DATA_DISTRIBUTION = "data_distribution"
    TRAIN_MODEL = "train_model"
    DATACARD_PREDICTION_PROBABILITY = "datacard_prediction_probabilty"
    DATACARD_SHAP_VALUES = "datacard_shap_values"
    DATACARD_DECISIONS = "datacard_decisions"
    DATACARD_TRAINED_MODEL = "datacard_trained_model"
    UPLOAD_MODEL_METADATA = "upload_model_metadata"

    @classmethod
    def step_order(cls):
        return {
            cls.SETUP: 0,
            cls.CONFIGURE_FACILITIES: 1,
            cls.CALCULATE_DATE_RANGE: 2,
            cls.FETCH_DATA: 3,
            cls.PREPROCESS_DATA: 4,
            cls.DATACARD_FACILITY_DISCOVERY: 5,
            cls.DATACARD_DATA_AVAILABILITY: 5,
            cls.MERGE_DATA: 6,
            cls.CHECK_AMOUNT_OF_DATA: 7,
            cls.DATES_CALCULATION: 8,
            cls.FEATURE_ENGINEERING: 9,
            cls.POST_FEATURE_ENGINEERING_MERGE: 10,
            cls.FEATURE_SELECTION: 11,
            cls.DATASETS_GENERATION: 12,
            cls.DATACARD_X_AED_Y_CHECK: 13,
            cls.DATA_DISTRIBUTION: 13,
            cls.TRAIN_MODEL: 14,
            cls.DATACARD_PREDICTION_PROBABILITY: 15,
            cls.DATACARD_SHAP_VALUES: 15,
            cls.DATACARD_DECISIONS: 15,
            cls.DATACARD_TRAINED_MODEL: 15,
            cls.UPLOAD_MODEL_METADATA: 16,
        }

    @classmethod
    def previous_step(cls, current_step):
        step_order = TrainingStep.step_order()
        current_step_index = step_order[current_step]
        if current_step_index == 0:
            return current_step
        else:
            previous_step_index = current_step_index - 1
            previous_step = None

            while previous_step is None:
                # Find the key for the previous step, if there are multiple steps with the same index (e.g. datacard steps) then return the step before that
                previous_steps = []
                
                for step, index in step_order.items():
                    if index == previous_step_index:
                        previous_steps.append(step)

                if len(previous_steps) == 1:
                    previous_step = previous_steps[0]
                else:
                    previous_step_index -= 1

            return previous_step


class DatasetProvider:
    def __init__(self, *, run_id: str, force_regenerate: typing.Optional[bool] = False) -> None:
        self.run_id = run_id
        self.file_format = "parquet"
        self.client = boto3.client('s3')
        self.force_regenerate = force_regenerate

    def make_key(self, *, step: TrainingStep, dataset_name: str, file_format: typing.Optional[str] = None) -> str:
        return str(Path(self.run_id) / step.value / dataset_name) + (f".{file_format}" if file_format else '')

    def make_dataset_filepath(self, *, step: TrainingStep, dataset_name: str, file_format: typing.Optional[str] = None) -> str:
        return 's3://' + str(Path(AUTOMATIC_TRAINING_S3_BUCKET) / self.make_key(step=step, dataset_name=dataset_name, file_format=file_format))

    def get(self, *, dataset_name: str, step: TrainingStep, exact_step: bool = False) -> pd.DataFrame:
        try:
            buffer = self.download_from_s3(filename=dataset_name, step=step, file_format=self.file_format)
            df = pd.read_parquet(buffer)
            return df
        except:
            return pd.DataFrame()

    def set(self, *, dataset_name: str, step: TrainingStep, df: pd.DataFrame, index: typing.Optional[bool] = None):
        buffer = io.BytesIO()
        bytes_data = df.to_parquet(buffer)

        self.upload_to_s3(filename=dataset_name, step=step, buffer=buffer, file_format=self.file_format)

    def store_df_csv(self, *, dataset_name: str, step: TrainingStep, df: pd.DataFrame, index: typing.Optional[bool] = None):
        buffer = io.BytesIO()
        bytes_data = df.to_csv(buffer)

        self.upload_to_s3(filename=dataset_name, step=step, buffer=buffer, file_format='csv')

    def does_file_exist(
        self,
        *,
        filename: str, 
        step: TrainingStep,
        file_format: typing.Optional[str] = None,
        ignore_force_regenerate: typing.Optional[bool] = False,
    ) -> bool:
        if not ignore_force_regenerate and self.force_regenerate:
            log.info(f"Force regenerate is set, skipping file existence check for {filename}")
            return False

        filepath = self.make_key(
            step=step,
            dataset_name=filename,
            file_format=file_format if file_format else self.file_format
        )

        res = self.client.list_objects_v2(Bucket=AUTOMATIC_TRAINING_S3_BUCKET, Prefix=str(filepath), MaxKeys=1)
        return 'Contents' in res
    
    def make_datacard_prefix(self) -> str:
        return str(Path("batch_datacards") / ENV / self.run_id)

    def make_datacard_s3_path(self) -> str:
        return f"s3://{DATACARDS_S3_BUCKET}/{self.make_datacard_prefix()}"

    def does_datacard_exist(
        self,
        *,
        step: TrainingStep,
        client: typing.Optional[str] = '',
    ) -> bool:

        filepath = self.make_datacard_prefix()

        res = self.client.list_objects_v2(Bucket=DATACARDS_S3_BUCKET, Prefix=filepath)

        for obj in res.get('Contents', []):
            file = obj['Key'].split('/')[-1]

            datacard_step = step.value.replace('datacard_', '')

            if 'report' in file and datacard_step in file and client in file:
                return True

        return False

    def upload_to_s3(self, *, filename: str, step: TrainingStep, file_format: str, buffer: io.BytesIO):
        buffer.seek(0)  # rewind pointer back to start
        
        key = self.make_key(
            step=step,
            dataset_name=filename,
            file_format=file_format
        )

        self.client.upload_fileobj(
            Fileobj=buffer,
            Bucket=AUTOMATIC_TRAINING_S3_BUCKET,
            Key=key
        )

    def download_from_s3(self, *, filename: str, step: TrainingStep, file_format: str) -> io.BytesIO:
        key = self.make_key(
            step=step,
            dataset_name=filename,
            file_format=file_format
        )

        try:
            buffer = io.BytesIO()
            self.client.download_fileobj(
                    Bucket=AUTOMATIC_TRAINING_S3_BUCKET,
                    Key=key,
                    Fileobj=buffer
                )
            buffer.seek(0)
            return buffer
        except Exception as e:
            log.error(e)
            raise Exception(f"Could not load file {key}")

    def store_json(self, *, filename: str, step: TrainingStep, data: object):
        buffer = io.BytesIO()
        buffer.write(json.dumps(data, default=str).encode('utf-8'))
        self.upload_to_s3(filename=filename, step=step, buffer=buffer, file_format='json')

    def load_json(self, *, filename: str, step: TrainingStep) -> typing.Union[dict, list]:
        return json.load(self.download_from_s3(filename=filename, step=step, file_format='json'))

    def store_pickle(self, *, filename: str, step: TrainingStep, data: object, protocol: int = 4):
        buffer = io.BytesIO()
        buffer.write(pickle.dumps(data, protocol=protocol))
        self.upload_to_s3(filename=filename, step=step, buffer=buffer, file_format='pickle')

    def load_pickle(self, *, filename: str, step: TrainingStep) -> typing.Union[dict, list]:
        buffer = self.download_from_s3(filename=filename, step=step, file_format='pickle')
        return pickle.loads(buffer.read())

    def store_txt(self, *, filename: str, step: TrainingStep, data: str):
        buffer = io.BytesIO()
        buffer.write(data.encode('utf-8'))
        self.upload_to_s3(filename=filename, step=step, buffer=buffer, file_format='txt')

    def load_txt(self, *, filename: str, step: TrainingStep) -> object:
        return self.download_from_s3(filename=filename, step=step, file_format='txt').read()

    def store_file(self, *, filename: str, step: TrainingStep, buffer: io.BytesIO, file_format: str):
        self.upload_to_s3(filename=filename, step=step, buffer=buffer, file_format=file_format)
        
    def generate_config_path(self, *, step: TrainingStep, prefix: str = ''):
        return 's3://' + str(Path(AUTOMATIC_TRAINING_S3_BUCKET) / (Path(self.run_id) / step.value)) + f"{prefix}/conf/training"

    def download_config(self, *, step: TrainingStep, prefix: str = ''):
        output = subprocess.run(
            f"aws s3 sync {self.generate_config_path(step=step, prefix=prefix)} {LOCAL_TRAINING_CONFIG_PATH}",
            shell=True,
            capture_output=True
        )
        log.info(output)

        if output.returncode != 0:
            raise Exception("Could not download config")

    def store_config(self, *, step: TrainingStep, prefix: str = ''):
        output = subprocess.run(
            f"aws s3 sync {LOCAL_TRAINING_CONFIG_PATH} {self.generate_config_path(step=step, prefix=prefix)}",
            shell=True,
            capture_output=True
        )
        log.info(output)

        if output.returncode != 0:
            raise Exception("Could not upload config")

    def get_datacards(self) -> typing.List[str]:
        res = self.client.list_objects_v2(Bucket=DATACARDS_S3_BUCKET, Prefix=self.make_datacard_prefix())

        datacards = {}

        datacards_save_metadata = [
            'facility_discovery',
            'data_availability',
            'xaedy',
            'prediction_probability'
        ]

        for obj in res.get('Contents', []):
            file = obj['Key'].split('/')[-1]

            file_name, file_format = file.split('.')

            file_name_parts = file_name.split('_')

            client = file_name_parts[1]
            ts = file_name_parts[-1]
            datacard_type = "_".join(file_name_parts[4:-2])

            datacard = datacards.get(datacard_type, {})
            client_datacard = datacard.get(client, {})
            
            if file_format in client_datacard:
                existing_file_ts = client_datacard[file_format].split('.')[0].split('_')[-1]
                if existing_file_ts > ts:
                    continue
    
            client_datacard[file_format] = f"s3://{DATACARDS_S3_BUCKET}/{obj['Key']}"
            
            if ('+' not in client) and ('metadata' not in client_datacard) and (datacard_type in datacards_save_metadata):
                client_datacard['metadata'] = {
                    'org_id': client
                }

            datacard[client] = client_datacard
            datacards[datacard_type] = datacard
                
        datacards = {datacard_type: list(datacard.values()) for datacard_type, datacard in datacards.items()}

        return datacards
