import logging
import typing

import pandas as pd
from omegaconf import OmegaConf

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from src.training_pipeline.strategies.common.base import BaseStrategy


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.MERGE_DATA


class MergeDataBase(BaseStrategy):
    def __init__(self) -> None:
        super().__init__(current_step=CURRENT_STEP)
        self.DATASETS_TO_LOAD = []

    def load_data(
        self,
        *,
        dataset_provider: DatasetProvider,
        client: str,
        feature_group: str,
    ) -> typing.Tuple[typing.Optional[pd.DataFrame], bool]:
        if dataset_provider.does_file_exist(filename=feature_group, step=CURRENT_STEP):
            log.info(f"Dataset {feature_group} already exists. Skipping merging step")
            return None, True
        
        dataset_name = f"{client}_{feature_group}"

        if not dataset_provider.does_file_exist(filename=dataset_name, step=TrainingStep.PREPROCESS_DATA, ignore_force_regenerate=True):
            log.info(f"Dataset {dataset_name} does not exist.")
            return None, False

        log.info(f"Loading dataset {dataset_name}.")

        df = dataset_provider.get(dataset_name=dataset_name, step=TrainingStep.PREPROCESS_DATA)

        df['masterpatientid'] = df['masterpatientid'].apply(lambda x: client + '_' + str(x))
        df['client'] = client

        return df, False

    def upload_config(
        self,
        *,
        dataset_provider: DatasetProvider,
        missing_datasets: typing.List[str],
        config: OmegaConf
    ):
        training_metadata = config.training_config.training_metadata
        training_metadata['missing_datasets'] = list(missing_datasets)

        conf = OmegaConf.create({'training_config': {'training_metadata': training_metadata}})
        OmegaConf.save(conf, f'{LOCAL_TRAINING_CONFIG_PATH}generated/training_metadata.yaml')
        
        super().upload_config(dataset_provider=dataset_provider)
