import datetime
import logging
import typing

from omegaconf import OmegaConf

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.saiva.training.utils import load_config
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseStrategy:
    def __init__(self, current_step: TrainingStep) -> None:
        self.CURRENT_STEP = current_step

    def get_dataset_provider(
        self,
        *,
        run_id: str,
        force_regenerate: typing.Optional[bool] = False,
    ) -> DatasetProvider:
        dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

        dataset_provider.download_config(step=TrainingStep.previous_step(self.CURRENT_STEP))

        return dataset_provider

    def get_dataset_provider_and_config(
        self,
        *,
        run_id: str,
        force_regenerate: typing.Optional[bool] = False,
    ) -> typing.Tuple[DatasetProvider, OmegaConf]:
        dataset_provider = self.get_dataset_provider(run_id=run_id, force_regenerate=force_regenerate)

        config = load_config(LOCAL_TRAINING_CONFIG_PATH)

        return dataset_provider, config
    
    def upload_config(
        self,
        *,
        dataset_provider: DatasetProvider
    ):
        dataset_provider.store_config(step=self.CURRENT_STEP)
