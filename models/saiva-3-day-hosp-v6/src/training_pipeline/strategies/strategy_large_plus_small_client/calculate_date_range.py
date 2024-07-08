import datetime
import logging
import typing
from dateutil.relativedelta import relativedelta

from omegaconf import OmegaConf
import pandas as pd

from src.training_pipeline.shared.helpers import DatasetProvider
from src.training_pipeline.shared.models import ClientConfiguration
from src.training_pipeline.strategies.strategy_multiple_clients.calculate_date_range import CalculateDateRangeMultipleClients


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CalculateDateRangeLargePlusSmallClient(CalculateDateRangeMultipleClients):
    def __init__(self) -> None:
        super().__init__(invalid_action_types=None)
    
    def calculate_train_start_date(
        self,
        *,
        dataset_provider: DatasetProvider,
        config: OmegaConf,
        client_configurations: typing.List[ClientConfiguration],
        test_end_date: datetime.date,
    ) -> datetime.date:
    
        train_start_date = test_end_date - relativedelta(months=30)

        return train_start_date
