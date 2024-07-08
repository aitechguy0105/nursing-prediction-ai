import logging

from src.training_pipeline.strategies.strategy_multiple_clients.preprocess_data import PreprocessDataMultipleClients


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PreprocessDataLargePlusSmallClient(PreprocessDataMultipleClients):
    pass
