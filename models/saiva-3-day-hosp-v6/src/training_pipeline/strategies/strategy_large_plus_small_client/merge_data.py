import logging

from src.training_pipeline.strategies.strategy_multiple_clients.merge_data import MergeDataMultipleClients


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MergeDataLargePlusSmallClient(MergeDataMultipleClients):
    pass
