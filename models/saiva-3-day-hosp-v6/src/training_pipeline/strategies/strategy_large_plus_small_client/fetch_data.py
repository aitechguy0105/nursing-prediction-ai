import logging

from src.training_pipeline.strategies.strategy_multiple_clients.fetch_data import FetchDataMultipleClients


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FetchDataLargePlusSmallClient(FetchDataMultipleClients):
    pass
