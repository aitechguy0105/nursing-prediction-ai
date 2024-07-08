import logging

from src.training_pipeline.strategies.strategy_multiple_clients.configure_facilities import ConfigureFacilitiesMultipleClients


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ConfigureFacilitiesLargePlusSmallClient(ConfigureFacilitiesMultipleClients):
    pass
