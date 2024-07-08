from enum import Enum


class Strategy(str, Enum):
    SINGLE_CLIENT = "single_client"
    MULTIPLE_CLIENTS = "multiple_clients"
    LARGE_PLUS_SMALL_CLIENT = "large_plus_small_client"
