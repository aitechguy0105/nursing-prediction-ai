from collections import defaultdict
import os

from saiva.model.shared.constants import ENV

MIN_CENSUS = 900000  # based on Trio
MAX_CENSUS = 1868519  # based on Marquis
AVERAGE_CENSUS = int((1070586 + MAX_CENSUS) / 2)  # (1469553) based on average of Trio and Marquis
AUTOMATIC_TRAINING_S3_BUCKET = os.environ.get('AUTOMATIC_TRAINING_S3_BUCKET', f"saiva-{ENV}-automated-training")
DATACARDS_S3_BUCKET = os.environ.get('DATACARDS_S3_BUCKET', f"saiva-datacards")