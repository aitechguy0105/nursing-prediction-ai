import boto3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import os 
import sys

from collections import defaultdict
from multiprocessing import Pool 

class BagOfOrders: 
    def __init__(self): 
        pass
    
    def featurize(self, prediction_times, pids, fids, orders): 
        pass