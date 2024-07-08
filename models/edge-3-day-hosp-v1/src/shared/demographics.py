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

class BasicDemographicsFeatures: 
    def __init__(self): 
        pass

    def featurize(self, prediction_times, demo_data): 
        """
        Encode demographic data.  Available columns are: 
        DOB -> age but DOB missing for some... 
        Gender - clean
        Education - messy; ignore for now. 
        Citizenship - almost all US; ignore
        Race - messy; ignore for now. 
        Religion - very messy; ignore! 
        State - long tail; ignore; plenty of blanks and tons of people from IL, IN, AR, KY, TX for some reason. 
        Primary Language - Mostly English, Unknown, Spanish, and a long tail of others...  Ignore for now. 
        """
        demo_df = prediction_times.merge(demo_data, 
                                         how='left', 
                                         left_on='masterpatientid',
                                         right_on='masterpatientid')
        
        time_since_birth = demo_df.predictiontimestamp - demo_df.dateofbirth
        age_in_days = np.array([td.days for td in time_since_birth])

        # Surprisingly, DOB is missing for some patients. 
        mask = np.isnan(age_in_days)
        age_in_days_na = np.zeros(len(age_in_days))
        age_in_days_na[mask] = 1
        age_in_days[mask] = 0
        age_df = pd.DataFrame({'demo_age_in_days': age_in_days, 
                               'demo_age_in_days_na': age_in_days_na})    

        # Gender
        gender = pd.get_dummies(demo_df.gender)
        gender.columns = 'demo_gender_' + gender.columns 
        demo_df = pd.concat([age_df, gender], axis=1)
        return demo_df
