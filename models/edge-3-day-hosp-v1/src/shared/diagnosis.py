import boto3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import os 
import sys
import datetime
import pickle as pkl

from collections import defaultdict
from multiprocessing import Pool 

# NB - this needs to be a top level wrt module function in order to work 
# with multiprocessing.Pool...  
def _BagAggDxWorker(job_data): 
    job_idx = job_data[0]
    code = job_data[1]
    codes = job_data[2]
    indices = job_data[3]
    nrows = job_data[4]

    code_mask = (codes == code)
    indices = indices[code_mask]
    indices = indices[ np.invert(np.isnan(indices)) ]
    indices = indices.astype(np.int)

    column = np.zeros(nrows) # Set dtype=np.float32?
    column[indices] = 1
    column = pd.Series(pd.arrays.SparseArray(column, fill_value=0))

    return column



class BagOfAggregatedDxCodes: 
    def __init__(self, min_count=100, min_code_length=1, stop_at_dots=False): 
        self.min_count = min_count
        self.min_code_length = min_code_length
        self.stop_at_dots = stop_at_dots
        self.code_map = {}
        pass

    def get_code_map(self, codes, save_file=None): 
        """
        Maps ICD10 codes based on its hierarchy and frequency
        """
        if save_file is not None: 
            save_file_path = Path(save_file) 
            if save_file_path.is_file(): 
                print(f'Loading code map from {save_file}')
                with open(save_file, 'rb') as f_in: 
                    self.code_map = pkl.load(f_in)
            else: 
                raise Exception(f'Invalid file path - {save_file}')
        else: 
            print(f"Aggregating codes to min_count = {self.min_count}")

            code_counts = defaultdict(int)
            code_map = defaultdict(str)

            # Create entries in code map for original codes with '.' delimiters. 
            if not self.stop_at_dots: 
                for code in codes: 
                    code_map[code] = code.replace('.', '')
                codes = [code_map[code] for code in codes] #TODO: code_map.values()
            
            # Get frequency of codes & add new codes to code_map
            for code in codes: 
                code_counts[code] += 1
                code_map[code] = code
            print(f"Got {len(code_counts)} codes")    
            
            code_lengths = sorted(set([len(code) for code in codes]), reverse=True)
            print(code_lengths)
            for code_length in code_lengths: 
                print(f">>>>> Merging codes of length {code_length} <<<<<")
                num_merged = 0
                new_code_counts = defaultdict(int)
                for code, code_count in code_counts.items(): 
                    if len(code) == code_length and code_count < self.min_count:
                        if len(code) > self.min_code_length: 
                            # If the code length is greater than threshold remove the last character in the code
                            new_code = code[:-1]
                            if self.stop_at_dots and new_code[-1] == '.': 
                                # If dot exists as last character, remove it
                                new_code = new_code[:-1]
                                code_count = self.min_count
                        else: 
                            new_code = 'Other'
                        new_code_counts[new_code] += code_count
                        code_map[code] = new_code
                        num_merged += 1
                    else: 
                        new_code_counts[code] += code_count
                code_counts = new_code_counts
                print(f"Merged {num_merged}, {len(code_counts)} codes, {len(code_map)} map entries")
        
            def get_final_map_value(code_map, code): 
                # Check whether code is not present in keys(), since defaultdict it should return ''
                if (code_map[code] == '') or (code_map[code] == code): 
                    return code
                else: 
                    return get_final_map_value(code_map, code_map[code]) 
            
            final_code_map = defaultdict(str)
            for code in code_map.copy().keys():
                final_code_map[code] = get_final_map_value(code_map, code)
                
            self.code_map = final_code_map

        return self.code_map

    def get_aggregated_codes(self, codes): 
        if not self.stop_at_dots: 
            codes = [c.replace('.', '') for c in codes]
        new_codes = [self.code_map[code] if code in self.code_map else 'UNK' for code in codes]
        return new_codes

    def featurize(self, prediction_times, dx_data, dx_colname): 
        prediction_times = prediction_times.copy()

        def get_keys(dates, pids, fids): 
            keys = []
            for date, pid, fid in zip(dates, pids, fids): 
                keys.append(f'{date}_{pid}_{fid}')
            return keys
        print(f"getting prediction keys")
        dates = (prediction_times.predictiontimestamp + pd.to_timedelta("-1 day")).dt.date.values
        predict_keys = get_keys(dates, 
                                prediction_times.masterpatientid.values,
                                prediction_times.facilityid.values)
        key_to_row = {k:idx for idx, k in enumerate(predict_keys)}

        print("Getting dx keys")
        dates = dx_data.onsetdate.dt.date.values
        dx_keys = get_keys(dates, 
                           dx_data.masterpatientid.values,
                           dx_data.facilityid.values)

        print("Mapping dx keys to prediction keys")
        dx_row_to_predict_row = [key_to_row[dx_key] if dx_key in key_to_row else np.nan for dx_key in dx_keys]

        # Get subset of dx_data... 
        dx_data = dx_data[['masterpatientid', 'facilityid', 'onsetdate', dx_colname]].copy()
        dx_data.loc[:,'PredictRow'] = dx_row_to_predict_row

        # Helper function to construct sparse column for one code. 
        print("Setting up jobs data")
#         unique_dx_codes = dx_data[dx_colname].unique()
        
        # Getting unique Med codes by converting them to set
        unique_dx_codes = list(set(self.code_map.values()))
        # Always retains the column order
        unique_dx_codes.sort()
        
        job_data = []
        for index, code in enumerate(unique_dx_codes): 
            job_data.append((index, 
                             code, 
                             dx_data[dx_colname].values, 
                             dx_data.PredictRow.values, 
                             len(prediction_times)))

        # Set up compute thread pool and start jobs
        print('Starting jobs!')
        with Pool(min(os.cpu_count() - 4, 24)) as pool:
            dx_columns = pool.map(_BagAggDxWorker, job_data) 

        # Construct final data frame
        print('Concatenating columns...')
        # dx_colnames = 'dx_' + unique_dx_codes 
        dx_colnames = ['dx_' + code for code in unique_dx_codes]
        dx_df = pd.concat(dx_columns, keys=dx_colnames, axis=1)

        return dx_df



    