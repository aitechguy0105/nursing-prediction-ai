import os
import pickle as pkl
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


def _BagAggMedNameWorker(job_data):
    job_idx = job_data[0]
    code = job_data[1]
    codes = job_data[2]
    indices = job_data[3]
    nrows = job_data[4]

    code_mask = (codes == code)
    indices = indices[code_mask]
    indices = indices[np.invert(np.isnan(indices))]
    indices = indices.astype(np.int)

    column = np.zeros(nrows)
    column[indices] = 1
    column = pd.Series(pd.arrays.SparseArray(column, fill_value=0))

    return column


class BagOfAggregatedMedNames:
    def __init__(self, min_count=200):
        self.min_count = min_count

    def aggregate_codes(self, meds, save_file=None):
        med_names = meds.pharmacymedicationname.values
        med_names = np.array([name.upper() if name is not None else name for name in med_names])
        if save_file is not None:
            if Path(save_file).is_file():
                print(f'Loading code map from {save_file}')
                with open(save_file, 'rb') as f_in:
                    code_map = pkl.load(f_in)
                self.code_map = code_map
            else:
                raise Exception(f'Invalid file path - {save_file}')
        else:
            print(f'Aggregating med names and gpi subclass descriptions')

            gpi_descs = meds.gpisubclassdescription.values
            med_name_to_gpi = defaultdict(str)
            for idx, med_name in enumerate(med_names):
                med_name_to_gpi[med_name] = gpi_descs[idx].upper() if gpi_descs[idx] is not None else med_name

            med_counts = defaultdict(int)

            for med_name in med_names:
                med_counts[med_name] += 1

            new_med_names = []
            for med_name in med_names:
                if med_counts[med_name] < self.min_count:
                    new_med_names.append(med_name_to_gpi[med_name])
                else:
                    new_med_names.append(med_name)
            new_med_names = np.array(new_med_names)

            code_map = {}
            for med_name, new_med_name in zip(med_names, new_med_names):
                code_map[med_name] = new_med_name
            self.code_map = code_map

        return self.code_map

    def get_aggregated_codes(self, codes):
        codes = [code.upper() if type(code) is str else code for code in codes]
        new_codes = [self.code_map[code] if code in self.code_map else 'UNK' for code in codes]
        return new_codes

    def featurize(self, prediction_times, meds_data, med_colname):
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
        key_to_row = {k: idx for idx, k in enumerate(predict_keys)}
        print("Getting med keys")
        dates = meds_data.orderdate.dt.date.values
        med_keys = get_keys(dates,
                            meds_data.masterpatientid.values,
                            meds_data.facilityid.values)
        print("Mapping med keys to prediction keys")
        med_row_to_predict_row = [key_to_row[med_key] if med_key in key_to_row else np.nan for med_key in med_keys]

        # Get subset of dx_data... 
        meds_data = meds_data[['masterpatientid', 'facilityid', 'orderdate', med_colname]].copy()
        meds_data.loc[:, 'PredictRow'] = med_row_to_predict_row

        # Helper function to construct sparse column for one code. 
        print("Setting up jobs data")
#         unique_med_codes = meds_data[med_colname].unique()
        
        # Getting unique Med codes by converting them to set
        unique_med_codes = list(set(self.code_map.values()))
        # Always retains the column order
        unique_med_codes.sort()
        
        job_indices = np.arange(len(unique_med_codes))
        job_data = []
        for job_idx in job_indices:
            job_data.append((job_idx,
                             unique_med_codes[job_idx],
                             meds_data[med_colname].values,
                             meds_data.PredictRow.values,
                             len(prediction_times)))

        # Set up compute thread pool and start jobs
        print('Starting jobs!')
        with Pool(min(os.cpu_count() - 4, 24)) as pool:
            med_columns = pool.map(_BagAggMedNameWorker, job_data)

            # Construct final data frame
        print('Concatenating columns...')
#         med_colnames = 'rx_' + unique_med_codes
        med_colnames = ['rx_' + code for code in unique_med_codes]
        meds_df = pd.concat(med_columns, keys=med_colnames, axis=1)

        return meds_df
