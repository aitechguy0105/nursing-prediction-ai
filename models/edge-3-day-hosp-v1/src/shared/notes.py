import boto3
import pandas as pd
import numpy as np
import json
import re
import os 
import sys
import datetime
import gc

from pathlib import Path
from gensim.utils import simple_preprocess
from gensim.models import FastText
from functools import lru_cache
from collections import defaultdict
from multiprocessing import Pool 

#@lru_cache(maxsize=100000)

ft_model = None

def _swemWorkerInitialize(initargs): 
    model_path = initargs
    global ft_model 
    ft_model = FastText.load(model_path)

def _swemWorker(job_data):
    text_data = job_data[0]
    avg_vector = job_data[1]
    agg_mode = job_data[2]
    #avg_vector = np.concatenate([avg_vector, avg_vector, avg_vector])

    global ft_model

    tokens = simple_preprocess(text_data)
    if tokens:
        word_vectors = [ft_model.wv[t] if t in ft_model.wv else avg_vector for t in tokens]
        if agg_mode == 'mean': 
            vector = np.mean(word_vectors, axis=0)
        else: 
            vector = np.max(word_vectors, axis=0)
        return vector
    else:
        return avg_vector 

class SWEMNotesFeaturizer(): 
    def __init__(self, model_path='/data/models'): 
        self.model_path = model_path

    def featurize(self, prediction_times, stitched_notes, agg_mode='mean'): 
        ptimes = prediction_times.copy()[['masterpatientid', 'predictiontimestamp']]
        ptimes['predictiondate'] = (ptimes.predictiontimestamp - pd.to_timedelta('1 day')).dt.date

        print("Splitting notes into eMAR and other notes...")
        note_types = np.array([str(el) for el in stitched_notes.progressnotetype.unique()])
        emar_sel = np.array([re.match(r'emar', note_type, re.IGNORECASE) is not None for note_type in note_types])
        emar_note_types = note_types[emar_sel]
        is_emar = stitched_notes['progressnotetype'].isin(emar_note_types)
        emar_notes = stitched_notes.loc[is_emar]
        progress_notes = stitched_notes.loc[~is_emar]

        print("Loading embedding models...")
        embedding_models = [
                {'name': 'eMar',
                 'model_file': 'ft_emar_notes_d200.model', 
                 'notes': emar_notes},
                {'name': 'pn',
                 'model_file': 'ft_progress_notes_d200.model',  
                 'notes': progress_notes}
            ]

        vectors_list = []
        for model in embedding_models:
            print(f"Starting {model['name']}, {model['model_file']}, {self.model_path}...")
            model_path = os.path.join(self.model_path, model['model_file'])
            print(f"\tLoading model from {model_path}...")
            ft_model = FastText.load(model_path)
            model_name = model['name']
            notes_df = model['notes']

            avg_vector = np.mean(ft_model.wv.vectors, axis=0) # Average embedding for the space

            print("\tConstructing jobs data...")
            jobs_data = []
            notes_text_data = notes_df.notetext.values
            for note_text in notes_text_data: 
                job_data = (note_text, avg_vector, agg_mode)
                jobs_data.append(job_data)

            print("\tStarting jobs...")
            with Pool(min(os.cpu_count() - 4, 32), 
                      initializer=_swemWorkerInitialize, 
                      initargs=(model_path,)) as pool: 
                note_vectors = pool.map(_swemWorker, jobs_data)

            # Reshape embeddings/vectors into a dataframe.  Each row is a vector 
            # for one note, in same order as notes_df.  
            vectors_2d = np.stack(note_vectors)
            vectors_df = pd.DataFrame(vectors_2d) 
            vectors_df.columns = [f'notes_swem_{agg_mode}_{model_name}_{n}' for n in vectors_df.columns]
            vectors_df['createddate'] = notes_df.createddate.dt.date
            vectors_df['masterpatientid'] = notes_df.masterpatientid.values

            # Create patient days groupby object
            vectors_df = vectors_df.sort_values(['masterpatientid', 'createddate'])
            vectors_by_day = vectors_df.groupby(['masterpatientid', 'createddate'])

            # Aggregating vectors per patient day
            print('\tAggregating per day...')
            if agg_mode == 'mean': 
                vectors_for_days = vectors_by_day.mean()
            else: 
                vectors_for_days = vectors_by_day.max()
            pids = [x[0] for x in vectors_for_days.index.values]
            dates = [x[1] for x in vectors_for_days.index.values]
            vectors_for_days['masterpatientid'] = pids
            vectors_for_days['createddate'] = dates
            vectors_for_days = vectors_for_days.reset_index(drop=True)

            #return vectors_for_days 

            # Merge with combined dataframe
            print('\tMerging with prediction times...')
            combined = ptimes.copy().merge(vectors_for_days, 
                                           how='left', 
                                           left_on=['masterpatientid', 'predictiondate'],
                                           right_on=['masterpatientid', 'createddate'])
            embedding_cols = [c for c in combined.columns if c.startswith('notes_swem_')]
            cols_to_drop = [c for c in combined.columns if c not in embedding_cols]
            combined = combined.drop(columns=cols_to_drop)

            # Deal with NA's... 
            combined = combined.fillna(0)
            vectors_list.append(combined)

        retval = pd.concat(vectors_list, axis=1)
        return retval


class NoteCountFeaturizer() : 
    def __init__(self): 
        pass

    def featurize(self, prediction_times, stitched_notes): 
        """
        Just count up the number of notes in a given day... 
        """
        ptimes = prediction_times[['masterpatientid', 'predictiontimestamp']].copy()
        # Get notes from day before prediction date... 
        ptimes['predictiondate'] = (ptimes.predictiontimestamp - pd.to_timedelta('1 day')).dt.date

        notes = stitched_notes[['masterpatientid', 'createddate']].copy()
        notes['createddate'] = notes.createddate.dt.date
        notes['indicator'] = 1
        grouped_notes = notes.groupby(['masterpatientid', 'createddate'])
        note_counts_by_day = grouped_notes.sum()

        # now merge with ptimes... 
        combined = ptimes.merge(note_counts_by_day, 
                                how='left', 
                                left_on=['masterpatientid', 'predictiondate'],
                                right_on=['masterpatientid', 'createddate'])
        combined = combined.fillna(0)
        return combined



