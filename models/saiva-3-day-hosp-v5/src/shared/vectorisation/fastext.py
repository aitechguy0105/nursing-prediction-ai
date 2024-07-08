import gc
import sys
import os
import subprocess
from multiprocessing import Pool

import numpy as np
from eliot import start_action, log_message
from gensim.models import FastText
from gensim.utils import simple_preprocess
from functools import lru_cache

sys.path.insert(0, '/src')

from shared.vectorisation.base import VectorisationModel


class FastTextModel(VectorisationModel):
    def __init__(self, notes, client):
        self.model = None
        self.avg_vector = None
        self.agg_mode = 'sum'
        super().__init__(notes, client)

    def get_embedding_models(self):
        self.download_nlp_models(self.client)
        emar_notes, progress_notes = self.get_grouped_notes()
        return self.get_config(emar_notes, progress_notes)

    def load(self, name):
        return FastText.load(name)

    @lru_cache(maxsize=100000)
    def vector_lookup(self, t):
        # Look up embedding for passed token
        # If token isn't found pass in the average embedding
        try:
            return self.model.wv[t]
        except KeyError:
            return self.avg_vector

    def _vectorise_job(self, text_data):
        tokens = simple_preprocess(text_data)
        if tokens:
            word_vectors = [self.vector_lookup(t) for t in tokens]
            if self.agg_mode == 'sum':
                vector = np.sum(word_vectors, axis=0)
            elif self.agg_mode == 'mean':
                vector = np.mean(word_vectors, axis=0)
            else:
                vector = np.max(word_vectors, axis=0)
            return vector
        else:
            return self.avg_vector

    def vectorise(self, notes_df, model):
        self.model = model
        # Average embedding alternative of any given word
        self.avg_vector = np.mean(model.wv.vectors, axis=0)

        log_message(message_type='info', message=f'Vectorising notetext using FastText Model...')
        # Vectorize all notes using parallel processing
        with Pool(os.cpu_count() - 2) as pool:
            vectors = pool.map(self._vectorise_job, notes_df['notetext'])  # Vectorize all notes

        return vectors

    def get_grouped_notes(self):
        """
        :return: emar_notes, progress_notes
        """

        # Fetch the valid sections of Note Embeddings from client specific class
        valid_sections = getattr(self.clientClass(), 'get_note_embeddings_valid_section')()

        log_message(message_type='info', message=f'Filter for EMAR & Progress Notes..')
        # Filter out None progressnotetypes
        self.notes_df = self.notes_df[
            self.notes_df['progressnotetype'].isna() == False
            ]

        # If client has configured for valid sections then filter it or else use all progress notes
        if valid_sections:
            # Filter for valid section of progress notes
            is_valid = self.notes_df.apply(
                lambda x: x['progressnotetype'] + '_' + x['section'], axis=1
            ).isin(valid_sections)
            valid_note_parts = self.notes_df.loc[is_valid]
        else:
            valid_note_parts = self.notes_df

        del self.notes_df

        # Join the notetext for the column group mentioned below
        valid_note_parts.sort_values(by=['facilityid', 'patientid', 'createddate', 'progressnoteid', 'progressnotetype',
                                         'section', 'sectionsequence', 'notetextorder'], inplace=True)
        grp_cols = ['facilityid', 'masterpatientid', 'createddate', 'progressnoteid', 'progressnotetype', 'section']
        valid_notes = (valid_note_parts.groupby(grp_cols).agg({'notetext': lambda note_parts: ''.join(
            note_parts)}).reset_index())
        del valid_note_parts

        # Filter for EMAR Notes & Progress Notes
        emar_types = getattr(self.clientClass(), 'get_note_embeddings_emar_types')()
        # Find all notetypes that contain the word emar and add it in emar_types list
        emar_types.extend([nt for nt in valid_notes['progressnotetype'].unique() if 'emar' in nt.lower()])

        is_emar = valid_notes['progressnotetype'].isin(emar_types)

        emar_notes = valid_notes.loc[is_emar]
        progress_notes = valid_notes.loc[~is_emar]

        assert len(emar_notes) + len(progress_notes) == len(valid_notes)
        del valid_notes
        # =============Trigger garbage colleFilter for EMAR & Progress Notes..ction to free-up memory ==================
        gc.collect()

        return emar_notes, progress_notes

    def download_nlp_models(self, client):
        """
        Downlaod the relevant Progress Note Embedding models from S3
        """
        with start_action(action_type='download_nlp_models'):
            subprocess.run(
                f'aws s3 sync s3://saiva-models/progress_note_embeddings/v1 /data/models/',
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def get_config(self, emar_notes, progress_notes):
        """
        - During predicition, fetch the word vector models from the
          downloaded facility model.
        - During Training, fetch it from local directory
        """
        assert len(emar_notes) != 0
        assert len(progress_notes) != 0

        embedding_models = [
            {
                'name': 'eMar',
                'model_path': f'/data/models/ft_emar.model',
                'notes': emar_notes,
            },
            {
                'name': 'pn',
                'model_path': f'/data/models/ft_non_emar.model',
                'notes': progress_notes,
            },
        ]

        return embedding_models
