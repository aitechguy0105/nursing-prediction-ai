import gc
import os
import re
import string
import sys
from functools import partial

import numpy as np
from eliot import log_message, start_action

import spacy

from .base import VectorisationModel

class SpacyModel(VectorisationModel):
    def __init__(self, notes, client):
        self.model = None
        super().__init__(notes, client)

    def get_embedding_models(self):
        self.notes_df = self.get_grouped_notes()
        return self.get_config(self.notes_df)

    def load(self, name):
        return spacy.load(name)

    def vectorise(self, notes_df, model):
        
        def _doc_to_vec(doc, default):
            vector = [word.vector for word in doc if (len(word.text) > 1 and not word.is_stop)]
            vector = np.mean(vector, axis=0) if vector else default
            return vector
        
        log_message(message_type='info', message=f'Notes - Vectorising notetext using Spacy Model.')
        # Vectorize all notes using parallel processing
        texts = notes_df['notetext'].str.lower().str.replace('[^A-Za-z-\\\]+', ' ', regex=True).str.strip('-\\')
        default = model('').vector # value for zero-length / zero-words notes
        docs = model.pipe(texts, n_process=-1, batch_size=250) # vectorize with pipe() using builtin parallelization
        vectors = [_doc_to_vec(doc, default) for doc in docs]
        return vectors

    def get_grouped_notes(self):
        """
        - Filter for notetypes
        - Arrange notes in sequence
        """
        # Fetch the valid sections of Note Embeddings from client specific class
        valid_sections = getattr(self.clientClass(), 'get_note_embeddings_valid_section')()

        log_message(message_type='info', message=f'Notes - Filter for valid Progress Notes.')
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
        # removing rows where progress notes is empty.
        valid_note_parts = valid_note_parts[~valid_note_parts['notetext'].isna()]
        valid_notes = (valid_note_parts.groupby(grp_cols).agg({'notetext': lambda note_parts: ''.join(
            note_parts)}).reset_index())
        del valid_note_parts
        gc.collect()

        return valid_notes

    def get_config(self, notes_df):
        """
        - Specify the spacy model name under model_path
        """
        assert len(notes_df) != 0

        embedding_models = [
            {
                'name': 'notes',
                'model_path': f'en_core_sci_lg',
                'notes': notes_df,
            }
        ]

        return embedding_models
