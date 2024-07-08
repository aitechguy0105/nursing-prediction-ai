"""
- NLP models reside at aws s3 sync s3://saiva-models/progress_note_embeddings/v1
- These models are added into AWS EFS. EFS is mounted into the runtime container
dynamically using `TaskDefinition`
- During dev run EFS is added as Docker Volume and attached the Docker runtime
container
"""
import gc
import sys

import numpy as np
import pandas as pd
from eliot import log_message
from shared.constants import VECTOR_MODELS

sys.path.insert(0, '/src')
from shared.utils import get_client_class
from shared.featurizer import BaseFeaturizer
from shared.vectorisation.spacy import SpacyModel
from shared.vectorisation.fastext import FastTextModel


class NoteFeatures(BaseFeaturizer):
    def __init__(self, census_df, notes, client, vector_model='SpacyModel', training=False):
        super(NoteFeatures, self).__init__()
        self.census_df = census_df
        self.notes_df = notes
        self.training = training
        self.client = client
        self.clientClass = get_client_class(self.client)
        self.vector_model = vector_model

    def load_model(self):
        if self.vector_model == 'SpacyModel':
            return SpacyModel(notes=self.notes_df, client=self.client)
        elif self.vector_model == 'FastTextModel':
            return FastTextModel(notes=self.notes_df, client=self.client)

    def generate_features(self):
        """
        :param final: Preprocessed dataframe containing all other tables accept Progress Notes
        :param emar_notes: Filtered Progress Note dataframe
        :param progress_notes: Filtered Progress Note dataframe
        inorder fetch the vector models from the downloaded path
        """
        log_message(message_type='info', message=f'Process word vectors..')

        # choose the NLP model
        nlp_model = self.load_model()
        # get segregated notes and models as dictionary
        embedding_models = nlp_model.get_embedding_models()

        # Loop through embedding model types
        for model in embedding_models:
            model_name = model['name']
            log_message(message_type='info', message=f'Start processing using {model_name} model...')
            notes_df = model['notes']
            if len(notes_df) == 0:
                continue
            # Vectorise the notes
            vectors = nlp_model.vectorise(notes_df, nlp_model.load(model['model_path']))

            # Reshape embeddings/vectors into a dataframe
            vectors_2d = np.reshape(vectors, (len(vectors), len(vectors[0])))
            vectors_df = pd.DataFrame(vectors_2d)
            vectors_df.columns = [
                f'{model_name}_{n}' for n in vectors_df.columns
            ]

            # Create unit vectors from raw vectors; make into a dataframe
            unit_vectors = (vectors_2d / np.linalg.norm(vectors_2d, axis=1)[:, None])
            unit_vectors_df = pd.DataFrame(unit_vectors)
            unit_vectors_df.columns = [f'{model_name}_unit_{n}' for n in unit_vectors_df.columns]

            # Concatentate notes with vectors and unit vectors
            notes = pd.concat(
                [
                    notes_df.reset_index(),
                    vectors_df.reset_index(),
                    unit_vectors_df.reset_index(),
                ],
                axis=1,
            )
            notes['censusdate'] = notes['createddate'].dt.normalize()

            log_message(message_type='info', message=f'Completed Concatentating vector dataframes..')

            # Create patient days groupby object
            notes_patient_days = notes.sort_values(['masterpatientid', 'censusdate']).groupby(
                ['masterpatientid', 'censusdate'])
            embedding_cols = [c for c in notes.columns if c.startswith('notes_')]

            # Sum vectors per patient day
            log_message(message_type='info', message=f'...patient-day aggs...')
            note_aggs = notes_patient_days[embedding_cols].apply(sum)
            # Create exponential weighted moving (EWM) averages for note embeddings by patient day
            note_aggs_cumulative = note_aggs.groupby('masterpatientid').apply(lambda x: x.ewm(halflife=7).mean())
            note_aggs_cumulative.columns = [c + '_ewm' for c in note_aggs_cumulative.columns]
            # Concat EWMAs onto patient_days dataframe
            note_aggs = pd.concat([note_aggs, note_aggs_cumulative], axis=1)

            # =============Trigger garbage collection & downcast to free-up memory ==================
            del notes
            del vectors
            del vectors_2d
            del vectors_df
            del unit_vectors
            del unit_vectors_df
            gc.collect()
            note_aggs = self.downcast_dtype(note_aggs)

            log_message(message_type='info', message=f'Merging Note vectors dataframe with base dataframe.')
            # Merge with combined dataframe
            self.census_df = self.census_df.merge(note_aggs, on=['masterpatientid', 'censusdate'], how='left')
            # =============Trigger garbage collection to free-up memory ==================
            del note_aggs
            gc.collect()

        log_message(message_type='info', message=f'...forward filling...')
        embedding_cols = [c for c in self.census_df.columns if c.startswith("notes_")]
        final = self.census_df.sort_values(["masterpatientid", "censusdate"])
        final[embedding_cols] = final.groupby("masterpatientid")[embedding_cols].fillna(method="ffill")

        # Ensure most patient days have embeddings (they should, after forward filling)
        # Test amount of NaN in vector columns
        threshold_value = getattr(self.clientClass(), 'get_note_embeddings_nan_threshold')()
        mean_na_count = final['notes_0'].isna().mean()
        log_message(message_type='info', message=f'mean na count = {mean_na_count}')
        if mean_na_count >= threshold_value:
            log_message(message_type='warning', message=f'WARNING: {mean_na_count} is less than {threshold_value}')

        log_message(message_type="info", message="Embeddings completed..")
        log_message(message_type="info", Final_Dataframe_Shape=final.shape)

        # drop unwanted columns
        final = self.drop_columns(
            final,
            '_masterpatientid|_facilityid|_x$|_y$|bedid|censusactioncode|payername|payercode'
        )

        return final
