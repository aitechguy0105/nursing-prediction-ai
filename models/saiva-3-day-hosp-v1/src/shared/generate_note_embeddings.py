import gc
import os
import re
import subprocess
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from eliot import log_message, start_action
from gensim.models import FastText
from shared.constants import MODELS

sys.path.insert(0, '/src')
from shared.utils import downcast_dtype

from multiprocessing import Pool

AVG_VECTOR = None
FT_MODEL = None


def generating_notes(progress_notes_parts, clientClass):
    """

    :param progress_notes_parts:
    :return:
    """
    # Fetch the valid sections of Note Embeddings from client specific class
    valid_sections = getattr(clientClass(), 'get_note_embeddings_valid_section')()

    log_message(message_type='info', message=f'Filter for EMAR & Progress Notes..')
    # Filter out None progressnotetypes
    progress_notes_parts = progress_notes_parts[
        progress_notes_parts['progressnotetype'].isna() == False
        ]

    # If client has configured for valid sections then filter it or else use all progress notes
    if valid_sections:
        # Filter for valid section of progress notes
        is_valid = progress_notes_parts.apply(
            lambda x: x['progressnotetype'] + '_' + x['section'], axis=1
        ).isin(valid_sections)
        valid_note_parts = progress_notes_parts.loc[is_valid]
    else:
        valid_note_parts = progress_notes_parts

    del progress_notes_parts

    # Join the notetext for the column group mentioned below
    valid_note_parts.sort_values(by=['facilityid', 'patientid', 'createddate', 'progressnoteid', 'progressnotetype',
                                     'section', 'sectionsequence', 'notetextorder'], inplace=True)
    grp_cols = ['facilityid', 'masterpatientid', 'createddate', 'progressnoteid', 'progressnotetype', 'section']
    valid_notes = (valid_note_parts.groupby(grp_cols).agg({'notetext': lambda note_parts: ''.join(
        note_parts)}).reset_index())
    del valid_note_parts

    # Filter for EMAR Notes & Progress Notes
    emar_types = getattr(clientClass(), 'get_note_embeddings_emar_types')()
    is_emar = valid_notes['progressnotetype'].isin(emar_types)

    emar_notes = valid_notes.loc[is_emar]
    progress_notes = valid_notes.loc[~is_emar]

    assert len(emar_notes) + len(progress_notes) == len(valid_notes)
    del valid_notes
    # =============Trigger garbage colleFilter for EMAR & Progress Notes..ction to free-up memory ==================
    gc.collect()

    return emar_notes, progress_notes


def get_embedding_models(client, emar_notes, progress_notes):
    """
    During predicition fetch the word vector models from the
    downloaded facility model & during Training fetch it from local directory
    """
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


@lru_cache(maxsize=100000)
def vector_lookup(t):
    # Look up embedding for passed token
    # If token isn't found pass in the average embedding
    global FT_MODEL
    global AVG_VECTOR
    try:
        return FT_MODEL.wv[t]
    except KeyError:
        return AVG_VECTOR


def vectorize_note(s):
    # Get embeddings for every token and sum them together
    # Return the sum vector
    global AVG_VECTOR
    tokens = tokenise(s)
    if tokens:
        word_vectors = [vector_lookup(t) for t in tokens]
        note_vector = np.sum(word_vectors, axis=0)
        return note_vector
    else:
        return (AVG_VECTOR)  # Use average embedding if sentence has no embeddings


# Unigram tokenizer
def tokenise(s):
    s = s.lower()
    tokens = re.split(r'\b', s)
    return tuple(t for t in tokens if len(t) > 0 and t != ' ')


def download_nlp_models(client):
    """
    Downlaod the relevant Progress Note Embedding models from S3
    """
    with start_action(action_type='download_nlp_models'):
        s3_folder_path = MODELS[client]['vector_model_s3_path']
        subprocess.run(
            f'aws s3 sync {s3_folder_path} /data/models/',
            shell=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )


def processing_word_vectors(final, emar_notes, progress_notes, client, clientClass):
    """
    :param final: Preprocessed dataframe containing all other tables accept Progress Notes
    :param emar_notes: Filtered Progress Note dataframe
    :param progress_notes: Filtered Progress Note dataframe
    inorder fetch the vector models from the downloaded path
    """
    global FT_MODEL
    global AVG_VECTOR

    # Download NLP models from S3
    download_nlp_models(client)

    log_message(message_type='info', message=f'Process word vectors..')
    embedding_models = get_embedding_models(client, emar_notes, progress_notes)

    # Loop through embedding model types
    for model in embedding_models:
        model_name = model['name']
        with start_action(action_type='processing_word_vectors', model=model_name):
            log_message(message_type='info', message=f'Starting {model_name}...')
            FT_MODEL = FastText.load(model['model_path'])
            notes_df = model['notes']
            # check: if emar notes are not present then skip it.
            if len(notes_df) == 0:
                continue
            # Average embedding alternative of any given word
            AVG_VECTOR = np.mean(FT_MODEL.wv.vectors, axis=0)

            log_message(message_type='info', message=f'Vectorising notetext...')
            # Vectorize all notes using parallel processing
            with Pool(os.cpu_count() - 2) as pool:
                vectors = pool.map(vectorize_note, notes_df['notetext'])  # Vectorize all notes

            # Reshape embeddings/vectors into a dataframe
            vectors_2d = np.reshape(vectors, (len(vectors), len(vectors[0])))
            vectors_df = pd.DataFrame(vectors_2d)
            vectors_df.columns = [
                f'e_{model_name}_{n}' for n in vectors_df.columns
            ]

            # Create unit vectors from raw vectors; make into a dataframe
            unit_vectors = (vectors_2d / np.linalg.norm(vectors_2d, axis=1)[:, None])
            unit_vectors_df = pd.DataFrame(unit_vectors)
            unit_vectors_df.columns = [f'e_{model_name}_unit_{n}' for n in unit_vectors_df.columns]

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
            embedding_cols = [c for c in notes.columns if c.startswith('e_')]

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
            AVG_VECTOR = None
            FT_MODEL = None
            gc.collect()
            note_aggs = downcast_dtype(note_aggs)

            log_message(message_type='info', message=f'Merging Note vectors dataframe with base dataframe.')
            # Merge with combined dataframe
            final = final.merge(note_aggs, on=['masterpatientid', 'censusdate'], how='left')
            # =============Trigger garbage collection to free-up memory ==================
            del note_aggs
            gc.collect()

    log_message(message_type='info', message=f'...forward filling...')
    embedding_cols = [c for c in final.columns if c.startswith("e_")]
    final = final.sort_values(["masterpatientid", "censusdate"])
    final[embedding_cols] = final.groupby("masterpatientid")[embedding_cols].fillna(method="ffill")

    # Ensure most patient days have embeddings (they should, after forward filling)
    # Test amount of NaN in vector columns
    threshold_value = getattr(clientClass(), 'get_note_embeddings_nan_threshold')()
    mean_na_count = final['e_pn_0'].isna().mean()
    log_message(message_type='info', message=f'mean na count = {mean_na_count}')
    if mean_na_count >= threshold_value:
        log_message(message_type='warning', message=f'WARNING: {mean_na_count} is less than {threshold_value}')
    # assert final['e_pn_0'].isna().mean() < threshold_value

    log_message(message_type="info", message="Embeddings completed..")
    log_message(message_type="info", Final_Dataframe_Shape=final.shape)

    return final
