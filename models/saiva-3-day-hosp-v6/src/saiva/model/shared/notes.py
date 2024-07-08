"""
- NLP models reside at aws s3 sync s3://saiva-models/progress_note_embeddings/v1
- These models are added into AWS EFS. EFS is mounted into the runtime container
dynamically using `TaskDefinition`
- During dev run EFS is added as Docker Volume and attached the Docker runtime
container
"""
import gc
import sys
import time
import numpy as np
import pandas as pd
from eliot import log_message, start_action

from .constants import VECTOR_MODELS
from .utils import get_client_class
from .featurizer import BaseFeaturizer
from .vectorisation.spacy import SpacyModel
from .vectorisation.fastext import FastTextModel


class NoteFeatures(BaseFeaturizer):
    def __init__(self, census_df, notes, client, vector_model='SpacyModel', training=False):
        super(NoteFeatures, self).__init__()
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
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
        with start_action(action_type=f"Progress Notes - generating progress note features", progress_note_shape = self.notes_df.shape):
            start = time.time()
            self.census_df['censusdate'] = pd.to_datetime(self.census_df['censusdate'])
            #==================================notes type categorical features processing starts=======================     
            note_event_df = self.notes_df.loc[:,['facilityid', 'masterpatientid', 'progressnoteid',  'progressnotetype', 
                                                 'highrisk', 'showon24hr', 'showonshift']]
            note_event_df['censusdate'] = pd.to_datetime(self.notes_df['createddate'])
            note_event_df.drop_duplicates(inplace=True)
            note_event_df['censusdate'] = pd.to_datetime(note_event_df['censusdate'].dt.date)
            note_event_df['indicator'] = 1
            note_event_df = self.sorter_and_deduper(
                note_event_df,
                sort_keys=['facilityid', 'masterpatientid', 'censusdate'],
                unique_keys=['facilityid', 'masterpatientid', 'censusdate', 'progressnotetype', 'highrisk', 'showon24hr', 'showonshift']
            )
            note_event_df['progressnotetype'] = note_event_df['progressnotetype'].str.lower()

            # get pivoted table for progressnotetype
            notetype_df = note_event_df[['masterpatientid','censusdate','progressnotetype','indicator']]
            notetype_pivoted = pd.pivot_table(notetype_df,
                                        index = ['masterpatientid','censusdate'],
                                        columns = 'progressnotetype',
                                        values='indicator',
                                        aggfunc = np.sum, 
                                        fill_value=0).reset_index()
            notetype_names = ['notes_notetype_'+ name for name in notetype_pivoted.columns[2:]]
            notetype_pivoted.columns = ['masterpatientid','censusdate'] + notetype_names
            log_message(
                message_type = 'info', 
                message = f'Notes - created features from progressnote types.', 
                notetype_pivoted_shape = notetype_pivoted.shape
            )
            # get pivoted table for highrisk
            highrisk_df = note_event_df[['masterpatientid','censusdate','highrisk']]
            highrisk_df['highrisk_indicator'] = np.where(highrisk_df['highrisk']=='Y', 1, 0)

            highrisk_grouped = highrisk_df.groupby(['masterpatientid','censusdate'])['highrisk_indicator'].sum().reset_index()
            highrisk_grouped.columns = ['masterpatientid','censusdate', 'notes_highrisk']
            log_message(
                message_type = 'info', 
                message = f'Notes - created features from highrisk notes.', 
                highrisk_grouped_shape = highrisk_grouped.shape
            )
            # get pivoted table for showon24hr
            showon24hr_df = note_event_df[['masterpatientid','censusdate','showon24hr']]
            showon24hr_df['showon24hr_indicator'] = np.where(showon24hr_df['showon24hr']=='Y', 1, 0)

            showon24hr_grouped = showon24hr_df.groupby(['masterpatientid','censusdate'])['showon24hr_indicator'].sum().reset_index()
            showon24hr_grouped.columns = ['masterpatientid','censusdate', 'notes_showon24hr']
            log_message(
                message_type = 'info', 
                message = f'Notes - created features from reports that will show within 24 hour shift.', 
                showon24hr_grouped_shape = showon24hr_grouped.shape
            )
            # get pivoted table for showonshift
            showonshift_df = note_event_df[['masterpatientid','censusdate','showonshift']]
            showonshift_df['showonshift_indicator'] = np.where(showonshift_df['showonshift']=='Y', 1, 0)

            showonshift_grouped = showonshift_df.groupby(['masterpatientid','censusdate'])['showonshift_indicator'].sum().reset_index()
            showonshift_grouped.columns = ['masterpatientid','censusdate', 'notes_showonshift']
            log_message(
                message_type = 'info', 
                message = f'Notes - created features from reports that will show within shift.', 
                showonshift_grouped_shape = showonshift_grouped.shape
            )
            # Merge notetype_pivoted, highrisk_pivoted, showon24hr_pivoted
            final_cate_note_df = notetype_pivoted.merge(highrisk_grouped,
                                                        on=['masterpatientid','censusdate'],
                                                        how='outer')
            final_cate_note_df = final_cate_note_df.merge(showon24hr_grouped, 
                                                          on=['masterpatientid','censusdate'], 
                                                          how='outer')
            final_cate_note_df = final_cate_note_df.merge(showonshift_grouped, 
                                                          on=['masterpatientid','censusdate'],
                                                          how='outer')

            # check if we loss any data form merging

            col_names = notetype_names + ['notes_highrisk', 'notes_showon24hr', 'notes_showonshift']
            # merge the data with census data
            events_df = self.census_df[['masterpatientid','censusdate']]\
                                        .merge(final_cate_note_df, 
                                             on=['masterpatientid','censusdate'],
                                             how='outer')
            assert events_df.duplicated(subset=['masterpatientid','censusdate']).any() == False
            events_df[col_names] = events_df[col_names].fillna(0)
            events_df['censusdate'] = pd.to_datetime(events_df['censusdate'])
            events_df = events_df.sort_values(by=['masterpatientid', 'censusdate'])\
                        .reset_index(drop=True)
            events_df = self.downcast_dtype(events_df)
            #++++++++++++cumsum for event happened start+++++++++++++++++++++++
            # Do cumulative summation
            log_message(message_type='info', message='Notes - cumulative summation, patient days with any events cumsum.')        
            cumsum_note_df = self.get_cumsum_features(col_names, events_df, cumidx=True)

            # Do cumulative index
            log_message(message_type='info', message='Notes - cumulative summation, total number of events cumsum.')        
            cumidx_note_df = self.get_cumsum_features(col_names, events_df, cumidx=False) 
            #++++++++++++cumsum for event happened end+++++++++++++++++++++++

            #++++++++++++get counts of days since last event for all events start+++++++++++++++++++++++
            log_message(message_type='info', message='Notes - count of days since last event.')
            days_last_event_df = self.apply_n_days_last_event(events_df, col_names)

            #++++++++++++get counts of days since last event for all events end+++++++++++++++++++++++++++

            del notetype_pivoted
            del highrisk_grouped
            del showon24hr_grouped
            del showonshift_grouped
            del final_cate_note_df
            del note_event_df 
            final_cate_df = self.census_df.merge(cumsum_note_df, 
                                                 on=['masterpatientid','censusdate'])
            final_cate_df = final_cate_df.merge(cumidx_note_df, 
                                                 on=['masterpatientid','censusdate'])
            final_cate_df = final_cate_df.merge(days_last_event_df,
                                                on=['masterpatientid','censusdate'])

            self.sanity_check(final_cate_df)
            del cumsum_note_df
            del days_last_event_df
            #==================================notes type categorical features processing end========================        

            log_message(message_type='info', message=f'Notes - Processing word vectors.')

            # choose the NLP model
            nlp_model = self.load_model()
            # get segregated notes and models as dictionary
            embedding_models = nlp_model.get_embedding_models()
            # Loop through embedding model types
            for model in embedding_models:
                model_name = model['name']
                notes_df = model['notes']
                log_message(
                    message_type='info', 
                    message=f'Notes - Start processing using {model_name} model.', 
                    notes_df_shape = notes_df.shape
                )
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
                log_message(
                    message_type='info', 
                    message=f'Notes - Created vectors out of note text.', 
                    vectors_df_shape  = vectors_df.shape
                )
                # Create unit vectors from raw vectors; make into a dataframe
                unit_vectors = (vectors_2d / np.linalg.norm(vectors_2d, axis=1)[:, None])
                unit_vectors_df = pd.DataFrame(unit_vectors)
                unit_vectors_df.columns = [f'{model_name}_unit_{n}' for n in unit_vectors_df.columns]
                log_message(
                    message_type='info', 
                    message=f'Notes - Created unit vectors out of note vectors.', 
                    unit_vectors_df_shape  = unit_vectors_df.shape
                )
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

                log_message(
                    message_type='info', 
                    message=f'Notes - Completed Concatentating vector dataframes.',
                    notes_shape  = notes.shape
                )

                # Create patient days groupby object
                notes_patient_days = notes.sort_values(['masterpatientid', 'censusdate']).groupby(
                    ['masterpatientid', 'censusdate'])
                embedding_cols = [c for c in notes.columns if c.startswith('notes_')]

                # Sum vectors per patient day
                note_aggs = notes_patient_days[embedding_cols].sum()
                log_message(
                    message_type='info', message=f'Notes - Sum vectors per patient day.',
                    note_aggs_shape = note_aggs.shape
                )
                # Create exponential weighted moving (EWM) averages for note embeddings by patient day
                note_aggs_cumulative = note_aggs.groupby('masterpatientid').apply(lambda x: x.ewm(halflife=7).mean())
                note_aggs_cumulative.columns = [c + '_ewm' for c in note_aggs_cumulative.columns]
                log_message(
                    message_type='info', 
                    message=f'Notes - Created exponential weighted moving average from note embeddings.',
                    note_aggs_shape = note_aggs.shape
                )
                
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

                log_message(message_type='info', message=f'Notes - Merging Note vectors dataframe with base dataframe.')
                # Merge with combined dataframe
                self.census_df = self.census_df.merge(note_aggs, on=['masterpatientid', 'censusdate'], how='left')
                # =============Trigger garbage collection to free-up memory ==================
                del note_aggs
                gc.collect()

            log_message(
                message_type='info', 
                message=f'Notes - forward filling empty embedding vectors.'
            )
            embedding_cols = [c for c in self.census_df.columns if c.startswith("notes_")]
            final = self.census_df.sort_values(["masterpatientid", "censusdate"])
            final[embedding_cols] = final.groupby("masterpatientid")[embedding_cols].fillna(method="ffill")

            # Ensure most patient days have embeddings (they should, after forward filling)
            # Test amount of NaN in vector columns
            threshold_value = getattr(self.clientClass(), 'get_note_embeddings_nan_threshold')()
            mean_na_count = final['notes_0'].isna().mean()
            log_message(
                message_type='info', 
                message=f'Notes - mean na count of embeddings first column', 
                mean_na_count = mean_na_count
            )
            if mean_na_count >= threshold_value:
                log_message(message_type='warning', message=f'WARNING: {mean_na_count} is less than {threshold_value}')

            log_message(
                message_type="info", 
                message="Notes - Embeddings completed.", 
                final_Dataframe_shape=final.shape
            )

            final = final.merge(final_cate_df, on=['facilityid','masterpatientid','censusdate'])

            del final_cate_df

            self.sanity_check(final)
            log_message(
                message_type = 'info', 
                message = f'Notes - notes features created.', 
                final_dataframe_shape = final.shape,
                time_taken=round(time.time()-start, 2)
            )
            return final
