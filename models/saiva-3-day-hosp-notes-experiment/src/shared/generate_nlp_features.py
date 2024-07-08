""" 
- If self.notes_df has more than 5 lk records
split the df by half and run multiprocessing on each seperatly.
Multiprocessing has payload size which cant handle large arrays hence
breaking the large dataframes to smaller size.
- If there are 60+ lk records set the multiprocessing splits to p+100
"""

import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from datetime import timedelta

from eliot import log_message

from nlp.load import NlpFeatureModel
from nlp.utils import topic_model_preprocess
import nltk

sys.path.insert(0, '/src')


def preprocess_nlp_data(notes_df):
    """
    Group the data as per patient per day 1 row of notes
    """
    # Filter for progressnotetype
    #         note_types = ['Alert Note', 'Behavior Charting','eMAR-Medication Administration Note',
    #                       'General Note','Physician Note','Pain Monitoring']
    #         self.notes_df = self.notes_df[self.notes_df['progressnotetype'].isin(note_types)]

    log_message(message_type='info', message='clean_data started ..', shape=notes_df.shape)

    notes_df = notes_df[notes_df['notetext'].notna()]
    # Remove timestamp
    notes_df['censusdate'] = notes_df['createddate'].dt.normalize()

    # club all notes per day per patient one record
    notes_df['notetext'] = notes_df.groupby(
        ["masterpatientid", "facilityid", "censusdate"]
    )['notetext'].transform(lambda x: '. '.join(x))

    # Drop all duplicates after clubbing
    notes_df.drop_duplicates(
        subset=["masterpatientid", "facilityid", "censusdate", "notetext"],
        keep='last',
        inplace=True
    )
    notes_df = notes_df.reset_index()

    return notes_df


class NERFeature(object):
    def __init__(self, notes_df):
        self.notes_df = notes_df
        nlp_model = NlpFeatureModel()
        self.nlp = nlp_model.load_ner_model()

    def _load_keywords(self):
        """
        Load keywords.csv
        """
        log_message(message_type='info', message='Load Keywords...')
        nlp_model = NlpFeatureModel()
        return nlp_model.get_ner_keywords()

    def _create_dummy_features(self, keywords_df):
        """
        Initialise new feature columns with 0
        """
        log_message(message_type='info', message='Create dummy features..')
        for index, row in keywords_df.iterrows():
            self.notes_df[f'note__{row["label"]}'] = 0

    def _process_notes(self, notes):
        for index, row in notes.iterrows():
            doc = self.nlp(row['notetext'])
            for e in doc.ents:
                # TODO: check int(not e._.negex)
                notes.loc[index, f'note__{e.label_}'] += int(not e._.negex)

        return notes

    def _handle_large_df(self):
        log_message(message_type='info', message='Handle Large dataframes..', shape=self.notes_df.shape[0])
        if self.notes_df.shape[0] > 500000:
            index_50 = round((self.notes_df.shape[0] * 50) / 100)
            notes_df1 = self.notes_df.iloc[:index_50]
            notes_df2 = self.notes_df.iloc[index_50:]
            log_message(message_type='info',
                        message='*******************Dataframe cut into 2*************',
                        shape=self.notes_df.shape,
                        shape1=notes_df1.shape,
                        shape2=notes_df2.shape
                        )
            notes_df1 = self._async_process_notes(notes_df1)
            notes_df2 = self._async_process_notes(notes_df2)
            self.notes_df = pd.concat([notes_df1, notes_df2], axis=0)
            log_message(message_type='info',
                        message='*******************Dataframe after merging *************',
                        shape=self.notes_df.shape
                        )
        else:
            self.notes_df = self._async_process_notes(self.notes_df)

    def _async_process_notes(self, notes_df):
        log_message(message_type='info', message='Process the notes..')
        p = (2 * os.cpu_count()) + 1
        process_pool = Pool(processes=p)
        split_dfs = np.array_split(notes_df, min((p + 100), len(notes_df)))
        pool_results = process_pool.map(self._process_notes, split_dfs)
        process_pool.close()
        process_pool.join()
        # combining the outputs of all processes.
        notes_df = pd.concat(pool_results, axis=0)
        log_message(message_type='info', message='_async_process_notes completed')
        return notes_df

    def _clean_columns(self):
        log_message(message_type='info', message='Clean columns..')
        # Filter for only columns needed
        note_cols = [col for col in self.notes_df.columns if col.startswith('note__')] + ['masterpatientid',
                                                                                          'censusdate']
        self.notes_df = self.notes_df[note_cols]

        # Reduce data size by converting data-type to int16
        self.notes_df.loc[:, self.notes_df.select_dtypes(include=['int64']).columns] = self.notes_df.select_dtypes(
            include=['int64']
        ).astype('int16')

    def _merge_with_base(self, base_df):
        """
        Merge has to happen before we cumsum. Merging with base will add all census dates
        from start till end
        """
        log_message(message_type='info', message='Merge with base..')

        df = base_df.merge(
            self.notes_df,
            on=['masterpatientid', 'censusdate'],
            how='left'
        )

        # After merge there might NaN values. Fill all NAN with 0 for all notes related columns
        cols = [cl for cl in df.columns if 'note__' in cl]
        for col in cols:
            df[col] = df[col].fillna(0)

        log_message(message_type='info', message='_merge_with_base completed')

        return df, cols

    def _generate_cumsum_features(self, df, cols):
        log_message(message_type='info', message='Generate CumSum columns..')

        df.sort_values(by=['masterpatientid', 'censusdate'], inplace=True)

        # cumulative summation for different durations
        _df30 = df.groupby('masterpatientid')[cols].rolling(30, min_periods=1).sum().reset_index(0, drop=True)
        _df_all = df.groupby('masterpatientid')[cols].cumsum()

        _df30.columns = 'cumsum_30_day' + _df30.columns
        _df_all.columns = 'cumsum_all_' + _df_all.columns

        # Drop original notes columns
        df.drop(cols, axis=1, inplace=True)

        log_message(message_type='info', message='cumsum features generated..')
        # Concat columns since both have same index
        df = pd.concat(
            [df, _df30, _df_all], axis=1
        )
        return df

    def execute(self, base_df):
        keywords_df = self._load_keywords()
        self._create_dummy_features(keywords_df)
        self._handle_large_df()
        self._clean_columns()
        df, cols = self._merge_with_base(base_df)
        df = self._generate_cumsum_features(df, cols)
        return df


class TopicModelFeature(object):
    def __init__(self, notes_df, name):
        self.notes_df = notes_df
        self.name = name
        self.lda_tfidf_model, self.dictionary = self.load_lda_tfidf_model()
        nltk.download('wordnet')

    def load_lda_tfidf_model(self):
        """ Set default model as avante_topic_model.
        If there is no client specific model then use avante_topic_model
        """
        log_message(message_type='info', message='Load LDA Model..')
        # TODO: change this config when new topic models are trained
        if self.name not in ['avante', 'trio']:
            self.name = 'avante'
        nlp_model = NlpFeatureModel()
        return nlp_model.load_topic_model(name=self.name)

    def filter_notes(self, prediction_date):
        # Filter for Notes that were created in last 7 days
        last_7th_date = np.datetime64(pd.to_datetime(prediction_date) - timedelta(days=7))
        self.notes_df = self.notes_df[
            self.notes_df.censusdate > last_7th_date
            ]

    def _process_row(self, row):
        bow_vector = self.dictionary.doc2bow(topic_model_preprocess(row['notetext']))
        for index, score in sorted(self.lda_tfidf_model[bow_vector], key=lambda tup: -1 * tup[1]):
            row[f'topic-{index}'] = score

        return row

    def _process_df(self, notes_df):
        return notes_df.apply(self._process_row, axis=1)

    def _async_process_notes(self, notes_df):
        p = (2 * os.cpu_count()) + 1
        process_pool = Pool(processes=p)
        split_dfs = np.array_split(notes_df, min((p + 100), len(notes_df)))
        pool_results = process_pool.map(self._process_df, split_dfs)
        process_pool.close()
        process_pool.join()
        # combining the outputs of all processes.
        notes_df = pd.concat(pool_results, axis=0)
        return notes_df

    def execute(self, base_df):
        log_message(message_type='info', message='Topic Model Async Process dataframe rows..')

        if self.notes_df.shape[0] > 500000:
            index_50 = round((self.notes_df.shape[0] * 50) / 100)
            notes_df1 = self.notes_df.iloc[:index_50]
            notes_df2 = self.notes_df.iloc[index_50:]
            log_message(message_type='info',
                        message='*******************Dataframe cut into 2*************',
                        shape=self.notes_df.shape,
                        shape1=notes_df1.shape,
                        shape2=notes_df2.shape
                        )
            notes_df1 = self._async_process_notes(notes_df1)
            notes_df2 = self._async_process_notes(notes_df2)
            self.notes_df = pd.concat([notes_df1, notes_df2], axis=0)
            log_message(message_type='info',
                        message='*******************Dataframe after merging *************',
                        shape=self.notes_df.shape
                        )
        else:
            self.notes_df = self._async_process_notes(self.notes_df)

        self.notes_df = self.notes_df.drop(
            ['createddate', 'facilityid', 'index', 'notetext',
             'notetextorder', 'patientid', 'progressnoteid', 'progressnotetype',
             'section', 'sectionsequence'],
            axis=1
        )
        if 'client' in self.notes_df.columns:
            self.notes_df = self.notes_df.drop(['client'], axis=1)

        # Fill all NaN's
        self.notes_df = self.notes_df.fillna(0)
        topic_cols = [col for col in self.notes_df.columns if 'topic-' in col]

        base_df = base_df.merge(
            self.notes_df,
            on=['masterpatientid', 'censusdate'],
            how='left'
        )
        # After merge there might NaN values. Fill all NAN with 0 for all topic related columns
        for col in topic_cols:
            base_df[col] = base_df[col].fillna(0)

        # Get mean rolling window of 7 days
        base_df.sort_values(by=['masterpatientid', 'censusdate'], inplace=True)
        log_message(message_type='info', message='Apply 7 day rolling window..')
        base_df[topic_cols] = (
            base_df.groupby('masterpatientid')[topic_cols]
                .rolling(7, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
        )

        return base_df
