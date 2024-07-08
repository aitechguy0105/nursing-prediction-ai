""""
This script is used when ever we want to retrain the Topic model
and save it directly  in S3.
TopicModel class is consumed by Jupyter Notebook train-topic-model.ipynb
"""

import pickle

import sys

import fire
import gensim

from eliot import log_message
from eliot import to_file
from gensim import models as gensim_models
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from s3fs import S3FileSystem
from nlp.utils import topic_model_preprocess

S3FS = S3FileSystem()
nltk.download('wordnet')

class TopicModel(object):
    def __init__(self, progress_notes_df, name):
        """
        self.dictionary : mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size,
        """
        self.notes_df = progress_notes_df
        self.name = name
        self.stemmer = SnowballStemmer('english')
        self.dictionary = None

    def process_notes(self):
        """ Filter out tokens that appear in
        less than 15 documents (absolute number) or
        more than 0.5 documents (fraction of total corpus size, not absolute number).
        After the above two steps, keep only the first 100000 most frequent tokens.
        For each row we create a dictionary reporting how many
        words and how many times those words appear. 
        """
        log_message(
            message_type='info',
            result=f'Process the notes....',
        )
        self.notes_df = self.notes_df[self.notes_df['notetext'].notna()]
        self.notes_df['censusdate'] = self.notes_df['createddate'].dt.normalize()
        # club all notes per day per patient one record
        self.notes_df['notetext'] = self.notes_df.groupby(
            ["masterpatientid", "facilityid", "censusdate"])['notetext'].transform(
            lambda x: '. '.join(x)
        )

        self.notes_df.drop_duplicates(
            subset=["masterpatientid", "facilityid", "censusdate", "notetext"],
            keep='last',
            inplace=True
        )
        self.notes_df = self.notes_df.reset_index()
        
        processed_docs = self.notes_df['notetext'].map(topic_model_preprocess)

        # Print a dictionary from ‘processed_docs’ containing the number of times a word appears in the training set.
        self.dictionary = gensim.corpora.Dictionary(processed_docs)

        self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

        return [self.dictionary.doc2bow(doc) for doc in processed_docs]

    def _generate_tfi_df_doc(self, bow_corpus):
        """ Apply TfidfModel on ‘bow_corpus’ and save it to ‘tfidf’ model, 
        then apply transformation to the entire corpus and call it ‘corpus_tfidf’. 
        """
        log_message(
            message_type='info',
            result=f'Apply TFI-DF model...',
        )
        tfidf = gensim_models.TfidfModel(bow_corpus)
        return tfidf[bow_corpus]

    def _generate_lda_model(self, corpus_tfidf, dictionary):
        """ Running LDA using TF-IDF 
        Generates 34 topics.
        """
        log_message(
            message_type='info',
            result=f'Generate LDA model!!',
        )
        return gensim_models.LdaMulticore(
            corpus_tfidf,
            num_topics=34,
            id2word=dictionary,
            passes=2,
            workers=6
        )

    def _save_model(self, lda_model_tfidf):
        """ Save models in S3 
        """
        with S3FS.open(f's3://saiva-models/en_saiva_nlp/{self.name}_topic_model/topic_model.pickle', 'wb') as f:
            pickle.dump(lda_model_tfidf, f, protocol=4)

        with S3FS.open(f's3://saiva-models/en_saiva_nlp/{self.name}_topic_model/dictionary.pickle', 'wb') as f:
            pickle.dump(self.dictionary, f, protocol=4)

        log_message(
            message_type='info',
            result=f'Topic Model successfully Saved into S3!!',
        )

    def execute(self):
        bow_corpus = self.process_notes()
        corpus_tfidf = self._generate_tfi_df_doc(bow_corpus)
        lda_model_tfidf = self._generate_lda_model(corpus_tfidf, self.dictionary)
        self._save_model(lda_model_tfidf)
