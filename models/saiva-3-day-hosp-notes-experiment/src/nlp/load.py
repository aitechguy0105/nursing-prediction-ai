"""
- NlpFeatureModel class is used to download topic-model & NER model
from S3.
- This class loads these models for any given consumer.
- The `load_ner_model` method will return a nlp object
which will have negation added in pipline based on parameter.
"""

import os
import pickle
import subprocess

import pandas as pd
import spacy
from eliot import start_action
from negspacy.negation import Negex
from negspacy.termsets import en_clinical
from s3fs.core import S3FileSystem

preceding_negations = ['0', 'assessment', 'consent', 'decrease', 'decreased', 'decreasing', 'deny',
                       'mild', 'mildly', 'minimal', 'n/c', 'neg', 'negative', 'neither', 'no s/s',
                       'no sign', 'free from', 'without',
                       'no signs', 'nor', 'perform', 'precaution', 'precautions', 'prevent',
                       'prevention', 'will consider', 'decrease',
                       'quarantine', 'recovered', 'restriction', 'restrictions',
                       'retesting', 'had',
                       'screen', 'slight', 'swabbed', 'vac', 'w/o sign', 'w/o signs', 'w/o',
                       'worsening', 'zero', "zero", "deny", "denies", "denied", "no", "not", "none", "no one", "nobody",
                       "nothing",
                       "neither", "advised",
                       "nowhere", "never", "hardly", "scarcely", "barely", "doesn't", "isn't", "wasn't",
                       "shouldn't", "in case",
                       "wouldn't", "couldn't", "won't", "can't", "don't", "0", "past history", "past"]

following_negations = ['assessment', 'care', 'cares', 'consent', 'crisis', 'decrease', 'decreased',
                       'decreasing', 'family',
                       'guidelines', 'mild', 'mildly', 'minimal', 'neg', 'negative', 'none', 'note',
                       'outbreak', 'loss',
                       'pandemic', 'perform', 'precaution', 'precautions', 'prevent', 'prevention',
                       'protocol',
                       'quarantine', 'recovered', 'regulations', 'restriction', 'restrictions',
                       'retesting',
                       'screen', 'swabbed', 'test', 'testing', 'vac'
                       ]

preceding_negations += en_clinical['preceding_negations']
following_negations += en_clinical['following_negations']
S3FS = S3FileSystem()


class NlpFeatureModel(object):
    """ This class is singleton class
    """
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def _download_ner_model(self):
        with start_action(action_type='download_nlp_models'):
            subprocess.run(
                f'aws s3 sync s3://saiva-models/en_saiva_nlp/en_saiva_nlp_model_feature/  ./en_saiva_nlp_model_feature',
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def _download_topic_model(self, name):
        with start_action(action_type='download_nlp_models'):
            subprocess.run(
                f'aws s3 sync s3://saiva-models/en_saiva_nlp/{name}_topic_model/  ./{name}_topic_model',
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def load_topic_model(self, name):
        self._download_topic_model(name)
        with open(f'./{name}_topic_model/topic_model.pickle', 'rb') as f: topic_model = pickle.load(f)
        with open(f'./{name}_topic_model/dictionary.pickle', 'rb') as f: dictionary = pickle.load(f)
        return topic_model, dictionary

    def load_ner_model(self):
        self._download_ner_model()
        nlp = spacy.load(f'./en_saiva_nlp_model_feature')
        negex = Negex(nlp,
                      language='en_clinical_sensitive',
                      preceding_negations=preceding_negations,
                      following_negations=following_negations
                      )

        nlp.add_pipe(negex)
        return nlp

    def get_ner_keywords(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(f'{base_path}/keywords/model_feature_keywords.csv')
        return df
