""""
This script is used when ever we want to retrain the model
with new keywords by updating the ./keywords/model_feature_keywords.csv file.
Once the model is bundled, its stored in S3.

python build_ner_model.py --version model_feature
"""

import re
import shutil
import string
import subprocess
import sys

import fire
import pandas as pd
from eliot import log_message
from eliot import to_file
from s3fs import S3FileSystem
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

to_file(sys.stdout)  # ECS containers log stdout to CloudWatch

S3FS = S3FileSystem()


class NERModel(object):
    def __init__(self, version):
        self.nlp = None
        self.version = version

    def _load_keywords(self):
        """
        Load keywords.csv
        """
        return pd.read_csv(f'./keywords/{self.version}_keywords.csv')

    def _get_entity_pattern(self, keywords_df):
        entity_pattern = []
        for index, row in keywords_df.iterrows():
            # Split based on space or punctuation
            pattern_list = [{'LOWER': word.lower()} for word in
                            re.findall(f"[\w]+|[{string.punctuation}]", row['pattern'])]
            entity_pattern.append({'label': row['label'], 'pattern': pattern_list})

        return entity_pattern

    def _load_nlp_model(self, entity_pattern):
        self.nlp = English()

        # Add sentencizer
        sentencizer = self.nlp.create_pipe("sentencizer")
        sentencizer.punct_chars.add('\n')
        self.nlp.add_pipe(sentencizer)

        # Add EntityRuler - adding labels and patterns
        ruler = EntityRuler(self.nlp)
        ruler.add_patterns(entity_pattern)
        self.nlp.add_pipe(ruler)

    def _save_nlp_object(self):
        local_model_path = f'./en_saiva_nlp_{self.version}'
        self.nlp.to_disk(local_model_path)

        log_message(
            message_type='info',
            result=f'Successfully Saved in local file system',
        )

        s3_path = f's3://saiva-models/en_saiva_nlp/en_saiva_nlp_{self.version}/'

        subprocess.run(
            f'aws s3 cp {local_model_path} {s3_path} --recursive',
            shell=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

        log_message(
            message_type='info',
            result=f'Successfully Saved into S3!!',
        )
        shutil.rmtree(local_model_path)
        log_message(
            message_type='info',
            result=f'Removed local folder..',
        )

    def execute(self):
        keywords_df = self._load_keywords()
        entity_pattern = self._get_entity_pattern(keywords_df)
        self._load_nlp_model(entity_pattern)
        self._save_nlp_object()


def main(version='highlight'):
    nlp = NERModel(version)
    return nlp.execute()


if __name__ == '__main__':
    fire.Fire(main)
