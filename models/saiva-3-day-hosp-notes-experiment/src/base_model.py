"""
Run Command :
python /src/run_model.py --client infinity-benchmark --facilityids '[37]' --prediction-date 2020-05-02
--s3-bucket saiva-dev-data-bucket --replace_existing_predictions True
--save-outputs-in-local True --local-folder /data/test
--sub_clients : Comma seperated sub-client names for the same data feed
--group_level : Indicate whether the ranking will be done on facility/unit/floor
"""

import os
import pickle
import subprocess

import pandas as pd
from eliot import start_action, start_task, log_message

from ranking import Ranking
from shared.constants import CACHE_STRATEGY_OPTIONS
from shared.constants import CHECK_CACHE_FIRST
from shared.constants import MODELS
from shared.database import DbEngine
from shared.generate_base_features import base_feature_processing
from shared.generate_complex_features import complex_feature_processing
from shared.generate_lab_features import get_lab_features
from shared.generate_note_embeddings import generating_notes
from shared.generate_note_embeddings import processing_word_vectors
from shared.generate_nlp_features import preprocess_nlp_data, NERFeature, TopicModelFeature
from shared.utils import get_client_class


class BasePredictions(object):
    def __init__(self):
        self.client = None
        self.s3_bucket = None
        self.training_start_date = None
        self.prediction_date = None
        self.facilityids = None
        self.sub_clients = None
        self.group_level = None
        self.test = False
        self.replace_existing_predictions = False
        self.test_group_percentage = 1.0
        self.cache_s3_folder = 'raw'
        self.cache_strategy = CHECK_CACHE_FIRST
        self.save_outputs_in_s3 = False
        self.save_outputs_in_local = False
        self.local_folder = None
        self.saiva_engine = None
        self.client_engine = None
        self.modelid = None

    def _run_asserts(self):
        assert self.cache_strategy in CACHE_STRATEGY_OPTIONS, \
            f'"{self.cache_strategy}" option for cache_strategy is not valid - it should be one of {CACHE_STRATEGY_OPTIONS}'

        if self.save_outputs_in_local and self.local_folder is None:
            assert False, 'local_folder cannot be empty if save_outputs_in_local is True'

    def _create_local_directory(self):
        """
        For test purpose save intermediate outputs in local folders.
        Create the directory if it does not exists
        """
        if self.save_outputs_in_local and self.local_folder is not None:
            if not os.path.exists(self.local_folder):
                os.makedirs(self.local_folder)

    def _load_db_engines(self):
        engine = DbEngine()
        self.saiva_engine = engine.get_postgresdb_engine()
        self.client_engine = engine.get_sqldb_engine(clientdb_name=self.client)

    def download_prediction_models(self, facilityid):
        """
        Downlaod the relevant prediction models required for facility's from S3
        :param facilityid: string
        """
        with start_action(action_type='download_prediction_models'):
            s3_folder_path = MODELS[self.client]['s3_folder']
            subprocess.run(
                f'aws s3 sync s3://saiva-models/{s3_folder_path}/{self.modelid} /data/models/{self.modelid}',
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def download_nlp_models(self):
        """
        Downlaod the relevant Progress Note Embedding models from S3
        """
        with start_action(action_type='download_nlp_models'):
            s3_folder_path = MODELS[self.client]['progress_note_model_s3_folder']
            subprocess.run(
                f'aws s3 sync s3://saiva-models/{s3_folder_path} /data/models/',
                shell=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    def _get_s3_path(self, facilityid):
        return f's3://{self.s3_bucket}/data/{self.client}/{self.prediction_date}/{facilityid}/{self.cache_s3_folder}'

    def _fill_na(self, df):
        with open(f'/data/models/{self.modelid}/artifacts/na_filler.pickle', 'rb') as f:
            na_filler = pickle.load(f)

        return df.fillna(na_filler)

    def _prep(self, df, facilityid):
        drop_cols = ['censusdate', 'masterpatientid', 'facilityid', 'bedid', 'resultdate']
        drop_cols = drop_cols + [col for col in df.columns if 'target' in col]
        x_frame = (df.drop(columns=drop_cols).reset_index(drop=True).astype('float32'))
        idens = df.loc[:, ['masterpatientid', 'censusdate', 'facilityid', 'bedid']]

        # Get all features for this model and reindex the x_frame
        all_feats = pd.read_csv(
            f'/data/models/{self.modelid}/artifacts/input_features.csv'
        )
        x_frame = x_frame.reindex(columns=all_feats.feature, fill_value=0)

        self._save_dataframe(x_frame, 'finalx_output', facilityid)

        return x_frame, idens

    def _save_dataframe(self, df, filename, facilityid):
        """
        Save the final input dataframe & output predictions for testing
        :param df: Can be either Final X-frame or generated predictions
        :param filename: Filename to save the dataframe
        """
        if self.save_outputs_in_s3:
            final_x_s3_path = self._get_s3_path(facilityid) + f'/{filename}.parquet'
            df.to_parquet(final_x_s3_path, index=False)

        if self.save_outputs_in_local:
            final_x_local_path = self.local_folder + f'/{filename}.parquet'
            df.to_parquet(final_x_local_path, index=False)

    def feature_engineering(self, facilityid, result_dict, prediction_start_date=None):
        """
        prediction_start_date: When ever we do a backfill we goto do a bulk_prediction for a given
        date range that starts from prediction_start_date and ends with prediction_date
        """
        with start_task(action_type='feature-engineering', client=self.client, facilityid=facilityid):
            clientClass = get_client_class(self.client)

            # ----------------------------Base Features fetched from tables-----------------------
            combined, result_dict = base_feature_processing(
                result_dict=result_dict,
                train_start_date=self.training_start_date,
                prediction_date=self.prediction_date,
                s3_bucket=self.s3_bucket,
                s3_location_path_prefix=self._get_s3_path(facilityid),
                save_outputs_in_s3=self.save_outputs_in_s3,
                local_folder=self.local_folder,
                save_outputs_in_local=self.save_outputs_in_local,
                training=False
            )
            # ----------------------------Generate Lab Features-----------------------
            if not result_dict.get('patient_lab_results', pd.DataFrame()).empty:
                combined = get_lab_features(
                    base=combined,
                    patient_lab_results=result_dict.get('patient_lab_results'),
                    training=False
                )
            # ---------------------Complex feature which are derived from existing features--------
            final = complex_feature_processing(
                combined,
                self._get_s3_path(facilityid),
                self.save_outputs_in_s3,
                self.local_folder,
                self.save_outputs_in_local,
            )
            drop_cols = [col for col in final.columns if 'target' in col]
            final = final.drop(columns=drop_cols)
            # -----------------------------Progress Notes----------------------------------
            if not result_dict.get('patient_progress_notes', pd.DataFrame()).empty:
                notes = preprocess_nlp_data(result_dict['patient_progress_notes'])

                ner_obj = NERFeature(notes.copy())
                final = ner_obj.execute(final)

                topic_obj = TopicModelFeature(notes_df=notes, name=self.client)
                # filter notes for the given prediction_start_date or prediction_date
                topic_obj.filter_notes(prediction_start_date or self.prediction_date)
                final = topic_obj.execute(final)
                # self.download_nlp_models()
                # emar_notes, progress_notes = generating_notes(
                #     result_dict['patient_progress_notes'],
                #     clientClass
                # )
                # final = processing_word_vectors(
                #     final,
                #     emar_notes,
                #     progress_notes,
                #     clientClass
                # )

            # -----------------------------Handle empty values------------------------------
            final = self._fill_na(final)
            final['resultdate'] = final['censusdate']
            final = final.dropna()

        return final, result_dict

    def prediction(self, model, final_x, final_idens, facilityid, prediction_date):
        with start_action(action_type='generate_predictions', facilityid=facilityid, client=self.client):
            preds = model.predict(final_x)
            predictions = final_idens
            predictions['facilityid'] = facilityid
            predictions['modelid'] = self.modelid
            predictions['predictionvalue'] = preds
            predictions['censusdate'] = prediction_date
            predictions['client'] = self.client

            log_message(message_type='info', num_patients=len(pd.unique(predictions['masterpatientid'])))
            log_message(message_type='info', min_predictions=predictions['predictionvalue'].min())
            log_message(message_type='info', max_prediction=predictions['predictionvalue'].max())

            predictions = predictions.reindex(
                columns=[
                    'masterpatientid',
                    'facilityid',
                    'bedid',
                    'censusdate',
                    'predictionvalue',
                    'modelid',
                    'client'
                ]
            )

        ranking = Ranking(
            predictions=predictions,
            facilityid=facilityid,
            modelid=self.modelid,
            client=self.client,
            prediction_date=prediction_date,
            saiva_engine=self.saiva_engine,
            client_engine=self.client_engine,
            group_level=self.group_level,
            subclients=self.sub_clients,
            replace_existing_predictions=self.replace_existing_predictions,
            test=self.test
        )
        predictions = ranking.execute()
        self._save_dataframe(predictions, 'predictions_output', facilityid)
