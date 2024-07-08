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
import shutil
from pathlib import Path

import pandas as pd
from eliot import start_action, log_message

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
from shared.utils import get_client_class


def handle_delete_error(func, path, exc_info):
    """
    handler if you get exceptions when deleting model files from local disk
    """
    log_message(message_type='warning', file_not_deleted=path, exception=exc_info)


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
        self.cleanup_downloaded_models = True
        self.saiva_engine = None
        self.client_engine = None
        self.modelid = None
        self.iden_cols = ['censusdate', 'masterpatientid', 'facilityid', 'bedid',
                          'censusactioncode', 'payername', 'payercode']

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
            models_path = Path('/data/models')
            models_path.mkdir(parents=True, exist_ok=True)
            sync_command = f'aws s3 sync s3://saiva-models/{s3_folder_path}/{self.modelid} /data/models/{self.modelid}'
            os.system(sync_command)

    #             subprocess.run(
    #                 f'aws s3 sync s3://saiva-models/{s3_folder_path}/{self.modelid} /data/models/{self.modelid}',
    #                 shell=True,
    #                 stderr=subprocess.DEVNULL,
    #                 stdout=subprocess.DEVNULL,
    #             )

    def delete_downloaded_models(self):
        """
        Delete nlp models downloaded to the disk (which takes up the bulk of the space!)
        :param facilityid: string
        """
        with start_action(action_type='DELETE_DOWNLOADED_MODELS'):
            dir_path = "/data/models/"
            log_message(message_type='info', dir_path_to_delete=dir_path)
            # Delete all contents of a directory and handle errors
            shutil.rmtree(dir_path, onerror=handle_delete_error)

    def _get_s3_path(self, facilityid):
        return f's3://{self.s3_bucket}/data/{self.client}/{self.prediction_date}/{facilityid}/{self.cache_s3_folder}'

    def _fill_na(self, df):
        with open(f'/data/models/{self.modelid}/artifacts/na_filler.pickle', 'rb') as f:
            na_filler = pickle.load(f)

        return df.fillna(na_filler)

    def filter_hospice_patients(self, df):
        """
        Filter out hospice patients.
        Filter out pre-configured censusactioncodes which needs to be excluded.
        """
        self.clientClass = get_client_class(self.client)
        censusactioncodes = getattr(self.clientClass(), 'get_excluded_censusactioncodes')()
        hospice_condition = (
                df['censusactioncode'].isin(censusactioncodes) | (df['payername'].str.contains("hospice", case=False))
        )
        hospice_df = df[hospice_condition]
        final_df = df[~hospice_condition]

        if not hospice_df.empty:
            hospice_df = hospice_df[self.iden_cols]  # Retain only iden_cols for hospice_df

        return final_df, hospice_df

    def _prep(self, df, facilityid):
        drop_cols = self.iden_cols + [col for col in df.columns if 'target' in col]
        x_frame = (df.drop(columns=drop_cols).reset_index(drop=True).astype('float32'))
        idens = df.loc[:, self.iden_cols]

        # Get all features for this model and reindex the x_frame
        all_feats = pd.read_csv(
            f'/data/models/{self.modelid}/artifacts/input_features.csv'
        )
        dropped_columns = set(x_frame.columns).difference(set(all_feats.feature))
        log_message(message_type='info',
                    dropped_column_count=len(dropped_columns),
                    facilityid=facilityid,
                    client=self.client,
                    columns=f'DROPPED_COLUMNS: {dropped_columns}',
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

    def feature_engineering(self, facilityid, result_dict):
        with start_action(action_type='feature-engineering', client=self.client, facilityid=facilityid):
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
                save_outputs_in_local=self.save_outputs_in_local
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
                emar_notes, progress_notes = generating_notes(
                    result_dict['patient_progress_notes'],
                    clientClass
                )
                final = processing_word_vectors(
                    final,
                    emar_notes,
                    progress_notes,
                    self.client,
                    clientClass
                )

            # -----------------------------Handle empty values------------------------------
            final = self._fill_na(final)

            # TODO: clean this code which drops NaN. This causes issues when ever we introduce new columns
            exclude_cols = ['bedid', 'censusactioncode', 'payername', 'payercode']
            include_cols = final.columns.difference(exclude_cols)
            final = final.dropna(subset=include_cols)

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
                    'client',
                    'censusactioncode',
                    'payername',
                    'payercode'
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

    def insert_hospice_patients(self, hospice_df, prediction_date, facilityid):
        """
        Insert Hospice & other excluded censusactioncode patients into daily_prediction table
        """

        if not hospice_df.empty:
            hospice_df['censusdate'] = prediction_date
            hospice_df['client'] = self.client

            log_message(
                message_type='info',
                message=f'Delete all old facility hospice rows for the given date: {prediction_date}'
            )
            self.saiva_engine.execute(
                f"""delete from daily_predictions 
                    where censusdate = '{prediction_date}' 
                    and facilityid = {facilityid} and 
                    client = '{self.client}' and predictionrank IS NULL """
            )

            log_message(
                message_type='info',
                message=f'Save facility hospice data for the given date: {prediction_date}, {self.client}, {facilityid}'
            )
            hospice_df.to_sql(
                'daily_predictions',
                self.saiva_engine,
                method='multi',
                if_exists='append',
                index=False
            )
