"""
Run Command :
python /src/run_model.py --client infinity-benchmark --facilityids '[37]' --prediction-date 2020-05-02 --s3-bucket saiva-dev-data-bucket --replace_existing_predictions True --save-outputs-in-local True --local-folder /data/test
"""
import os
import pickle
import pickle as pkl
import subprocess
import sys

import fire
import mmh3
import numpy as np
import pandas as pd
import scipy
from eliot import start_action, start_task, to_file, log_message
from scipy.sparse import csr_matrix
from shared.featurizers import Featurizer1, Featurizer2
from shared.patient_stays import get_patient_stays

from shared import utils
from shared.constants import CACHE_STRATEGY_OPTIONS
from shared.constants import CHECK_CACHE_FIRST
from shared.constants import MODELS
from shared.database import DbEngine
from shared.load_raw_data import fetch_prediction_data
from shared.utils import get_client_class
from data_models import BaseModel
from explanations import generate_explanations

to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


class RunPredictions(object):
    def __init__(self):
        self.client = None
        self.s3_bucket = None
        self.training_start_date = None
        self.prediction_date = None
        self.facilityids = None
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
        self.time_of_day = "07:00:00"
        self.clf = None
        self.final_csr = None
        self.idens = None
        self.result_dict = None

    def __run_asserts(self):
        assert self.cache_strategy in CACHE_STRATEGY_OPTIONS, \
            f'"{self.cache_strategy}" option for cache_strategy is not valid - it should be one of {CACHE_STRATEGY_OPTIONS}'

        if self.save_outputs_in_local and self.local_folder is None:
            assert False, 'local_folder cannot be empty if save_outputs_in_local is True'

    def __create_local_directory(self):
        """
        For test purpose save intermediate outputs in local folders.
        Create the directory if it does not exists
        """
        if self.save_outputs_in_local and self.local_folder is not None:
            if not os.path.exists(self.local_folder):
                os.makedirs(self.local_folder)

    def __load_db_engines(self):
        engine = DbEngine()
        self.saiva_engine = engine.get_postgresdb_engine()
        self.client_engine = engine.get_sqldb_engine(clientdb_name=self.client)

    def download_prediction_models(self, facilityid):
        """
        Downlaod the relevant prediction models required for facility's from S3
        :param facilityid: string
        """
        with start_action(action_type='download_prediction_models'):
            modelid = MODELS[self.client][facilityid]
            s3_folder_path = MODELS[self.client]['s3_folder']
            subprocess.run(
                f'aws s3 sync s3://saiva-models/{s3_folder_path}/{modelid} /data/models/{modelid}',
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

    def __get_s3_path(self, facilityid):
        return f's3://{self.s3_bucket}/data/{self.client}/{self.prediction_date}/{facilityid}/{self.cache_s3_folder}'

    def __prep(self, feature_csr, feature_colnames, prediction_times, modelid):

        # Get all features for this model and reindex the x_frame
        all_colnames = pd.read_csv(
            f'/data/models/{modelid}/artifacts/input_features.csv'
        )

        idens = prediction_times.loc[:, ['masterpatientid', 'facilityid', 'predictiontimestamp']]
        idens.rename(
            columns={'predictiontimestamp': 'censusdate'},
            inplace=True
        )

        # Filter records for the given prediction_date
        censusdate = self.prediction_date + ' ' + self.time_of_day
        idens = idens[idens['censusdate'] == censusdate]
        idens_index = idens[idens['censusdate'] == censusdate].index
        feature_csr = feature_csr[idens_index, :]
        
        # Check training and prediction columns are same
        assert list(all_colnames.feature) == feature_colnames
        
        # Reshape & reorder the features_csr columns to match the stored input features columns
#         all_colnames = list(all_colnames.feature)
#         row = feature_csr.shape[0]
#         col = len(all_colnames)
#         # create dummy csr matrix with the required dimension
#         final_csr = csr_matrix((row, col), dtype=np.float32)
#         for to_index, col in enumerate(all_colnames):
#             if col in feature_colnames:
#                 from_index = feature_colnames.index(col)
#                 final_csr[:, to_index] = feature_csr[:, from_index]

        return feature_csr, idens

    def __save_dataframe(self, df, filename, facilityid, type='parquet'):
        """
        Save the final input dataframe & output predictions for testing
        :param df: Can be either Final X-frame or generated predictions
        :param filename: Filename to save the dataframe
        """
        if self.save_outputs_in_s3:
            final_x_s3_path = self.__get_s3_path(facilityid) + f'/{filename}.{type}'
            if type == 'parquet':
                df.to_parquet(final_x_s3_path, index=False)
            else:
                scipy.sparse.save_npz(final_x_s3_path, df)

        if self.save_outputs_in_local:
            final_x_local_path = self.local_folder + f'/{filename}.{type}'
            if type == 'parquet':
                df.to_parquet(final_x_local_path, index=False)
            else:
                scipy.sparse.save_npz(final_x_local_path, df)

    def execute(self):
        self.__run_asserts()
        self.__create_local_directory()
        self.__load_db_engines()

        clientClass = get_client_class(self.client)

        with start_task(action_type='run_model'):
            for facilityid in self.facilityids:
                self.download_prediction_models(facilityid)
                self.download_nlp_models()

                with start_action(
                        action_type='fetching_prediction_data', facilityid=facilityid
                ):
                    # ------------Fetch the data from client SQL db and store in S3 buckets----------
                    result_dict = fetch_prediction_data(
                        client_sql_engine=self.client_engine,
                        prediction_date=self.prediction_date,
                        facilityid=facilityid,
                        train_start_date=self.training_start_date,
                        client=self.client,
                        cache_strategy=self.cache_strategy,
                        s3_location_path_prefix=self.__get_s3_path(facilityid),
                        local_folder=self.local_folder,
                        save_outputs_in_local=self.save_outputs_in_local
                    )
                    result_dict['stays'] = get_patient_stays(
                        result_dict['patient_census'],
                        result_dict['patient_rehosps']
                    )

                prediction_times = utils.get_prediction_timestamps(
                    result_dict['stays'],
                    self.time_of_day
                )
                log_message(
                    message_type='info',
                    message=f"Got {len(prediction_times)} prediction times..."
                )

                modelid = MODELS[self.client][facilityid]

                with start_action(
                        action_type='feature-engineering', facilityid=facilityid
                ):
                    # ----------------------------Fetch features-----------------------
                    featurizer1 = Featurizer1(result_dict, prediction_times, modelid)
                    feature1_csr, feature1_names = featurizer1.process()

                    featurizer2 = Featurizer2(result_dict, prediction_times)
                    features_csr, feature_names = featurizer2.process(feature1_csr, feature1_names)
                    
                    # --------------------Create X-frame and Identifier columns------------------------
                    final_csr, idens = self.__prep(features_csr, feature_names, prediction_times, modelid)
                    self.__save_dataframe(final_csr, 'final_csr', facilityid, type='npz')
                    

                with start_action(action_type='load_model', facilityid=facilityid, modelid=modelid):
                    with open(f'/data/models/{modelid}/artifacts/{modelid}.pickle', 'rb') as f:
                        clf = pickle.load(f)

                with start_action(action_type='generate_predictions', facilityid=facilityid):
                    preds = clf.predict(final_csr)
                    predictions = idens
                    predictions['facilityid'] = facilityid
                    predictions['modelid'] = modelid
                    predictions['predictionvalue'] = preds
                    predictions['predictionrank'] = predictions.predictionvalue.rank(ascending=False)
                    predictions['censusdate'] = self.prediction_date
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
                            'predictionrank',
                            'modelid',
                            'client'
                        ]
                    )
                    
                    # ---Save these results in the object so that it can be used in the explanations notebook
                    # ---   it isn't needed for any other purpose
                    self.clf = clf
                    self.final_csr = final_csr
                    self.idens = idens
                    self.result_dict = result_dict

                    def control_or_test(masterpatientid, test_group_percentage):
                        """
                        Create 2 groups - test & controlled for monitoring purpose.
                        Deterministically returns whether a given masterpatientid is
                        within the test group or control group.
                        hospice patients are who wants to die in the Nursing center, hence less care given to them.
                        So we exclude hospice patients from our predictions.
                        """

                        is_exp_group = mmh3.hash(str(masterpatientid)) % 1000 <= (1000 * test_group_percentage)
                        hospice_payercodes = getattr(clientClass(), 'get_hospice_payercodes')()

                        if hospice_payercodes:
                            payer = result_dict['patient_census'].loc[
                                (result_dict['patient_census']['MasterPatientID'] == masterpatientid) &
                                (result_dict['patient_census'][
                                     'CensusDate'] == self.prediction_date), 'payercode'].array[0]
                            if payer in hospice_payercodes:
                                return False
                            else:
                                return bool(is_exp_group)
                        else:
                            return is_exp_group

                    predictions['experiment_group'] = predictions.masterpatientid.apply(
                        control_or_test,
                        test_group_percentage=self.test_group_percentage
                    )
                    predictions['experiment_group_rank'] = predictions.groupby(
                        'experiment_group'
                    ).predictionvalue.rank(ascending=False)

                    predictions = predictions.reindex(
                        columns=[
                            'masterpatientid',
                            'facilityid',
                            'bedid',
                            'censusdate',
                            'predictionvalue',
                            'predictionrank',
                            'modelid',
                            'client',
                            'experiment_group',
                            'experiment_group_rank'
                        ]
                    )
                    self.__save_dataframe(predictions, 'predictions_output', facilityid, type='parquet')

                # For a test run, don't write to Database and don't generate explanations
                if not self.test:
                    with start_action(action_type='upsert_predictions', facilityid=facilityid):
                        print('INSERT DATA..')
                        if self.replace_existing_predictions:
                            self.saiva_engine.execute(
                                f"""delete from daily_predictions where censusdate = '{self.prediction_date}'
                                and facilityid = '{facilityid}' and client = '{self.client}' and modelid = '{modelid}'"""
                            )

                        predictions.to_sql(
                            'daily_predictions',
                            self.saiva_engine,
                            method='multi',
                            if_exists='append',
                            index=False
                        )

                    with start_action(action_type='generate_explanations', facilityid=facilityid):
                        generate_explanations(
                            self.prediction_date,
                            clf,
                            final_csr,
                            idens,
                            result_dict,
                            self.client,
                            facilityid,
                            self.saiva_engine,
                            self.save_outputs_in_s3,
                            self.__get_s3_path(facilityid),
                            self.save_outputs_in_local,
                            self.local_folder
                        )

    def run_model(
            self,
            client,
            s3_bucket,
            prediction_date,
            facilityids=None,
            test=False,
            replace_existing_predictions=False,
            test_group_percentage=1.0,
            cache_s3_folder='raw',
            cache_strategy=CHECK_CACHE_FIRST,
            save_outputs_in_s3=False,
            save_outputs_in_local=False,
            local_folder=None):

        self.client = client
        # Use preconfigured training start date
        self.training_start_date = MODELS[client]['training_start_date']
        self.s3_bucket = s3_bucket
        self.prediction_date = prediction_date
        self.facilityids = facilityids
        self.test = test
        self.replace_existing_predictions = replace_existing_predictions
        self.test_group_percentage = test_group_percentage
        self.cache_s3_folder = cache_s3_folder
        self.cache_strategy = cache_strategy
        self.save_outputs_in_s3 = save_outputs_in_s3
        self.save_outputs_in_local = save_outputs_in_local
        self.local_folder = local_folder

        log_message(
            message_type='info', client=client, s3_bucket=s3_bucket,
            prediction_date=prediction_date, facilityids=facilityids, test=test,
            replace_existing_predictions=replace_existing_predictions,
            trainset_start_date=self.training_start_date, test_group_percentage=test_group_percentage,
            cache_s3_folder=cache_s3_folder, cache_strategy=cache_strategy,
            save_outputs_in_s3=save_outputs_in_s3,
            save_outputs_in_local=save_outputs_in_local, local_folder=local_folder
        )

        self.execute()


if __name__ == '__main__':
    prediction = RunPredictions()
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(prediction.run_model)
