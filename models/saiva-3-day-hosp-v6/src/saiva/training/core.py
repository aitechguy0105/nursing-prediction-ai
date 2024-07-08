import gc
import glob
import logging
import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import optuna.integration.lightgbm as olgb
from lightgbm import early_stopping, log_evaluation, Dataset
import lightgbm as lgb
from omegaconf import OmegaConf
import mlflow
from mlflow import log_metric, log_param, log_artifact, set_tag
import timeit
from sklearn.linear_model import LogisticRegression

from .callbacks import save_trials
from .data_models import BaseModel
from .utils import get_facilities_from_train_data, get_date_diff
from .metrics import get_pos_neg, logloss_metric, get_recall_function, logloss_objective, run_test_set, update_thresholds
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class IdensDataset(Dataset):

    def __init__(self, *args, **kwargs):
        assert 'idens' in kwargs, "Required keyword parameter `idens` is missing"
        idens = kwargs.pop('idens')
        super().__init__(*args, **kwargs)
        self.idens = idens


def get_valid_data(valid_data, hyper_parameter_tuning):
    if hyper_parameter_tuning:
        return [valid_data]
    else:
        return None
    
    
def objective(trial, 
              train_data, 
              valid_data, 
              feval,
              log_evaluation_step, 
              early_stopping_round,
              HYPER_PARAMETER_TUNING = True):
    params = {
        'metric': 'auc',
        'verbosity': 5,
        'boosting_type': 'gbdt',
        'num_iterations':1000,
        'seed': 1,
        'feature_pre_filter': False,
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256, log=True),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-08, 10, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-08, 10, log=True),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 400, log=True)
    }

    model = lgb.train(params,
                      fobj=logloss_objective,
                      train_set=train_data, 
                      valid_sets=get_valid_data(valid_data, HYPER_PARAMETER_TUNING),
                      feval = feval,
                      callbacks=[log_evaluation(log_evaluation_step), early_stopping(early_stopping_round, first_metric_only=True)],
                    )

    preds = model.predict(valid_data.data)
    auc = roc_auc_score(valid_data.get_label(), preds)
    return auc

def check_files_exist():
    check_files = {
        './feature_group_drop_stats.txt',
        './feature_cumulative_drop_stats.txt',
        './all_null_dropped_col_names.json',
        './cate_columns.pickle',
    }
    for fname in check_files:
        assert Path.exists(Path(fname)),\
            f"At least one file that should have been generated at the previous steps is missing, namely: {fname}"
        
def mlflow_log_data_info(train_data, TRAINING_DATA, EXPERIMENT_DATES):
    pos, neg, n2p_ratio = get_pos_neg(train_data.get_label())
    facilities = get_facilities_from_train_data(train_data.idens)

    log_param('01_TRAINING_DATA', TRAINING_DATA)
    # MLflow allows only the strings of the size <= 250 characters
    all_facilities = ', '.join(facilities)
    for i, start in enumerate(range(0, len(all_facilities), 250)):
        key = '02_FACILITIES'
        if i > 0:
            key += f'({i + 1})'
        log_param(key, all_facilities[start:start + 250])
    log_param('03_FACILITIES_COUNT', len(facilities))
    log_param('04_ALL_FEATURE_COUNT', train_data.data.shape[1])
    log_param('05_TRAIN_START_DATE', EXPERIMENT_DATES['train_start_date'])
    log_param('06_TRAIN_END_DATE', EXPERIMENT_DATES['train_end_date'])
    log_param('07_TRAIN_DURATION_days', get_date_diff(EXPERIMENT_DATES['train_start_date'], EXPERIMENT_DATES['train_end_date']))
    log_param('08_TRAIN_POS_COUNT', pos)
    log_param('09_TRAIN_NEG_COUNT', neg)
    log_param('10_TRAIN_N2P_RATIO', n2p_ratio)

def mlflow_log_artifacts():
    file_names = [
        'feature_group_drop_stats.txt',  #generated in 05 notebook
        'feature_cumulative_drop_stats.txt', #generated in 05 notebook
        'all_null_dropped_col_names.json', #generated in 05 notebook
        'cate_columns.pickle', #generated in 06 notebook
        'performance_train_base.csv', 
        'performance_valid_base.csv',
        'performance_test_base.csv',
        'model_config.json',
        'input_features.csv',
    ]
    for file in file_names:
        try:
            log_artifact(f'./{file}')
        except:
            print(f'File {file} does not exist')

def get_feval(use_all_metrics):
    if use_all_metrics:
        feval = [logloss_metric, 
                get_recall_function('recall_overall'),
                get_recall_function('recall(LOS>30)', 'LFS', np.greater, 30),
                get_recall_function('recall_short_term', 'long_short_term', np.equal, 'short'),
                get_recall_function('recall_long_term', 'long_short_term', np.equal, 'long')]
        log_param('feval', ['recall_overall', 'recall(LOS>30)', 'recall_short_term', 'recall_long_term'])
    else:
        feval = [logloss_metric]
        log_param('feval', ['logloss_metric'])
    return feval

def recalibration(valid_data, base_model):
    if base_model.config['train_model'].get('recalibration')  is None:
        pass
    elif base_model.config['train_model'].get('recalibration').lower() == 'platt':
        raw_scores = base_model.model.predict(valid_data.data).reshape((-1,1))
        assert raw_scores.min() < 0, "The logit values are expected, the ones scaled 0-1 received"
        recalibrator = LogisticRegression(penalty=None)
        recalibrator.fit(raw_scores, valid_data.get_label())
        base_model.recalibration = {
            'type': 'platt',
            'coef': recalibrator.coef_[0][0],
            'intercept': recalibrator.intercept_[0]
        }
    else:
        raise NotImplementedError("Only Platt recalibration is supported")
        

def train_optuna_integration(
    params,
    train_data,
    valid_data,
    test_data,
    vector_model,
    MODEL_TYPE,
    EXPERIMENT_DATES,
    HYPER_PARAMETER_TUNING,
    TRAINING_DATA,
    SELECTED_MODEL_VERSION,
    MODEL_DESCRIPTION,
    EXPERIMENT_ID,
    OPTUNA_TIME_BUDGET=172800,
    pandas_categorical=None,
    model_config=None,
    use_all_metrics = False,
    log_evaluation_step = 100,
    early_stopping_round = 100
) -> BaseModel:
    
    check_files_exist()

    print(f'Training LGB models with parameter: {params}')
    with mlflow.start_run():
        start_time = timeit.default_timer()
        run_uuid = mlflow.active_run().info.run_uuid
        mlflow_log_data_info(train_data, TRAINING_DATA, EXPERIMENT_DATES)

        for param in params:
            log_param(f'hp__{param}', params[param])

        set_tag('model', 'lgb')
        log.info("=============================Training started...=============================")

        # redirectly output to text file has been deprecated, due to stdout/stderr separation
        valid_data.idens = update_thresholds(valid_data.idens, model_config.train_model.cutoff)
        feval = get_feval(use_all_metrics)
        
        log_param('fobj', logloss_objective.__name__)

        log_param('early_stopping_round', early_stopping_round)

        log_param('training method', 'optuna integration')

        model = olgb.train(params, 
                           train_set=train_data, 
                           valid_sets=get_valid_data(valid_data, HYPER_PARAMETER_TUNING),
                           feval = feval,
                           fobj=logloss_objective,
                           callbacks=[log_evaluation(log_evaluation_step), early_stopping(early_stopping_round, first_metric_only=True)],
                           optuna_callbacks=[save_trials],
                           optuna_seed = 1, 
                           time_budget=OPTUNA_TIME_BUDGET)
        if not pandas_categorical is None:
            model.pandas_categorical = pandas_categorical
        log.info("=============================Training completed...=============================")
        t_sec = round(timeit.default_timer() - start_time)
        log_param('01_RUN_TIME_sec', f'{t_sec}')
        gc.collect()

        # record parameters:
        relevant_params = [
            key for key in list(model.params.keys()) if ('categorical_column' not in key) and ('early_stopping_round' not in key)
        ]
        for param_name in relevant_params:
            log_param(param_name, model.params[param_name])

        inference_time_start = timeit.default_timer()

        ## running metrics against training
        run_test_set(
            model,
            run_uuid,
            run_uuid,
            EXPERIMENT_DATES['train_start_date'],
            EXPERIMENT_DATES['train_end_date'],
            train_data.data,
            train_data.get_label(),
            train_data.idens,
            MODEL_TYPE,
            SELECTED_MODEL_VERSION,
            dataset='TRAIN',
            threshold=model_config.train_model.cutoff
        )
        inference_time_train = round(timeit.default_timer() - inference_time_start)
        log_param('01_INFERENCE_TIME_TRAIN_sec', f'{inference_time_train}')

        ## running metrics against validation
        valid_total_aucroc, _, _, _, _, _ = run_test_set(
            model,
            run_uuid,
            run_uuid,
            EXPERIMENT_DATES['validation_start_date'],
            EXPERIMENT_DATES['validation_end_date'],
            valid_data.data,
            valid_data.get_label(),
            valid_data.idens,
            MODEL_TYPE,
            SELECTED_MODEL_VERSION,
            dataset='VALID',
            threshold=model_config.train_model.cutoff
        )

        inference_time_valid = round(timeit.default_timer() - inference_time_train)
        log_param('01_INFERENCE_TIME_VALID_sec', f'{inference_time_valid}')
        log_param('p__best_score', round(model.best_score['valid_0']['logloss'], 3))
        log_param('p__best_iteration', model.best_iteration)

        ## running metrics against test
        test_total_aucroc, test_recall, _, _, _, _ = run_test_set(
            model,
            run_uuid,
            run_uuid,
            EXPERIMENT_DATES['test_start_date'],
            EXPERIMENT_DATES['test_end_date'],
            test_data.data,
            test_data.get_label(),
            test_data.idens,
            MODEL_TYPE,
            SELECTED_MODEL_VERSION,
            dataset='TEST',
            threshold=model_config.train_model.cutoff
        )

        inference_time_test = round(timeit.default_timer() - inference_time_valid)
        log_param('01_INFERENCE_TIME_TEST_sec', f'{inference_time_test}')

        metadata = {
            'modelid': run_uuid,
            'dayspredictionvalid': 3,
            'model_algo': 'lgbm',
            'predictiontask': MODEL_TYPE,
            'modeldescription': MODEL_DESCRIPTION,
            'client_data_trained_on': TRAINING_DATA,
            'vector_model': vector_model,
            'model_s3_folder': EXPERIMENT_ID,
            'prospectivedatestart': f'{datetime.now().date()}',
            'training_start_date': f"{EXPERIMENT_DATES['train_start_date']}",
            'training_end_date': f"{EXPERIMENT_DATES['train_end_date']}",
            'valid_start_date': f"{EXPERIMENT_DATES['validation_start_date']}",
            'valid_end_date': f"{EXPERIMENT_DATES['validation_end_date']}",
            'test_start_date': f"{EXPERIMENT_DATES['test_start_date']}",
            'test_end_date': f"{EXPERIMENT_DATES['test_end_date']}",
            'valid_auc': str(valid_total_aucroc),
            'test_auc': str(test_total_aucroc),
            'test_recall_at_rank_15': str(test_recall),
        }
        model_config.metadata = metadata

        #        base_models.append(model_config)

        # ================= Save model related artifacts =========================
        # TODO: Add datacard support
        #         log_artifact(f'distribution_{CLIENT}.png') # log the distribution graph generated in 06a notebook

        # ================= Save feature drop related artifacts =========================
        with open('./model_config.json', 'w') as outfile:
            json.dump(metadata, outfile)
            
        base_model = BaseModel(model_name=run_uuid, model_type='lgb', model=model, config=OmegaConf.to_object(model_config))

        # Recalibration block
        recalibration(valid_data, base_model)

        with open(f'./{run_uuid}.pickle', 'wb') as f:
            pickle.dump(base_model, f)
        log_artifact(f'./{run_uuid}.pickle')

        input_features = pd.DataFrame(model.feature_name(), columns=['feature'])
        input_features.to_csv('./input_features.csv', index=False)

        mlflow_log_artifacts()

        # =============== Save the code used to training in S3 ======================
        # for notebook in list(Path('/src/saiva/notebooks').glob('0*.ipynb')):
        #     log_artifact(str(notebook))
        #
        # for shared_code in list(Path('/src/saiva/shared').glob('*.py')):
        #     log_artifact(str(shared_code))
        #
        for conf in list(glob.glob('/src/saiva/conf/training', recursive=True)):
            log_artifact(str(conf))

    return base_model



def train_optuna_pure_lgbm_model(
    params,
    train_data,
    valid_data,
    test_data,
    vector_model,
    MODEL_TYPE,
    EXPERIMENT_DATES,
    HYPER_PARAMETER_TUNING,
    TRAINING_DATA,
    SELECTED_MODEL_VERSION,
    MODEL_DESCRIPTION,
    EXPERIMENT_ID,
    OPTUNA_TIME_BUDGET=172800,
    pandas_categorical=None,
    model_config=None,
    use_all_metrics = False,
    log_evaluation_step = 100,
    early_stopping_round = 100,
    n_trials = 67,
) -> BaseModel:
    
    check_files_exist()

    print(f'Training LGB models with parameter: {params}')
    start_time = timeit.default_timer()

    with mlflow.start_run():
        run_uuid = mlflow.active_run().info.run_uuid
        mlflow_log_data_info(train_data, TRAINING_DATA, EXPERIMENT_DATES)

        for param in params:
            log_param(f'hp__{param}', params[param])

        set_tag('model', 'lgb')
        log.info("=============================Training started...=============================")

        # redirectly output to text file has been deprecated, due to stdout/stderr separation
        valid_data.idens = update_thresholds(valid_data.idens, model_config.train_model.cutoff)

        feval = get_feval(use_all_metrics)
        fobj = logloss_objective
        log_param('fobj', fobj.__name__)
        
        log_param('early_stopping_round', early_stopping_round)

        log_param('training method', 'optuna pure lgbm')

        optimize_obj = lambda trial: objective(trial,
                                               train_data,
                                               valid_data,
                                               feval,
                                               log_evaluation_step,
                                               early_stopping_round
                                               )
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=1))
        study.optimize(optimize_obj, n_trials=n_trials, timeout=OPTUNA_TIME_BUDGET, callbacks=[save_trials])
            
        best_params = study.best_trial.params
        best_params['objective'] = fobj
        best_params['metric'] = 'auc'
        best_params['verbosity'] = 5
        best_params['boosting_type'] = 'gbdt'
        best_params['seed'] = 1,
        best_params['num_iterations'] = 1000
        print('Best trial:', best_params)

        # Train the final model with the best parameters
        model = lgb.train(best_params,
                            fobj=fobj,
                            train_set=train_data, 
                            valid_sets=get_valid_data(valid_data, HYPER_PARAMETER_TUNING),
                            feval = feval,
                            callbacks=[log_evaluation(log_evaluation_step), early_stopping(early_stopping_round, first_metric_only=True)],
                            )
        
        if not pandas_categorical is None:
            model.pandas_categorical = pandas_categorical
        log.info("=============================Training completed...=============================")
        t_sec = round(timeit.default_timer() - start_time)
        log_param('01_RUN_TIME_sec', f'{t_sec}')
        gc.collect()

        # record parameters:
        relevant_params = [
            key for key in list(model.params.keys()) if ('categorical_column' not in key) and ('early_stopping_round' not in key)
        ]
        for param_name in relevant_params:
            log_param(param_name, model.params[param_name])

        inference_time_start = timeit.default_timer()

        ## running metrics against training
        run_test_set(
            model,
            run_uuid,
            run_uuid,
            EXPERIMENT_DATES['train_start_date'],
            EXPERIMENT_DATES['train_end_date'],
            train_data.data,
            train_data.get_label(),
            train_data.idens,
            MODEL_TYPE,
            SELECTED_MODEL_VERSION,
            dataset='TRAIN',
            threshold=model_config.train_model.cutoff
        )
        inference_time_train = round(timeit.default_timer() - inference_time_start)
        log_param('01_INFERENCE_TIME_TRAIN_sec', f'{inference_time_train}')

        ## running metrics against validation
        valid_total_aucroc, _, _, _, _, _ = run_test_set(
            model,
            run_uuid,
            run_uuid,
            EXPERIMENT_DATES['validation_start_date'],
            EXPERIMENT_DATES['validation_end_date'],
            valid_data.data,
            valid_data.get_label(),
            valid_data.idens,
            MODEL_TYPE,
            SELECTED_MODEL_VERSION,
            dataset='VALID',
            threshold=model_config.train_model.cutoff
        )

        inference_time_valid = round(timeit.default_timer() - inference_time_train)
        log_param('01_INFERENCE_TIME_VALID_sec', f'{inference_time_valid}')
        log_param('p__best_score', round(model.best_score['valid_0']['logloss'], 3))
        log_param('p__best_iteration', model.best_iteration)

        ## running metrics against test
        test_total_aucroc, test_recall, _, _, _, _ = run_test_set(
            model,
            run_uuid,
            run_uuid,
            EXPERIMENT_DATES['test_start_date'],
            EXPERIMENT_DATES['test_end_date'],
            test_data.data,
            test_data.get_label(), 
            test_data.idens,
            MODEL_TYPE,
            SELECTED_MODEL_VERSION,
            dataset='TEST',
            threshold=model_config.train_model.cutoff
        )

        inference_time_test = round(timeit.default_timer() - inference_time_valid)
        log_param('01_INFERENCE_TIME_TEST_sec', f'{inference_time_test}')

        metadata = {
            'modelid': run_uuid,
            'dayspredictionvalid': 3,
            'model_algo': 'lgbm',
            'predictiontask': MODEL_TYPE,
            'modeldescription': MODEL_DESCRIPTION,
            'client_data_trained_on': TRAINING_DATA,
            'vector_model': vector_model,
            'model_s3_folder': EXPERIMENT_ID,
            'prospectivedatestart': f'{datetime.now().date()}',
            'training_start_date': f"{EXPERIMENT_DATES['train_start_date']}",
            'training_end_date': f"{EXPERIMENT_DATES['train_end_date']}",
            'valid_start_date': f"{EXPERIMENT_DATES['validation_start_date']}",
            'valid_end_date': f"{EXPERIMENT_DATES['validation_end_date']}",
            'test_start_date': f"{EXPERIMENT_DATES['test_start_date']}",
            'test_end_date': f"{EXPERIMENT_DATES['test_end_date']}",
            'valid_auc': str(valid_total_aucroc),
            'test_auc': str(test_total_aucroc),
            'test_recall_at_rank_15': str(test_recall),
        }
        model_config.metadata = metadata

        with open('./model_config.json', 'w') as outfile:
            json.dump(metadata, outfile) 

        base_model = BaseModel(model_name=run_uuid, model_type='lgb', model=model, config=OmegaConf.to_object(model_config))

        # Recalibration block
        recalibration(valid_data, base_model)

        with open(f'./{run_uuid}.pickle', 'wb') as f:
            pickle.dump(base_model, f)
        log_artifact(f'./{run_uuid}.pickle')

        input_features = pd.DataFrame(model.feature_name(), columns=['feature'])
        input_features.to_csv('./input_features.csv', index=False)
        
        # ================= Save feature drop related and performance artifacts =========================
        mlflow_log_artifacts()

        for conf in list(glob.glob('/src/saiva/conf/training', recursive=True)):
            log_artifact(str(conf))

    return base_model
