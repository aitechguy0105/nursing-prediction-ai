import boto3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import os 
import sys

from collections import defaultdict, namedtuple 
from multiprocessing import Pool 
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

def _bootstrapWorker(job_data): 
    Y = job_data[0]
    Yhat = job_data[1]
    ptimes = job_data[2]
    stat_funcs = job_data[3]
    indices = job_data[4]
    ptimes = ptimes.copy()

    # Take bootstrap sample of ptimes.
    N = len(ptimes)
    ptimes = ptimes.iloc[indices]
    if Yhat is None: # If no Yhat is provided, generate a new one every round... 
        Yhat = np.random.uniform(size=len(Y))
    Y = Y[indices]
    Yhat = Yhat[indices]
    stats = []
    for func_name, func in stat_funcs.items(): 
        stats.append(func(Y, Yhat, ptimes))
    return np.array(stats)

def saiva_recall_at_top_K(Y, Yhat, ptimes, K=15): 
    """
    Clean implementation of Saiva's existing formulation of 
    recall at top-K - if we take the top-K predictions in each 
    facility each day, how many of the true cases do we flag 
    at least once over the test period?  This is done on a per
    stay basis, not a per-patient basis.  
    """

    ptimes = ptimes.copy()
    ptimes = ptimes.reset_index(drop=True)
    ptimes['predictiondate'] = ptimes.predictiontimestamp.dt.date.values

    case_indices = np.nonzero(Y == 1)[0]
    case_stay_indices = ptimes.stayrowindex.iloc[case_indices].unique()

    # Keep track of hits in this set... 
    case_stay_index_hits = set()

    grouped_ptimes = ptimes.groupby(['predictiondate', 'facilityid'])


    for group_idx, group_ptimes in grouped_ptimes: 
        group_indices = group_ptimes.index.values
        group_y = Y[group_indices]
        group_yhat = Yhat[group_indices]
    
        top_k_indices = np.argsort(group_yhat)[-K:]
        for idx in top_k_indices: 
            if group_y[idx]: 
                stay_idx = group_ptimes.iloc[idx].stayrowindex
                case_stay_index_hits.add(stay_idx)

    return len(case_stay_index_hits) / len(case_stay_indices)


def recall_at_top_K(Y, Yhat, ptimes, K=15): 
    """
    Traditional per day recall at top K, with grouping
    by facility and date.  
    """

    ptimes = ptimes.copy()
    ptimes = ptimes.reset_index(drop=True)
    ptimes['predictiondate'] = ptimes.predictiontimestamp.dt.date.values

    grouped_ptimes = ptimes.groupby(['predictiondate', 'facilityid'])

    stats = []
    for group_idx, group_ptimes in grouped_ptimes: 
        group_indices = group_ptimes.index.values
        group_y = Y[group_indices]
        group_yhat = Yhat[group_indices]
        num_cases = np.sum(group_y)
        top_k_indices = np.argsort(group_yhat)[-K:]
        num_true_positives = np.sum(group_y[top_k_indices])
        if num_cases > 0: 
            recall = num_true_positives / num_cases
            stats.append(recall)
    stats = np.array(stats)
    return np.nanmean(stats)#, stats

def ppv_at_top_K(Y, Yhat, ptimes, K=15): 
    """
    Traditional per day ppv at top K, with grouping
    by facility and date.  
    """

    ptimes = ptimes.copy()
    ptimes = ptimes.reset_index(drop=True)
    ptimes['predictiondate'] = ptimes.predictiontimestamp.dt.date.values

    grouped_ptimes = ptimes.groupby(['predictiondate', 'facilityid'])

    stats = []
    for group_idx, group_ptimes in grouped_ptimes: 
        group_indices = group_ptimes.index.values
        group_y = Y[group_indices]
        group_yhat = Yhat[group_indices]
        top_k_indices = np.argsort(group_yhat)[-K:]
        num_true_positives = np.sum(group_y[top_k_indices])
        recall = num_true_positives / K
        stats.append(recall)
    stats = np.array(stats)
    return np.mean(stats)#, stats

def auroc_score(Y, Yhat, ptimes, K=None): 
    """
    Wrapper around sklearn auroc
    """
    return roc_auc_score(Y, Yhat)

def auprc_score(Y, Yhat, ptimes, K=None): 
    """
    Wrapper around sklearn mean precision
    """
    return average_precision_score(Y, Yhat)


def _sample_indices_by(ptimes, colname): 
    keys = ptimes[colname].unique()
    N = len(keys)
    keys_to_use = list(np.random.choice(keys, N, replace=True))
    key_to_indices = defaultdict(list)
    for idx, key in enumerate(ptimes[colname].values): 
        key_to_indices[key].append(idx)

    indices = []
    for key in keys_to_use: 
        indices.extend(key_to_indices[key])
    return indices


def bootstrap_confidence_intervals(Y, Yhat, ptimes, B=200, 
                                   stat_funcs=None, sample_by='patient'): 
    """
    Take bootstrap samples and calculate statistics using stat_funcs.  
    Do the samples on a per patient basis?  Or per stay basis?  Or per 
    prediction time?  The latter seems wrong...  
    Not clear which is correct...  
    """
    print(f"Estimating confidence intervals from {B} bootstrap samples...")

    # set up stat_funcs
    if stat_funcs == None: 
        stat_funcs = {'Saiva Recall at top K': saiva_recall_at_top_K, 
                      'Recall at top K': recall_at_top_K, 
                      'AUROC': auroc_score, 
                      'ave Precision': auprc_score}
        # stat_funcs = [saiva_recall_at_top_K, 
        #               recall_at_top_K, 
        #               auroc_score, 
        #               auprc_score]

    # Parallelize this... 
    print(f"Constructing samples by sampling {sample_by}...")
    N = len(ptimes)
    jobs_data = []
    for b in range(B): 
        # How should we do resampling?  By stay or by patient?...  
        if sample_by == 'stay': 
            indices = _sample_indices_by(ptimes, 'stayrowindex')
        elif sample_by == 'patient': 
            indices = _sample_indices_by(ptimes, 'masterpatientid')
        else: 
            indices = np.random.choice(N, N, replace=True)
        jobs_data.append((Y, Yhat, ptimes, stat_funcs, indices))
    
    print("Running jobs...")
    with Pool(min(os.cpu_count() - 4, 32)) as pool:
        bootstrap_stats_list = pool.map(_bootstrapWorker, jobs_data)

    print("Collating statistics")
    raw_stats = np.zeros((B, len(stat_funcs)))
    for idx, stats in enumerate(bootstrap_stats_list): 
        raw_stats[idx,:] = stats

    ci_results = np.zeros((len(stat_funcs), 3))
    for idx in range(ci_results.shape[0]): 
        lower, upper = np.quantile(raw_stats[:,idx], (0.025, 0.975))
        mean = np.mean(raw_stats[:,idx])
        ci_results[idx] = [mean, lower, upper]

    ci_results = pd.DataFrame(ci_results)
    ci_results.index = [k for k, v in stat_funcs.items()]
    ci_results.columns = ['Statistic', 'Lower CI', 'Upper CI']

    return ci_results, raw_stats


def paired_bootstrap_confidence_intervals(Y, Yhat1, Yhat2, ptimes, B=200, 
                                          stat_funcs=None, sample_by='patient'): 

    if stat_funcs == None: 
        stat_funcs = {'Saiva Recall at top K': saiva_recall_at_top_K, 
                      'Recall at top K': recall_at_top_K, 
                      'AUROC': auroc_score, 
                      'ave Precision': auprc_score}
        # stat_funcs = [saiva_recall_at_top_K, 
        #               recall_at_top_K, 
        #               auroc_score, 
        #               auprc_score]

    # Parallelize this... 
    print(f"Constructing samples by sampling {sample_by}...")
    N = len(ptimes)
    jobs_data_1, jobs_data_2 = [], []
    for b in range(B): 
        # How should we do resampling?  By stay or by patient?...  
        if sample_by == 'stay': 
            indices = _sample_indices_by(ptimes, 'stayrowindex')
        elif sample_by == 'patient': 
            indices = _sample_indices_by(ptimes, 'masterpatientid')
        else: 
            indices = np.random.choice(N, N, replace=True)
        jobs_data_1.append((Y, Yhat1, ptimes, stat_funcs, indices))
        jobs_data_2.append((Y, Yhat2, ptimes, stat_funcs, indices))
    
    print("Running jobs...")
    with Pool(min(os.cpu_count() - 4, 32)) as pool:
        bootstrap_stats_list_1 = pool.map(_bootstrapWorker, jobs_data_1)
    with Pool(min(os.cpu_count() - 4, 32)) as pool:
        bootstrap_stats_list_2 = pool.map(_bootstrapWorker, jobs_data_2)

    print("Collating statistics")
    raw_stats_1 = np.zeros((B, len(stat_funcs)))
    for idx, stats in enumerate(bootstrap_stats_list_1): 
        raw_stats_1[idx,:] = stats    
    raw_stats_2 = np.zeros((B, len(stat_funcs)))
    for idx, stats in enumerate(bootstrap_stats_list_2): 
        raw_stats_2[idx,:] = stats    

    raw_stats = raw_stats_2 - raw_stats_1

    ci_results = np.zeros((len(stat_funcs), 3))
    for idx in range(ci_results.shape[0]): 
        lower, upper = np.quantile(raw_stats[:,idx], (0.025, 0.975))
        mean = np.mean(raw_stats[:,idx])
        ci_results[idx] = [mean, lower, upper]

    ci_results = pd.DataFrame(ci_results)    
    ci_results.index = [k for k, v in stat_funcs.items()]
    ci_results.columns = ['Statistic', 'Lower CI', 'Upper CI']

    return ci_results, raw_stats

