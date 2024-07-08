import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from lightgbm import Dataset
from scipy import special
import mlflow
from mlflow import log_metric, log_param, log_artifact, set_tag, log_text
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from eliot import log_message
from .utils import get_facilities_from_train_data, get_date_diff
from saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH, saiva_api
from saiva.training.utils import load_config

# added EPSILON for numeric stability
EPSILON = sys.float_info.epsilon

def get_active_facilities():
    try:
        config = load_config(LOCAL_TRAINING_CONFIG_PATH)
        training_config = config.training_config
        org = training_config.organization_configs[0]['organization_id']

        client_metadata = saiva_api.facilities.get_all_active(org_id=org)
        active_facility_list = []
        for i in range(len(client_metadata)):
            active_facility_list.append(org + '_' + client_metadata[i].customers_identifier)

        return active_facility_list
    except Exception as e:
        log_message(message_type='error', message=f'Error in getting active facilities: {e}')
        return []
    
def get_pos_neg(np_series):
    total = len(np_series)
    neg = np.count_nonzero(np_series==0)
    pos = np.count_nonzero(np_series==1)
    return pos, neg, round(neg/pos, 3)

def get_positive_date_col(idens):
    positive_date_cols = [col for col in idens.columns if col.startswith('positive_date')]
    if len(positive_date_cols) == 1:
        return positive_date_cols[0]
    else:
        raise ValueError(f"There must be 1 column in `idens` with `positive_date_`, found {len(positive_date_cols)}")

def update_thresholds(idens, threshold):
    base = idens.copy()
    base['n_ranked_residents'] = base.groupby(['facilityid', 'censusdate'])['masterpatientid'].transform('count')

    if isinstance(threshold, int):
        base['threshold'] = threshold
    else:
        base['threshold'] = (threshold * base['n_ranked_residents']).astype(int)

    return base

def performance_base_processing(idens, preds, target_3_day, threshold=None):

    assert target_3_day.shape[0] == idens.shape[0]

    positive_date_col = get_positive_date_col(idens)

    base = idens.copy()
    if threshold: # if threshold is None, the values from idens table will be used
        base = update_thresholds(base, threshold)

    base.loc[:,'predictionvalue'] = preds
    base.loc[:,'target'] = target_3_day
    
    base['predictionrank'] = base.groupby(['facilityid', 'censusdate'])['predictionvalue'].rank(ascending=False,
                                                                                                     method='first')
    base['show_in_report'] = (base['predictionrank'] <= base['threshold'])

    return base

def get_auc(target_3_day, preds):
    total_aucroc = roc_auc_score(target_3_day, preds)
    total_aucroc_25_fpr = roc_auc_score(target_3_day, preds, max_fpr=0.25)
    total_ap = average_precision_score(target_3_day, preds)

    return total_aucroc, total_aucroc_25_fpr, total_ap

def f_beta_score(precision, recall, beta=2):
    return ((1+beta**2)*(precision*recall)) / ((beta**2)*precision + recall)

def get_pline_precision(df):
    
    """The precision at the given threshold calculated the label-wise
    """

    return df.loc[df['show_in_report'], 'target'].mean()

def get_recall(df):
    num, denom = df['show_in_report'].sum(), df.shape[0]
    recall = num/denom
    return [num, denom, recall]

def get_incidents_recall(performance_base, model_type):
    
    df = convert_to_events(performance_base)

    print(f'Total {model_type}s after processing = ',df.shape[0])
    result = {}
    result['recall_all'] = get_recall(df)
    print('Total Recall = ',result['recall_all'][2])

    result['recall_LOS_LE30'] = get_recall(df.query('LFS <= 30'))

    result['recall_LOS_G30']= get_recall(df.query('LFS > 30'))
    
    result['recall_short_term'] = get_recall(df[df.long_short_term=='short'])
    
    result['recall_long_term'] = get_recall(df[df.long_short_term=='long'])
    
    pos_no_payer = (df.long_short_term=='no payer info')
    if pos_no_payer.sum()>0:
        result['recall_without_payer'] = get_recall(df[pos_no_payer])
        
        print(f'{pos_no_payer.sum()} positive patient days without payertype info')
        num_TP = (pos_no_payer&(df.show_in_report)).sum()
        print(f'{num_TP} of them are True Positive')
        
    del df
    return result

def get_stay_length(staylength):
    if staylength > 120:
        return 120
    else:
        return staylength
    
def get_line_graph(df, selectedField, label_name, color):
    fig = px.line(
        df, 
        y=[selectedField], 
        x=list(df.index), 
        labels={
            'x':'Length Of Stay',
            'y': 'Recall'
        }
    )
    fig['data'][0]['line']['color']=color
    newnames = {selectedField:label_name}
    
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))

    return fig

def draw_graph(df, recall, sdate, edate):
    incidents = [0 for i in range(0,121)]  
    caught_incidents = [0 for i in range(0,121)]
    missed_incidents = [0 for i in range(0,121)]
    for index, row in df.iterrows():
        j = int(get_stay_length(row['LFS']))
        incidents[j] += 1
        if row['min_predictionrank'] <= 15:
            caught_incidents[j] += 1
        if row['min_predictionrank'] > 15:
            missed_incidents[j] += 1
    
    # create a dataframe from the above 4 lists
    metric_df = pd.DataFrame ({"incidents": incidents, "caught_incidents": caught_incidents, "missed_incidents": missed_incidents})
    metric_df['recall'] = recall
    cumsum_df = metric_df[['caught_incidents','incidents']].cumsum()
    cumsum_df['fraction_incidents'] = cumsum_df['caught_incidents'] / cumsum_df['incidents'] ## harsh def for recall
    reverse_cumsum_df = pd.DataFrame()
    reverse_cumsum_df['caught_incidents'] = metric_df.loc[::-1, 'caught_incidents'].cumsum()[::-1]
    reverse_cumsum_df['incidents'] = metric_df.loc[::-1, 'incidents'].cumsum()[::-1]
    reverse_cumsum_df['reverse_fraction_incidents'] = reverse_cumsum_df['caught_incidents'] / reverse_cumsum_df['incidents']  ## harsh def

    metric_df['reverse_fraction_incidents'] = reverse_cumsum_df['reverse_fraction_incidents']
    metric_df['fraction_incidents'] = cumsum_df['fraction_incidents']
    
    print(reverse_cumsum_df['reverse_fraction_incidents'].shape)
    print(cumsum_df['fraction_incidents'].shape)    
    
    fig = make_subplots()

    plot1 = get_line_graph(metric_df,'fraction_incidents', 'Recall at LOS <= x axis', 'blue')
    plot2 = get_line_graph(metric_df,'reverse_fraction_incidents', 'Recall at LOS > x axis', 'red') 
    
    
    fig.add_trace(
        plot1["data"][0]
    )
    fig.add_trace(
        plot2["data"][0]
    )
    
    fig.add_vline(x=30, 
                  line_width=3, 
                  line_dash="dot", 
                  line_color="black",
                  annotation_text='LOS=30'
                 )# 30 LOS
    
    recall = round(recall, 2)
    fig.add_hline(y=recall,
                  line_width=3, 
                  line_dash="dot", 
                  line_color="green",
                  annotation_text=f'Total Recall={recall}'
                 ) # Total Recall Indicator
    
    
    facilities = get_facilities_from_train_data(df)
    
    fig.update_layout(hovermode="x")
    
    days = (pd.to_datetime(edate) - pd.to_datetime(sdate)).days
    
    fig.update_layout(
        title_text=f"{CLIENT}, Duration: {sdate} to {edate}, {days} days, Facility count={len(facilities)}",
        font_size=9,
        yaxis=dict(
            title="Recall",
            tickformat=',.0%',
            range = [0,1]
        ),
        xaxis=dict(
            title='Length of Stay'
        )
    )
    
    filename = 'recall_plot.png'
    pio.write_image(fig, filename)
    log_artifact(filename)  # Export to MlFlow
    
def generate_recall_curve(performance_base, recall, sdate, edate):
    df = performance_base.query('target_3_day == 1')
    df['min_predictionrank'] = df.groupby([f'positive_date{MODEL_TYPE}', 'facilityid', 'masterpatientid'])\
                                    ['predictionrank'].transform('min')
    df =  df.groupby([f'positive_date{MODEL_TYPE}', 'facilityid', 'masterpatientid']).last()
    print(f'Total {MODEL_TYPE}s = ',df.shape[0])
    draw_graph(df, recall, sdate, edate)    
    
def generate_auc_curve(actual_y, preds_y, aucroc, run_id):
    plt.clf()
    filename = f'auc_curve_{run_id}.png'
    fpr, tpr, thresh = roc_curve(actual_y, preds_y)
    plt.plot(fpr,tpr)
    plt.xlabel("AUC="+str(aucroc))
    plt.savefig(filename)
    log_artifact(filename)  # Export to MlFlow
    
def set_zero_to_epsilon(potential_zero):
    return max(potential_zero, EPSILON)

def convert_to_events(base):
    
    positive_date_col = get_positive_date_col(base)
    df = base.loc[base.target == 1].copy()
    df['min_predictionrank'] = df.groupby([positive_date_col, 'facilityid', 'masterpatientid'])\
                                        ['predictionrank'].transform('min')
    
    df['show_in_report'] = df.groupby([positive_date_col, 'facilityid', 'masterpatientid'])\
                                    ['show_in_report'].transform('max')

    df = df.groupby([positive_date_col, 'facilityid', 'masterpatientid']).last().reset_index()
    return df

def model_performance(preds, eval_data, idens, eval_type='LFS'):
    
    base = performance_base_processing(idens, preds, eval_data)
    events = convert_to_events(base)
    
    result = events[['facilityid', 'censusdate', eval_type, 'min_predictionrank', 'show_in_report']].copy()
    
    return result

def auc(preds, eval_data):
    preds_data = preds.get_label() if isinstance(preds, Dataset) else preds
    eval_data = eval_data.get_label() if isinstance(eval_data, Dataset) else eval_data
    aucroc = roc_auc_score(eval_data, preds_data)
    return 'auc', aucroc, True

def get_recall_function(name, filter_column='LFS', filter_operator=None, filter_value=None):
    '''
    params:
        name -- string, recall type, e.g.'recall_overall', 'recall(LOS>30)', 'long_term', 'short_term'
        filter_column -- string, the name of the column we want to filter on
        filter_operator -- func, an operator returns (a list of) true or false, e.g. np.greater, np.less_equal, np.equal
        filter_value -- the value used for filtering 
    
    return: a recall function
    '''
  
    def recall_function(preds, eval_data):
        preds_data = preds.get_label() if isinstance(preds, Dataset) else preds
        idens = eval_data.idens
        eval_data = eval_data.get_label() if isinstance(eval_data, Dataset) else eval_data
        df = model_performance(preds, eval_data, idens, eval_type=filter_column)

        if filter_column and filter_operator and filter_value:
            mask = filter_operator(df[filter_column].values, filter_value)
            df_values = df.loc[mask, 'show_in_report'].values 
        else:
            df_values = df['show_in_report'].values

        recall = np.sum(df_values) / df_values.shape[0]

        return name, recall, True
    
    return recall_function

def logloss_metric(preds, train_data):
    y = train_data.get_label()
    p = special.expit(preds)

    ll = np.empty_like(p)
    pos = y == 1
    
    p_pos = [set_zero_to_epsilon(p) for p in p[pos]]
    p_neg = [set_zero_to_epsilon(1 - np) for np in p[~pos]]
    
#     p_neg = set_zero_to_epsilon(potential_zero = 1 - p[~pos])
    
    ll[pos] = np.log(p_pos)
    ll[~pos] = np.log(p_neg)

    is_higher_better = False
    return 'logloss', -ll.mean(), is_higher_better

def logloss_objective(preds, train_data):
    y = train_data.get_label()
    p = special.expit(preds)
    grad = p - y
    hess = p * (1 - p)
    return grad, hess

def run_test_set(
    model,
    selected_modelid,
    run_id,
    test_start_date,
    test_end_date,
    x_df,
    target_3_day,
    idens,
    model_type,
    selected_model_version=None,
    dataset = 'TEST',
    recall_plot=False,
    threshold=15,
    log_in_mlflow=True,
):
    
    pos, neg, n2p_ratio = get_pos_neg(target_3_day)
    
    # ===========================Predict on test dataset=======================
    preds = model.predict(x_df)
    print("=============================Prediction completed...=============================")
    # =========================== TOTAL AUCROC on TEST SET ===========================
    total_aucroc,total_aucroc_25_fpr, total_ap = get_auc(target_3_day, preds)

    performance_test_base = performance_base_processing(idens, preds, target_3_day, threshold=threshold)
    performance_test_base.to_csv(f'performance_{dataset.lower()}_base.csv', index=False)
    
    active_facility_list = get_active_facilities()
    active_facility_result = {}

    if active_facility_list:
        active_facility_performance_test_base = performance_test_base[performance_test_base['facilityid'].isin(active_facility_list)]
        
    duplicate_mask =  performance_test_base.duplicated(subset=['censusdate', 'facilityid', 'predictionvalue'], keep=False)
    if sum(duplicate_mask) > 0: # check for number of 1s in the mask
        duplicate_rows = performance_test_base.loc[duplicate_mask]
        duplicate_rows.to_csv("duplicate_rows_performance_{}_base.csv".format(dataset), index=False)
        log_message(message_type='info', message=f'{sum(duplicate_mask)} duplicate prediction probabilities detected.')
        
    pline_precision = get_pline_precision(performance_test_base)
    result = get_incidents_recall(performance_test_base, model_type)
    if active_facility_list and dataset=='TEST':
        log_text(json.dumps(active_facility_list), "active_facilities.json")
        active_facility_result = get_incidents_recall(active_facility_performance_test_base, model_type)
    recall_types = ['recall_all', 'recall_LOS_LE30', 'recall_LOS_G30', 'recall_short_term', 'recall_long_term', 'recall_without_payer']
    
    cutoff_string = str(threshold) if isinstance(threshold, int) else f'{threshold*100}_percent'

    if log_in_mlflow:
        log_param(f'11_{dataset}_START_DATE', test_start_date)
        log_param(f'12_{dataset}_END_DATE', test_end_date)
        log_param(f'13_{dataset}_DURATION_days', get_date_diff(test_start_date, test_end_date))
        log_param(f'14_{dataset}_POS_COUNT', pos)
        log_param(f'15_{dataset}_NEG_COUNT', neg)
        log_param(f'16_{dataset}_N2P_RATIO', n2p_ratio)
        log_param(f'17_{dataset}_patient_days_per_AED', f'{str(performance_test_base.shape[0])}/{str(result["recall_all"][1])} = {round(performance_test_base.shape[0]/result["recall_all"][1], 3)}')
        
        if not selected_model_version is None:
            log_param(f'18_MODEL_DESCRIPTION', selected_model_version)
        log_param(f'19_MODEL_ID', selected_modelid)

        log_metric(f'{dataset}_01_aucroc', total_aucroc)
        for i, recall_type in enumerate(recall_types, 2):
            if result.get(recall_type, None) is not None:
                log_metric(f'{dataset}_0{i}_{model_type}_{recall_type}_at_rank_{cutoff_string}', result[recall_type][2])
                log_metric(f'{dataset}_0{i}a_{model_type}_{recall_type}_at_rank_{cutoff_string} - num', int(result[recall_type][0]))
                log_metric(f'{dataset}_0{i}b_{model_type}_{recall_type}_at_rank_{cutoff_string} - denom', result[recall_type][1])
                log_param(f'{dataset}_0{i}c_{model_type}_{recall_type}_at_rank_{cutoff_string}  - str', f'{str(result[recall_type][0])}/{str(result[recall_type][1])} = {str(round(result[recall_type][2], 3))}')
            if dataset=='TEST' and (active_facility_result.get(recall_type, None) is not None) and active_facility_list:
                log_metric(f'active_facility_{dataset}_0{i}_{model_type}_{recall_type}_at_rank_{cutoff_string}', active_facility_result[recall_type][2])
                log_metric(f'active_facility_{dataset}_0{i}a_{model_type}_{recall_type}_at_rank_{cutoff_string} - num', int(active_facility_result[recall_type][0]))
                log_metric(f'active_facility_{dataset}_0{i}b_{model_type}_{recall_type}_at_rank_{cutoff_string} - denom', active_facility_result[recall_type][1])
                log_param(f'active_facility_{dataset}_0{i}c_{model_type}_{recall_type}_at_rank_{cutoff_string}  - str', f'{str(active_facility_result[recall_type][0])}/{str(active_facility_result[recall_type][1])} = {str(round(active_facility_result[recall_type][2], 3))}')
                
        log_metric(f'{dataset}_09_{model_type}_Pline_precision_at_rank_{cutoff_string}', pline_precision)
    else:
        print('Total_aucroc ->', total_aucroc)
        for i, recall_type in enumerate(recall_types, 2):
            if result.get(recall_type, None) is not None:
                print(f'{dataset}_0{i}_{model_type}_{recall_type}_at_rank_{cutoff_string} - {result[recall_type][0]}/{result[recall_type][1]} = {result[recall_type][2]}')
            if active_facility_result.get(recall_type, None) is not None and active_facility_list and dataset=='TEST':
                if active_facility_result.get(recall_type, None) is not None:
                    print(f'active_facility_{dataset}_0{i}_{model_type}_{recall_type}_at_rank_{cutoff_string} - {active_facility_result[recall_type][0]}/{active_facility_result[recall_type][1]} = {active_facility_result[recall_type][2]}')
            
    
    generate_auc_curve(target_3_day, preds, total_aucroc, run_id)
    if recall_plot:
        generate_recall_curve(performance_test_base, recall, test_start_date, test_end_date)
    
    return total_aucroc, result['recall_all'][2], result['recall_LOS_LE30'][2], result['recall_LOS_G30'][2], result['recall_short_term'][2],  result['recall_long_term'][2]

class FocalLoss:

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better