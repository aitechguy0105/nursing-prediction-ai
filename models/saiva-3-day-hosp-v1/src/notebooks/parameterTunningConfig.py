# This cell just creates a python file containing the contents of this cell

from hyperopt import hp

# Parameter tunning
lgb_param_space = {
 'application': 'binary',
 'objective': 'binary',
 'metric': 'auc',
 #'boosting_type': hp.choice('boosting_type', ['gbdt']),
 #'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
 'learning_rate': hp.uniform('learning_rate', 0.001, 0.05),
 #'max_depth': hp.quniform('max_depth', -1, 10, 1),
 #'min_child_samples': 20,
 #'min_child_weight': 0.001,
 #'min_split_gain': 0.0,
 'n_estimators': hp.quniform('n_estimators',100,400,10),
 'n_jobs': -1,
#  'num_leaves': hp.quniform('num_leaves', 30, 300, 1),
 #'subsample': hp.uniform('subsample', 0, 1),
 #'subsample_for_bin': hp.quniform('subsample_for_bin', 200000, 500000, 1000),
 'verbose': 3,
 'is_unbalance': hp.choice('is_unbalance', [True, False]),
 #'max_bin': hp.quniform('max_bin', 100,1000, 100),
 'early_stopping_round': None,
}
