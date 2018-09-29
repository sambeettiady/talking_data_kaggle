import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pyarrow.feather as pyfa
import lightgbm as lgb
import gc

train_data = pyfa.read_feather('train_data.feather')

test_data = train_data[(train_data.shape[0] - 5000000):train_data.shape[0]]
train_data = train_data[0:(train_data.shape[0] - 5000000)]
gc.collect()

target = 'is_attributed'
predictors = train_data.columns
categorical = ['app', 'device', 'os', 'channel', 'hour']

xgtrain = lgb.Dataset(train_data[predictors].values, label=train_data[target].values, feature_name=predictors,categorical_feature=categorical, free_raw_data=False)
xgtrain.save_binary('train_data.bin')
del train_data
gc.collect()

xgtest = lgb.Dataset(test_data[predictors].values, label=test_data[target].values,feature_name=predictors,categorical_feature=categorical,free_raw_data = False,reference=xgtrain)
xgtest.save_binary('test_data.bin')
del test_data
gc.collect()

lgb_params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 10000,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.5,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequency of subsample, <=0 means no enable
    'colsample_bytree': 0.75,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 200,  # because training data is extremely unbalanced
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 8,
    'verbose': 1
}

evals_results = {}
lgb_model = lgb.train(lgb_params,xgtrain,valid_sets=[xgtest],valid_names=['test'],evals_result=evals_results,num_boost_round=1500,early_stopping_rounds=30,verbose_eval=50,feval=None)

n_estimators = lgb_model.best_iteration
print("\nModel Report")
print("n_estimators : ", n_estimators)
print("AUC:", evals_results['test']['auc'][n_estimators-1])

#fi_gain = lgb.plot_importance(lgb_model,title = 'Light GBM Feature Importance - Gain',importance_type = 'gain',figsize=(15,15))
#fi_split = lgb.plot_importance(lgb_model,title = 'Light GBM Feature Importance - Split',importance_type = 'split',figsize=(15,15))

lgb_model.save_model('lgb_gc.txt')
