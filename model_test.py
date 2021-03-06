#INTENT: To limit RAM usage when importing/storing data
#Importing all the data at once took > 16 GB, which is my computer's RAM capacity. I made this script to keep the RAM usage low while loading data.

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
import lightgbm as lgb
import gc
import os
import matplotlib.pyplot as plt
import csv
import seaborn as sns

os.chdir('/home/sambeet/data/kaggle/talking_data/')
features = [ 'app', 'device', 'os', 'channel', 'hour', 'is_attributed', 'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count']
int_features = [ 'app', 'device', 'os', 'channel','hour','ip_day_hour_count', 'ip_app_count', 'ip_app_os_count']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

for feature in features:
    print("Loading ", feature)
    #Import data one column at a time
    train_unit = pd.read_csv("train_file_with_new_features.csv", usecols=[feature]) #Change this from "train_sample" to "train" when you're comfortable!
    
    #Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
    if feature in int_features:    train_unit = pd.to_numeric(train_unit[feature], downcast='unsigned')
    #Convert time data to datetime data, instead of strings
    elif feature in time_features: train_unit=pd.to_datetime(train_unit[feature])
    #Converts the target variable from int64 to boolean. Can also get away with uint16.
    elif feature in bool_features: train_unit = train_unit[feature].astype('bool')
    
    #Make and append each column's data to a dataframe.
    if feature == 'app': train = pd.DataFrame(train_unit)
    else: train[feature] = train_unit

del train_unit
gc.collect()
print("vars and data type: ")
train.info()

X_train, X_test = train_test_split(train,test_size = 0.2,random_state=3737)
del train
gc.collect()

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count']
categorical = ['app', 'device', 'os', 'channel', 'hour']

xgtrain = lgb.Dataset(X_train[predictors].values, label=X_train[target].values,feature_name=predictors,
                          categorical_feature=categorical,free_raw_data = False)
xgtrain.save_binary('training_data.bin')
del X_train
gc.collect()

xgtest = lgb.Dataset(X_test[predictors].values, label=X_test[target].values,feature_name=predictors,
                          categorical_feature=categorical,free_raw_data = False,reference=xgtrain)
xgtest.save_binary('testing_data.bin')
del X_test
gc.collect()

lgb_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 10000,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.5,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequency of subsample, <=0 means no enable
    'colsample_bytree': 1,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 99, # because training data is extremely unbalanced
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 4,
    'verbose': 1
}

evals_results = {}
lgb_model = lgb.train(lgb_params,xgtrain,valid_sets=[xgtest],valid_names=['test'],
                    evals_result=evals_results,num_boost_round=15000,early_stopping_rounds=50,
                    verbose_eval=100,feval=None)

n_estimators = lgb_model.best_iteration
print("\nModel Report")
print("n_estimators : ", n_estimators)
print("AUC:", evals_results['test']['auc'][n_estimators-1])

#fi_gain = lgb.plot_importance(lgb_model,title = 'Light GBM Feature Importance - Gain',importance_type = 'gain',figsize=(8,8))
#fi_split = lgb.plot_importance(lgb_model,title = 'Light GBM Feature Importance - Split',importance_type = 'split',figsize=(8,8))

lgb_model.save_model('lgb_Lr0.01_n15000.txt')
