import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import gc
import os

os.chdir('/home/sambeet/data/kaggle/talking_data/')

def read_data(filename='train_subset_new.csv'):
    for feature in features:
        print("Loading ", feature)
        # Import data one column at a time
        train_unit = pd.read_csv(filename,usecols=[feature])  # Change this from "train_sample" to "train" when you're comfortable!
        # Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
        if feature in int_features:
            train_unit = pd.to_numeric(train_unit[feature], downcast='unsigned')
        # Convert time data to datetime data, instead of strings
        elif feature in time_features:
            train_unit = pd.to_datetime(train_unit[feature])
        # Converts the target variable from int64 to boolean. Can also get away with uint16.
        elif feature in bool_features:
            train_unit = train_unit[feature].astype('bool')
        else:
            train_unit = train_unit[feature].astype('float16')

        # Make and append each column's data to a dataframe.
        if feature == 'app':
            train = pd.DataFrame(train_unit)
        else:
            train[feature] = train_unit
    del train_unit
    gc.collect()
    print("vars and data type: ")
    train.info()
    return train
'''
del xgtrain,xgtest
gc.collect()
'''
features = ['app', 'device', 'os', 'channel', 'hour', 'is_attributed', 'ip_n', 'app_n','ip_hour_count','ip_app_hour_count','app_channel_hour_count','ip_app_os_hour_count']
int_features = ['app', 'device', 'os', 'channel', 'hour','ip_d', 'app_d', 'channel_d','device_d', 'os_d', 'ip_n', 'app_n','channel_n', 'device_n', 'os_n', 'ip_hour_count', 'app_hour_count','channel_hour_count', 'device_hour_count',
                'os_hour_count','ip_app_hour_count','ip_channel_hour_count','app_channel_hour_count', 'ip_app_channel_count','ip_app_os_hour_count']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

train_data = read_data()

target = 'is_attributed'
predictors = ['app', 'device', 'os', 'channel', 'hour', 'ip_n', 'app_n','ip_hour_count','ip_app_hour_count','app_channel_hour_count','ip_app_os_hour_count']
categorical = ['app', 'device', 'os', 'channel', 'hour']

xgtrain = lgb.Dataset(train_data[predictors].values, label=train_data[target].values, feature_name=predictors,categorical_feature=categorical, free_raw_data=False)
xgtrain.save_binary('train_data_subset_final.bin')
del train_data
gc.collect()

test_data = read_data('test_subset_new.csv')
xgtest = lgb.Dataset(test_data[predictors].values, label=test_data[target].values,feature_name=predictors,categorical_feature=categorical,free_raw_data = False,reference=xgtrain)
xgtest.save_binary('test_data_subset_final.bin')
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
lgb_model = lgb.train(lgb_params,xgtrain,valid_sets=[xgtest],valid_names=['test'],evals_result=evals_results,num_boost_round=1500,early_stopping_rounds=50,verbose_eval=50,feval=None)

n_estimators = lgb_model.best_iteration
print("\nModel Report")
print("n_estimators : ", n_estimators)
print("AUC:", evals_results['test']['auc'][n_estimators-1])

fi_gain = lgb.plot_importance(lgb_model,title = 'Light GBM Feature Importance - Gain',importance_type = 'gain',figsize=(15,15))
fi_split = lgb.plot_importance(lgb_model,title = 'Light GBM Feature Importance - Split',importance_type = 'split',figsize=(15,15))

lgb_model.save_model('lgb_base_final.txt')
