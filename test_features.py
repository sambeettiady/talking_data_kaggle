# INTENT: To limit RAM usage when importing/storing data
# Importing all the data at once took > 16 GB, which is my computer's RAM capacity. I made this script to keep the RAM usage low while loading data.

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import gc
import os

os.chdir('/home/sambeet/data/kaggle/talking_data/')

def read_data(filename='train_subset.csv'):
    for feature in features:
        print("Loading ", feature)
        # Import data one column at a time
        train_unit = pd.read_csv(filename,usecols=[feature],skiprows=range(1,68941878	))  # Change this from "train_sample" to "train" when you're comfortable!
        # Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
        if feature in int_features:
            train_unit = pd.to_numeric(train_unit[feature], downcast='unsigned')
        # Convert time data to datetime data, instead of strings
        elif feature in time_features:
            train_unit = pd.to_datetime(train_unit[feature])
        # Converts the target variable from int64 to boolean. Can also get away with uint16.
        elif feature in bool_features:
            train_unit = train_unit[feature].astype('bool')

        # Make and append each column's data to a dataframe.
        if feature == 'ip':
            train = pd.DataFrame(train_unit)
        else:
            train[feature] = train_unit
    del train_unit
    gc.collect()
    print("vars and data type: ")
    train.info()
    return train

def create_count_feature(dataframe,group_by):
    print('group by...')
    gp = dataframe[group_by + ['click_time']].groupby(by=group_by)[['click_time']].count().reset_index().rename(index=str, columns={'click_time': 'new_var'})
    gc.collect()
    print('merge...')
    dataframe = dataframe.merge(gp, on=group_by, how='left')
    del gp
    gc.collect()
    dataframe['new_var'] = dataframe['new_var'].astype('uint16')
    return dataframe['new_var']

features = ['ip','app', 'device', 'os', 'channel', 'is_attributed','click_time']
int_features = ['ip','app', 'device', 'os', 'channel']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

train_data = read_data('train.csv')
gc.collect()
train_data ['hour'] = pd.to_datetime(train_data .click_time).dt.hour.astype('uint8')

gc.collect()
cl_files = ['ip_clicks_global.csv','app_clicks_global.csv']

#Join with pre-calculated global click counts
for file in cl_files:
    print file
    temp = pd.read_csv(file)
    temp[temp.columns[1]] = temp[temp.columns[1]].astype('uint16')
    train_data = train_data.merge(temp,on=file.split('_')[0], how='left')
    train_data.fillna(0,inplace=True)
gc.collect()

#Count by hour variables
train_data['ip_hour_count'] = create_count_feature(train_data,['ip','hour'])
train_data['ip_app_hour_count'] =  create_count_feature(train_data,['ip', 'app','hour'])
train_data['app_channel_hour_count'] = create_count_feature(train_data,['app', 'channel','hour'])
train_data['ip_app_os_hour_count'] = create_count_feature(train_data,['ip', 'app', 'os','hour'])
gc.collect()

print("vars and data type: ")
train_data.info()
train_data.to_csv('train_subset_day8_9.csv',index = False)
gc.collect()

target = 'is_attributed'
predictors = ['app', 'device', 'os', 'channel', 'hour', 'ip_n', 'app_n','ip_hour_count','ip_app_hour_count','app_channel_hour_count','ip_app_os_hour_count']
categorical = ['app', 'device', 'os', 'channel', 'hour']

xgtrain = lgb.Dataset(train_data[predictors].values, label=train_data[target].values, feature_name=predictors,categorical_feature=categorical, free_raw_data=False)
xgtrain.save_binary('train_data_subset_day8_9.bin')
del train_data
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

#evals_results = {}
lgb_model = lgb.train(lgb_params,xgtrain,num_boost_round=700)#valid_sets=[xgtrain],valid_names=['test'],evals_result=evals_results,#early_stopping_rounds=50,verbose_eval=100,feval=None)

'''
n_estimators = lgb_model.best_iteration
print("\nModel Report")
print("n_estimators : ", n_estimators)
print("AUC:", evals_results['test']['auc'][n_estimators-1])
'''

fi_gain = lgb.plot_importance(lgb_model,title = 'Light GBM Feature Importance - Gain',importance_type = 'gain',figsize=(8,8))
fi_split = lgb.plot_importance(lgb_model,title = 'Light GBM Feature Importance - Split',importance_type = 'split',figsize=(8,8))

lgb_model.save_model('lgb_features_test_day8_9.txt')
