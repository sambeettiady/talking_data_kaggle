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
'''
['ip', 'app', 'device', 'os', 'channel', 'hour', 'is_attributed','click_time', 'attributed_time', 'ip_hour_count', 'app_hour_count','channel_hour_count', 'device_hour_count', 'os_hour_count',
       'ip_mean', 'app_mean', 'channel_mean', 'device_mean', 'os_mean','ip_app_hour_count', 'ip_channel_hour_count','app_channel_hour_count', 'ip_app_channel_count', 'ip_app_os_hour_count']
'''

def read_data(filename='train_subset.csv'):
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

features = ['ip','app', 'device', 'os', 'channel', 'hour', 'is_attributed' ,'click_time', 'attributed_time']
int_features = ['ip','app', 'device', 'os', 'channel', 'hour']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

test_data = read_data('validation_subset.csv')

dt_files = ['ip_mean_download_time.csv','app_mean_download_time.csv','channel_mean_download_time.csv','device_mean_download_time.csv','os_mean_download_time.csv']
me_files = ['ip_mean_encoding.csv','app_mean_encoding.csv','channel_mean_encoding.csv','device_mean_encoding.csv','os_mean_encoding.csv']
dl_files = ['ip_downloads.csv','app_downloads.csv','channel_downloads.csv','device_downloads.csv','os_downloads.csv']
cl_files = ['ip_clicks.csv','app_clicks.csv','channel_clicks.csv','device_clicks.csv','os_clicks.csv']

#Join avg download times
for file in dt_files:
    print file
    temp = pd.read_csv(file)
    temp[temp.columns[1]] = temp[temp.columns[1]].astype('float16')
    temp.replace(np.inf,np.nan,inplace=True)
    temp.fillna(temp[temp.columns[1]].max(),inplace=True)
    test_data= test_data.merge(temp,on=file.split('_')[0], how='left')
    test_data.fillna(0,inplace=True)
gc.collect()

#Join with pre-calculated mean encodings
for file in me_files:
    print file
    temp = pd.read_csv(file)
    temp[temp.columns[1]] = temp[temp.columns[1]].astype('float16')
    test_data = test_data.merge(temp,on=file.split('_')[0], how='left')
    test_data.fillna(0.002471,inplace=True)
gc.collect()

#Join with pre-calculated global download counts
for file in dl_files:
    print file
    temp = pd.read_csv(file)
    temp[temp.columns[1]] = temp[temp.columns[1]].astype('uint16')
    test_data = test_data.merge(temp,on=file.split('_')[0], how='left')
    test_data.fillna(0,inplace=True)
gc.collect()

#Join with pre-calculated global click counts
for file in cl_files:
    print file
    temp = pd.read_csv(file)
    temp[temp.columns[1]] = temp[temp.columns[1]].astype('uint16')
    test_data = test_data.merge(temp,on=file.split('_')[0], how='left')
    test_data.fillna(0,inplace=True)
gc.collect()

test_data['ip_hour_count'] = create_count_feature(test_data,['ip','hour'])
test_data['app_hour_count'] = create_count_feature(test_data,['app','hour'])
test_data['channel_hour_count'] = create_count_feature(test_data,['channel','hour'])
test_data['device_hour_count'] = create_count_feature(test_data,['device','hour'])
test_data['os_hour_count'] =  create_count_feature(test_data,['os','hour'])
test_data['ip_app_hour_count'] =  create_count_feature(test_data,['ip', 'app','hour'])
test_data['ip_channel_hour_count'] = create_count_feature(test_data,['ip', 'channel','hour'])
test_data['app_channel_hour_count'] = create_count_feature(test_data,['app', 'channel','hour'])
test_data['ip_app_channel_count'] = create_count_feature(test_data,['ip', 'app', 'channel','hour'])
test_data['ip_app_os_hour_count'] = create_count_feature(test_data,['ip', 'app', 'os','hour'])
gc.collect()

print("vars and data type: ")
test_data.info()
test_data.to_csv('test_subset_new.csv',index = False)
gc.collect()
