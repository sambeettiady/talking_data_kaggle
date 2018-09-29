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

lgb_model = lgb.Booster(model_file='../lgb_150.txt')  #init model
test_data = pd.read_csv('test.csv')
test_data ['hour'] = pd.to_datetime(test_data.click_time).dt.hour.astype('uint8')

cl_files = ['ip_clicks.csv','app_clicks.csv']

#Join with pre-calculated global click counts
for file in cl_files:
    print file
    temp = pd.read_csv(file)
    temp[temp.columns[1]] = temp[temp.columns[1]].astype('uint16')
    test_data = test_data.merge(temp,on=file.split('_')[0], how='left')
    test_data.fillna(0,inplace=True)
gc.collect()

test_data['ip_hour_count'] = create_count_feature(test_data,['ip','hour'])
test_data['ip_app_hour_count'] =  create_count_feature(test_data,['ip', 'app','hour'])
test_data['app_channel_hour_count'] = create_count_feature(test_data,['app', 'channel','hour'])
test_data['ip_app_os_hour_count'] = create_count_feature(test_data,['ip', 'app', 'os','hour'])
test_data.drop(['ip','click_time'],axis=1,inplace=True)
gc.collect()

print("vars and data type: ")
test_data.info()
test_data.to_csv('test_sub_file.csv',index = False)
gc.collect()

predictors = ['app', 'device', 'os', 'channel', 'hour', 'ip_n', 'app_n','ip_hour_count','ip_app_hour_count','app_channel_hour_count','ip_app_os_hour_count']

sub = pd.DataFrame()
sub['click_id'] = test_data['click_id'].astype('int')

print("Predicting...")
sub['is_attributed'] = lgb_model.predict(test_data[predictors].values)
print("writing...")
sub.to_csv('sub_lgb_17042018_2.csv',index=False)
print("done...")
