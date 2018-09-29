import pandas as pd
import numpy as np
import gc
import os

os.chdir('/home/sambeet/data/kaggle/talking_data/')

features = ['ip','app','click_time']#, 'device', 'os', 'channel', 'is_attributed','click_time', 'attributed_time']
int_features = ['ip','app', 'device', 'os', 'channel']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

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

def create_click_count_feature(dataframe,group_by,filename,varname):
    print('Computing Counts...')
    gp = dataframe[group_by + ['click_time']].groupby(by=group_by)[['click_time']].count().reset_index().rename(index=str, columns={'click_time': varname})
    gp.to_csv(filename,index=False)
    gc.collect()

def create_download_count_feature(dataframe,group_by,filename,varname):
    print('Computing Downloads...')
    gp = dataframe[group_by + ['is_attributed']].groupby(by=group_by)[['is_attributed']].sum().reset_index().rename(index=str, columns={'is_attributed': varname})
    gp.to_csv(filename,index=False)
    gc.collect()

def create_mean_encoding(dataframe,group_by,filename,varname):
    print('Computing Mean Encodings...')
    gp = dataframe[group_by + ['is_attributed']].groupby(by=group_by)[['is_attributed']].mean().reset_index().rename(index=str, columns={'is_attributed': varname})
    gp.to_csv(filename,index=False)
    gc.collect()

def create_mean_download_time(dataframe,group_by,filename,varname):
    print('Computing Avg. Download Time...')
    gp = dataframe[group_by + ['difft']].groupby(by=group_by)[['difft']].mean().reset_index().rename(index=str, columns={'difft': varname})
    gp.to_csv(filename,index=False)
    gc.collect()

train = read_data('train.csv')
train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')
gc.collect()
#train = train[train.day != 9]

#Compute clicks count
create_click_count_feature(train,['ip'],'ip_clicks_global.csv','ip_n')
create_click_count_feature(train,['app'],'app_clicks_global.csv_','app_n')
