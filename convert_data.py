import pandas as pd
import numpy as np
import os
import gc
import pyarrow.feather as pyfa

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
        if feature == 'ip':
            train = pd.DataFrame(train_unit)
        else:
            train[feature] = train_unit
    del train_unit
    gc.collect()
    print("vars and data type: ")
    train.info()
    return train

features = ['ip','app', 'device', 'os', 'channel', 'click_time','is_attributed']
int_features = ['click_id','ip','app', 'device', 'os', 'channel','hour']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

train_data = read_data('train.csv')

features = ['ip','click_id','app', 'device', 'os', 'channel', 'click_time']
int_features = ['click_id','ip','app', 'device', 'os', 'channel']
test_sup = read_data('test_supplement.csv')

train_data = pd.concat([train_data,test_sup],axis = 0,ignore_index=True)
train_data.reset_index()
train_data.to_feather('data.feather')
