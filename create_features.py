import pandas as pd
import gc

features = ['ip','app', 'device', 'os', 'channel', 'is_attributed','click_time', 'attributed_time']
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

def create_mean_encoding(dataframe,group_by):
    print('group by...')
    gp = dataframe[group_by + ['is_attributed']].groupby(by=group_by)[['is_attributed']].mean().reset_index().rename(index=str, columns={'is_attributed': 'new_var'})
    gc.collect()
    print('merge...')
    dataframe = dataframe.merge(gp, on=group_by, how='left')
    del gp
    gc.collect()
    dataframe['new_var'] = dataframe['new_var'].astype('float16')
    return dataframe['new_var']


