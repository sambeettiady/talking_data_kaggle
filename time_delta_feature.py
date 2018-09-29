import pandas as pd
import numpy as np
import os
import gc
import pyarrow.feather as pyfa

ONE_SECOND = 1000000000    # Number of time units in one second
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
        if feature == 'click_id':
            train = pd.DataFrame(train_unit)
        else:
            train[feature] = train_unit
    del train_unit
    gc.collect()
    print("vars and data type: ")
    train.info()
    return train

features = ['click_id','ip','app', 'device', 'os', 'channel', 'click_time']
int_features = ['click_id','ip','app', 'device', 'os', 'channel','hour', 'ip_n', 'app_n','ip_hour_count','ip_app_hour_count','app_channel_hour_count','ip_app_os_hour_count']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

#######  READ THE DATA  #######
df = pyfa.read_feather(source='data.feather',nthreads=8)
#df = read_data('test.csv')
df['click_time'] = df.click_time.astype('int64').floordiv(ONE_SECOND).astype('int32')

#######  GENERATE COMBINED CATEGORY FOR GROUPING  #######
# Collapse all categorical features into a single feature
imax = df.ip.max()
amax = df.app.max()
dmax = df.device.max()
omax = df.os.max()
cmax = df.channel.max()
print( imax, amax, dmax, omax, cmax )
df['category'] = df.ip.astype('int64')
df.drop(['ip'], axis=1, inplace=True)
df['category'] *= amax
df['category'] += df.app
df.drop(['app'], axis=1, inplace=True)
df['category'] *= dmax
df['category'] += df.device
df.drop(['device'], axis=1, inplace=True)
df['category'] *= omax
df['category'] += df.os
df.drop(['os'], axis=1, inplace=True)
df['category'] *= cmax
df['category'] += df.channel
df.drop(['channel'], axis=1, inplace=True)
df.drop(['click_id'], axis=1, inplace=True)
gc.collect()

# Replace values for combined feature with a group ID, to make it smaller
print('\nGrouping by combined category...')
df['category'] = pd.Categorical(df['category']).codes.astype('uint32')
gc.collect()

#######  SORT BY CATEGORY AND INDEX  #######
# Collapse category and index into a single column
df['category'] = df.category.astype('int64').multiply(2**32).add(df.index.values.astype('int32'))
gc.collect()

# Sort by category+index (leaving each category separate, sorted by index)
print('\nSorting...')
df = df.sort_values(['category'])
gc.collect()

# Retrieve category from combined column
df['category'] = df.category.floordiv(2**32).astype('int32')
gc.collect()

#######  GENERATE TIME DELTAS  #######
# Calculate time deltas, and replace first record for each category by NaN
df['catdiff'] = df.category.diff().fillna(1).astype('uint8')
df.drop(['category'],axis=1,inplace=True)
df['backward_time_delta'] = df.click_time.diff().astype('float32')
df.loc[df.catdiff==1,'backward_time_delta'] = np.nan
df['forward_time_delta'] = df.backward_time_delta.shift(-1)

# Re-sort time_deltas back to the original order
df = df[['backward_time_delta','forward_time_delta']].fillna(-1).astype('int32')
gc.collect()
df.sort_index(inplace=True)
gc.collect()

#######  WRITE FEATHER FILE  #######
# Write to disk
print('\nWriting...')
df.to_feather('bidirectional_time_deltas_test.feather')
print( 'Done')
