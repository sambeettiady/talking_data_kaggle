import pandas as pd
import numpy as np
import os
import gc
import pyarrow.feather as pyfa

def do_mean( df, group_cols, counted, agg_type='float32', show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df.groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:'new_var'})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df.new_var = df.new_var.astype(agg_type)
    gc.collect()
    return df.new_var

def do_var( df, group_cols, counted, agg_type='float32',show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df.groupby(group_cols)[counted].var().reset_index().rename(columns={counted:'new_var'})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df.new_var = df.new_var.astype(agg_type)
    gc.collect()
    return df.new_var

train_data = pyfa.read_feather(source='data_intermediate.feather',nthreads=8)

print 'Calculating mean and var...'
train_data['ip_chan_day_var_hour'] = do_var(train_data[['ip', 'day', 'channel', 'hour']], ['ip', 'day', 'channel'], 'hour');gc.collect()
train_data['ip_app_os_var_hour'] = do_var(train_data[['ip', 'app', 'os', 'hour']], ['ip', 'app', 'os'], 'hour');gc.collect()
train_data['ip_app_chan_var_day'] = do_var(train_data[['ip', 'app', 'channel','day']], ['ip', 'app', 'channel'], 'day');gc.collect()
train_data['ip_app_chan_mean_hour'] = do_mean(train_data[['ip','app','channel','hour']], ['ip', 'app', 'channel'], 'hour');gc.collect()

train_data.columns = [str(col) for col in train_data.columns]
print 'Saving feather...'
train_data.to_feather('train_with_new_features.feather')
print 'Done!'
