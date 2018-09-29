import pandas as pd
import numpy as np
import os
import gc
import pyarrow.feather as pyfa
import lightgbm as lgb

def do_count(df, group_cols, agg_type='uint32',show_agg=True):
    if show_agg:
        print("Aggregating by ", group_cols, '...')
    gp = df.groupby(group_cols).size().rename('new_var').to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df.new_var = df.new_var.astype(agg_type)
    gc.collect()
    return df.new_var

def do_countuniq(df, group_cols, counted, agg_type='uint32',show_agg=True):
    if show_agg:
        print("Counting unqiue ", counted, " by ", group_cols, '...')
    gp = df.groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted: 'new_var'})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    df.new_var = df.new_var.astype(agg_type)
    gc.collect()
    return df.new_var

def do_cumcount(df, group_cols, counted, agg_type='uint32',show_agg=True):
    if show_agg:
        print("Cumulative count by ", group_cols, '...')
    gp = df.groupby(group_cols)[counted].cumcount()
    df['new_var'] = gp.values
    del gp
    df.new_var = df.new_var.astype(agg_type)
    gc.collect()
    return df.new_var

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

test = True
if test == False:
    train_data = pyfa.read_feather(source='data.feather',nthreads=8)
else:
    train_data = pd.read_csv('test.csv')

print('Extracting new features...')
train_data['hour'] = pd.to_datetime(train_data.click_time).dt.hour.astype('uint8')
train_data['day'] = pd.to_datetime(train_data.click_time).dt.day.astype('uint8')
train_data['minute'] = pd.to_datetime(train_data.click_time).dt.minute.astype('uint8')
gc.collect()

print 'Calculating unique count...'
train_data['uniq_chan_by_ip'] = do_countuniq(train_data[['ip','channel']], ['ip'], 'channel', 'uint8');gc.collect()
train_data['uniq_hour_by_ip_day'] = do_countuniq(train_data[['ip', 'day','hour']], ['ip', 'day'], 'hour', 'uint8');gc.collect()
train_data['uniq_app_by_ip'] = do_countuniq(train_data[['ip','app']], ['ip'], 'app', 'uint8');gc.collect()
train_data['uniq_os_by_ip_app'] = do_countuniq(train_data[['ip', 'app', 'os']], ['ip', 'app'], 'os', 'uint8');gc.collect()
train_data['uniq_dev_by_ip'] = do_countuniq(train_data[['ip','device']], ['ip'], 'device', 'uint16');gc.collect()
train_data['uniq_chan_by_app'] = do_countuniq(train_data[['app','channel']], ['app'], 'channel');gc.collect()
train_data['uniq_app_by_ip_dev_os'] = do_countuniq(train_data[['ip','device', 'os','app']], ['ip', 'device', 'os'], 'app');gc.collect()
gc.collect()

print 'Calculating cumcount...'
train_data['cumcount_app_by_ip_dev_os'] = do_cumcount(train_data[['ip', 'device', 'os','app']], ['ip', 'device', 'os'], 'app');gc.collect()
train_data['cumcount_os_by_ip'] = do_cumcount(train_data[['ip','os']], ['ip'], 'os');gc.collect()
gc.collect()

print 'Calculating count...'
train_data['ip_count'] = do_count(train_data[['ip']], ['ip']);gc.collect()
train_data['app_count'] = do_count(train_data[['app']], ['app']);gc.collect()
train_data['os_dev_count'] = do_count(train_data[['os','device']], ['os', 'device']);gc.collect()
train_data['ip_app_count'] = do_count(train_data[['ip', 'app']], ['ip', 'app']);gc.collect()
train_data['ip_app_os_count'] = do_count(train_data[['ip', 'app', 'os']], ['ip', 'app', 'os'], 'uint16');gc.collect()
train_data['ip_dev_count'] = do_count(train_data[['ip', 'device']], ['ip', 'device']);gc.collect()
gc.collect()
train_data['app_chan_count'] = do_count(train_data[['app', 'channel']], ['app', 'channel']);gc.collect()
train_data['app_chan_os_count'] = do_count(train_data[['os', 'app','channel']], ['os', 'app','channel']);gc.collect()
train_data['ip_day_hour_count'] = do_count(train_data[['ip', 'day', 'hour']], ['ip', 'day', 'hour']);gc.collect()
train_data['ip_day_hour_minute_count'] = do_count(train_data[['ip', 'day', 'hour','minute']], ['ip', 'day', 'hour','minute']);gc.collect()
train_data['app_chan_day_hour_count'] = do_count(train_data[['app', 'channel', 'day', 'hour']], ['app', 'channel', 'day', 'hour']);gc.collect()
gc.collect()
train_data['ip_os_day_hour_count'] = do_count(train_data[['ip', 'day', 'os', 'hour']], ['ip', 'day', 'os', 'hour'], 'uint16'); gc.collect()
train_data['ip_app_day_hour_count'] = do_count(train_data[['ip', 'day', 'app', 'hour']], ['ip', 'day', 'app', 'hour'], 'uint16'); gc.collect()
train_data['ip_app_os_day_hour_count'] = do_count(train_data[['ip', 'day', 'app', 'os', 'hour']], ['ip', 'day', 'app', 'os', 'hour'], 'uint16'); gc.collect()
train_data['app_day_hour_count'] = do_count(train_data[['app', 'day', 'hour']], ['app', 'day', 'hour'], 'uint32'); gc.collect()

print 'Calculating mean and var...'
train_data['ip_chan_day_var_hour'] = do_var(train_data[['ip', 'day', 'channel', 'hour']], ['ip', 'day', 'channel'], 'hour');gc.collect()
train_data['ip_app_os_var_hour'] = do_var(train_data[['ip', 'app', 'os', 'hour']], ['ip', 'app', 'os'], 'hour');gc.collect()
train_data['ip_app_chan_var_day'] = do_var(train_data[['ip', 'app', 'channel','day']], ['ip', 'app', 'channel'], 'day');gc.collect()
train_data['ip_app_chan_mean_hour'] = do_mean(train_data[['ip','app','channel','hour']], ['ip', 'app', 'channel'], 'hour');gc.collect()

if test:
    sub = pd.DataFrame()
    sub['click_id'] = train_data.click_id.astype('int')
    train_data.drop(['click_id','day','minute'],axis=1,inplace=True)
    train_data.drop(['ip','click_time'],axis=1,inplace=True)
    btdtest = pyfa.read_feather('bidirectional_time_deltas_test.feather')
    train_data = pd.concat([train_data,btdtest],axis=1)
    lgb_model = lgb.Booster(model_file='../lgb_150.txt')  # init model
    sub['is_attributed'] = lgb_model.predict(train_data.values)
    sub.to_csv('sub_nfe_2.csv',index=False)
else:
    print 'Saving feather...'
    train_data.columns = [str(col) for col in train_data.columns]
    train_data.to_feather('train_with_new_features.feather')
    print 'Done!'
