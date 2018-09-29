import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

os.chdir('/home/sambeet/data/kaggle/talking_data/')

lgb_model = lgb.Booster(model_file='lgb_gc.txt')  #init model

test = pd.read_csv('test.csv')
predictors = ['app','device','os', 'channel', 'hour', 'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count']

sub = pd.DataFrame()
sub['click_id'] = test['click_id'].astype('int')

test['hour'] = pd.to_datetime(test.click_time).dt.hour.astype('uint8')
test['day'] = pd.to_datetime(test.click_time).dt.day.astype('uint8')

print('group by...')
gp = test[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_hour_count'})
gc.collect()

print('merge...')
test = test.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

print('group by...')
gp = test[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
gc.collect()

print('merge...')
test = test.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()

# # of clicks for each ip-app-os combination
print('group by...')
gp = test[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
gc.collect()

print('merge...')
test = test.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

print("vars and data type: ")
test.info()
test['ip_day_hour_count'] = test['ip_day_hour_count'].astype('uint16')
test['ip_app_count'] = test['ip_app_count'].astype('uint16')
test['ip_app_os_count'] = test['ip_app_os_count'].astype('uint16')

print("Predicting...")
sub['is_attributed'] = lgb_model.predict(test[predictors].values)
print("writing...")
sub.to_csv('sub_lgb_19.csv',index=False)
print("done...")
