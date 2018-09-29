import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.feather as pyfa
import gc
from sklearn.externals import joblib

train_data = pyfa.read_feather('train_data.feather',nthreads=8)

target = 'is_attributed'
predictors = list(train_data.columns)
predictors.remove(target)
categorical = ['app', 'device', 'os', 'channel', 'hour']

print 'Train test split'
y_test = train_data[(train_data.shape[0] - 30000000):train_data.shape[0]][target].values
x_test = train_data[(train_data.shape[0] - 30000000):train_data.shape[0]][predictors].values
y_train = train_data[0:(train_data.shape[0] - 30000000)][target].values
train_data = train_data[0:(train_data.shape[0] - 30000000)][predictors].values
gc.collect()

print 'Training'
lgb_estimator = lgb.LGBMClassifier(n_estimators=2000,boosting_type='gbdt',learning_rate=0.2,num_leaves = 31, max_depth = 5,min_child_samples = 10000,
                                    objective='binary',scale_pos_weight=200,subsample=0.5,colsample_bytree=0.7,min_child_weight=0,subsample_for_bin=200000,
                                    max_bin=100,silent = False)

lgb_estimator.fit(train_data, y_train,eval_set=[(x_test,y_test)],eval_metric='auc',early_stopping_rounds=25,verbose=True,feature_name=predictors,categorical=categorical)

print(lgb_estimator.best_iteration_, lgb_estimator.best_score_)

joblib.dump(lgb_estimator, 'lgb.pkl')
print 'Done'
