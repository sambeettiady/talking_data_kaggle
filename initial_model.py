#INTENT: To limit RAM usage when importing/storing data
#Importing all the data at once took > 16 GB, which is my computer's RAM capacity. I made this script to keep the RAM usage low while loading data.

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import csv
import os

os.chdir('/home/sambeet/data/kaggle/talking_data/')
features = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
int_features = ['ip', 'app', 'device', 'os', 'channel']
time_features = ['click_time', 'attributed_time']
bool_features = ['is_attributed']

for feature in features:
    print("Loading ", feature)
    #Import data one column at a time
    train_unit = pd.read_csv("train.csv", usecols=[feature]) #Change this from "train_sample" to "train" when you're comfortable!
    
    #Pandas imports the numeric data as int64...the following should downgrade that to uint16, saving ~1GB in RAM for each column
    if feature in int_features:    train_unit = pd.to_numeric(train_unit[feature], downcast='unsigned')
    #Convert time data to datetime data, instead of strings
    elif feature in time_features: train_unit=pd.to_datetime(train_unit[feature])
    #Converts the target variable from int64 to boolean. Can also get away with uint16.
    elif feature in bool_features: train_unit = train_unit[feature].astype('bool')
    
    #Make and append each column's data to a dataframe.
    if feature == 'ip': train = pd.DataFrame(train_unit)
    else: train[feature] = train_unit

del train_unit
gc.collect()

train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')
gc.collect()

print('group by...')
gp = train[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_hour_count'})
gc.collect()

print('merge...')
train = train.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

# # of clicks for each ip-app combination
print('group by...')
gp = train[['ip','app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
gc.collect()

print('merge...')
train = train.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()

# # of clicks for each ip-app-os combination
print('group by...')
gp = train[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
gc.collect()

print('merge...')
train = train.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

print("vars and data type: ")
train.info()
train['ip_day_hour_count'] = train['ip_day_hour_count'].astype('uint16')
train['ip_app_count'] = train['ip_app_count'].astype('uint16')
train['ip_app_os_count'] = train['ip_app_os_count'].astype('uint16')
train.to_csv('train_file_with_new_features.csv',index = False)
gc.collect()

'''
cv_result_lgb = lgb.cv(lgb_params, 
                       dtrain_lgb, 
                       num_boost_round=1000, 
                       nfold=5, 
                       stratified=True, 
                       early_stopping_rounds=50, 
                       verbose_eval=100, 
                       show_stdv=True)
num_boost_rounds_lgb = len(cv_result_lgb['multi_logloss-mean'])
print('num_boost_rounds_lgb=' + str(num_boost_rounds_lgb))
'''

#X_train, X_test = train_test_split(train,test_size = 0.2,random_state=37)

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count']
categorical = ['app', 'device', 'os', 'channel', 'hour']

'''
xgtrain = lgb.Dataset(X_train[predictors].values, label=X_train[target].values,feature_name=predictors,
                          categorical_feature=categorical)

del X_train
gc.collect()

xgtest = lgb.Dataset(X_test[predictors].values, label=X_test[target].values,feature_name=predictors,
                          categorical_feature=categorical)
'''
xg_all = lgb.Dataset(train[predictors].values, label=train[target].values,feature_name=predictors,
                          categorical_feature=categorical,free_raw_data=False)

#del X_test, 
del train
gc.collect()

lgb_params = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100000,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.5,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 1,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99, # because training data is extremely unbalanced
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 8,
    'verbose': 1
}

'''
evals_results = {}
lgb_model = lgb.train(lgb_params,xgtrain,valid_sets=[xgtrain, xgtest],valid_names=['train','test'],
                    evals_result=evals_results,num_boost_round=500,early_stopping_rounds=50,
                    verbose_eval=10,feval=None)

n_estimators = lgb_model.best_iteration
print("\nModel Report")
print("n_estimators : ", n_estimators)
print("AUC:", evals_results['test']['auc'][n_estimators-1])
evals_results = {}
'''

lgb_model_final = lgb.train(lgb_params,xg_all,num_boost_round=2000,feval=None)
lgb_model_final.save_model('lgb_full_not_tuned_new_2.txt')

fi_gain = lgb.plot_importance(lgb_model_final,title = 'Light GBM Feature Importance - Gain',importance_type = 'gain',figsize=(8,8))
fi_split = lgb.plot_importance(lgb_model_final,title = 'Light GBM Feature Importance - Split',importance_type = 'split',figsize=(8,8))

#lgb.plot_tree(lgb_model_final)

'''
def save_cv_results_to_csv(gsearch, export_file):
	with open(export_file, 'w') as outfile:
		csvwriter = csv.writer(outfile, delimiter=',')
		cv_results = gsearch.cv_results_
		# Create the header using the parameter names 
		header = ["train_mean","train_std","test_mean","test_std"]
		param_names = [param for param in cv_results['params'][0]]
		header.extend(param_names)
		csvwriter.writerow(header)
		for i in range(len(cv_results['params'])):
			# Get mean and standard deviation
			mean_train = cv_results['mean_train_score'][i]
			std_train = cv_results['std_train_score'][i]
			mean_test = cv_results['mean_test_score'][i]
			std_test = cv_results['std_test_score'][i]
			row = [mean_train,std_train,mean_test,std_test]
			# Get the list of parameter settings and add to row
			params = [str(p) for p in cv_results['params'][i].values()]
			row.extend(params)
			csvwriter.writerow(row)

lgb_params = {
    'num_leaves': [7,15,31,64],  # we should let it be smaller than 2^(max_depth)
    'max_depth': [-1],  # -1 means no limit
    'min_child_samples': [100],  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': [100],  # Number of bucketed bin for feature values
    'subsample': [.7],  # Subsample ratio of the training instance.
    'subsample_freq': [1],  # frequence of subsample, <=0 means no enable
    'colsample_bytree': [0.7],  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': [0],  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': [200000],  # Number of samples for constructing bin
    'min_split_gain': [0],  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': [0],  # L1 regularization term on weights
    'reg_lambda': [0],  # L2 regularization term on weights
    'scale_pos_weight':[99] # because training data is extremely unbalanced 
}

scores = 'roc_auc'

lgb_estimator = lgb.LGBMClassifier(categorical_feature=[0,1,2,3,4],n_estimators=200,boosting_type='gbdt',
                                    learning_rate=0.1,objective='binary')
    
gsearch = GridSearchCV(estimator=lgb_estimator,param_grid=lgb_params,cv=5,scoring=scores,verbose=2) 

lgb_model = gsearch.fit(X = train[predictors],y = train[target])

print(lgb_model.best_params_, lgb_model.best_score_)

train_scores_mean = lgb_model.cv_results_['mean_train_score']
train_scores_std = lgb_model.cv_results_['std_train_score']
test_scores_mean = lgb_model.cv_results_['mean_test_score']
test_scores_std = lgb_model.cv_results_['std_test_score']
lw = 2
param_range = range(1,5)
plt.figure().set_size_inches(8, 6)
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.show()

save_cv_results_to_csv(gsearch,'lgbm_cv_auc.csv')
'''

