import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import gc
import os
import matplotlib.pyplot as plt

os.chdir('/home/sambeet/data/kaggle/talking_data/model/')
models = os.listdir('/home/sambeet/data/kaggle/talking_data/model/')
models.remove('lgb_fe_40_1.txt')
models.remove('lgb_fe_40_2.txt')

feature_importance = dict()
for model in models:
    lgb_model = lgb.Booster(model_file=model)  #init model
    total_gain = np.sum(lgb_model.feature_importance('gain'))
    if model == 'lgb_fe_170.txt':
        for feature, importance in zip(lgb_model.feature_name(),lgb_model.feature_importance('gain')):
            feature_importance.update({feature : importance/total_gain})
    else:
        for feature, importance in zip(lgb_model.feature_name(),lgb_model.feature_importance('gain')):
            feature_importance[feature] = feature_importance[feature] + (importance/total_gain)
for feature,importance in zip(feature_importance.keys(),feature_importance.values()):
    feature_importance[feature] = 100*importance/len(models)
feature_importance
import collections as cl
feature_importance = cl.OrderedDict(sorted(feature_importance.items(), key=lambda t: t[1], reverse=True))
feature_importance.keys()
predictors = ['app','channel','app_count','ip_app_count','forward_time_delta','uniq_chan_by_ip','os','app_day_hour_count','app_chan_day_hour_count','uniq_app_by_ip','app_chan_count','ip_app_os_day_hour_count','ip_count','hour','uniq_dev_by_ip','os_dev_count','device']
list(set.difference(set(feature_importance.keys()),set(predictors)))
plt.rcdefaults()
fig, ax = plt.subplots()
# Example data
y_pos = np.arange(len(feature_importance.keys()))
ax.barh(y_pos, feature_importance.values(),  align='center',color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_importance.keys())
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Importance')
ax.set_title('Importance')
plt.show()
