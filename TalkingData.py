import os
cwd = os.getcwd()
os.chdir('/home/sambeet/data/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
import xgboost as xgb
#%matplotlib inline

dtypes = {
        
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8'
        }

train_df  = pd.read_csv('train.csv', dtype=dtypes, chunksize = 100000, 
                        usecols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'])

train_df.get_chunk(1).head()

aggs = []

print('-'*38)

for chunk in train_df:
    agg = chunk.groupby(['app','device', 'os',
                         'channel'])['is_attributed'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)
    print('.')
print('')


aggs = pd.concat(aggs, axis=0)

aggs.head()
aggs.tail()
#type(aggs)
#aggs.info()
#aggs.loc[aggs['mean'] > 0]

agg = aggs.groupby(['app','device', 'os','channel']).sum().reset_index()
agg['count'] -= agg['sum']
agg = agg.rename(columns={'sum':'downloads','count':'clicks'})
agg.head()

CLICK_WEIGHT = 0.01
agg['relevance'] = agg['downloads'] + CLICK_WEIGHT * agg['clicks']
agg.head()

agg = pd.DataFrame(agg)

train  = pd.read_csv('train.csv', dtype=dtypes, 
                        usecols = [ 'app', 'device', 'os', 'channel', 'is_attributed'])
 
del(train_df)
del(chunk)
del(aggs)



aggData = train.merge(agg, how = 'inner', on = ['app','device', 'os','channel'],right_index = True)
aggData.head()
aggData[0:1000].to_csv('agg_sub.csv')

temp = pd.read_csv('agg_sub.csv')
temp.head()
temp.info()

del(agg)
del(train)


target = temp['is_attributed'].astype('category')
temp.drop('is_attributed', inplace = True, axis =1)
temp.drop('Unnamed: 0', inplace = True, axis = 1)
temp.columns.values

params = {'eta': 0.1, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'silent': True}
          
x1, x2, y1, y2 = train_test_split(temp, target, test_size=0.1, random_state=99)
print(x1.info())
print(x2.info())


watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 250, watchlist, maximize=True, verbose_eval=10)

test = pd.read_csv("test.csv")
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop(['ip', 'click_id', 'click_time'], axis=1, inplace=True)
test = test.merge(agg, how = 'left', left_on = ['app','device', 'os','channel'],right_index = True).fillna(0)
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_sub.csv',index=False)
