#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import os
cwd = os.getcwd()
os.chdir('/home/sambeet/data/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import xlearn as xl

test_df  = pd.read_csv('test.csv', dtype="category", 
                       usecols = ['app', 'device', 'os', 'channel'])
                       
test_df['is_attributed'] = ""
test_df['is_attributed'] = test_df['is_attributed'].astype("category")
test_df.head()
test_df.info()

#train_df  = pd.read_csv('train.csv', dtype="category", 
#                       usecols = ['app', 'device', 'os', 'channel', 'is_attributed'])

#train_df.app.unique()
#train_df.device.unique()
#train_df.os.unique()
#train_df.channel.unique()
#train_df.is_attributed.unique()

#train= train_df[0:166413501]
#valid = train_df[166413501:]




def convert_to_ffm(df,type,numerics,categories,features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
         catdict[x] = 0
    for x in categories:
         catdict[x] = 1
    
    nrows = df.shape[0]
    ncolumns = len(features)
    with open(str(type) + "_ffm.txt", "w") as text_file:

    # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str((datarow['is_attributed']))
            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):
                     datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                else:
         # For a new field appearing in a training example
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode #encoding the feature
         # For already encoded fields
                    elif(datarow[x] not in catcodes[x]):
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode #encoding the feature
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"

            datastring += '\n'
            text_file.write(datastring)


categories = test_df.columns[0:4]
numerics = []
features = test_df.columns[0:4]

#convert_to_ffm(train, "TrainFinal", numerics , categories, features)
#convert_to_ffm(valid, "ValidFinal", numerics , categories, features)
convert_to_ffm(test_df, "TestFinal", numerics , categories, features)

################# FFM ##################

#ffm_model = xl.create_ffm()
#ffm_model.setTrain("TrainFinal_ffm.txt")
#ffm_model.setValidate("ValidFinal_ffm.txt")

#param = {'task':'binary', # ‘binary’ for classification, ‘reg’ for Regression
#         'k':4,           # Size of latent factor
#         'lr':0.1,        # Learning rate for GD
#         'lambda':0.0002, # L2 Regularization Parameter
#         'metric':'auc',  # Metric for monitoring validation set performance
#         'epoch':10,      # Maximum number of Epochs
#         'opt':'adagrad'    # Adagrad adapts the learing rate to the parameters 
#        }
        

#ffm_model.fit(param, "model.out")

