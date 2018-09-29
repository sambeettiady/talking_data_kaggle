import numpy as np
import pandas as pd
import  os

os.chdir('/home/sambeet/data/')

submission_file_names = ['sub_9695.csv','sub_9718.csv','sub_9736.csv','sub_9761.csv','sub_9769.csv','sub_9774.csv','sub_9776.csv']
to_submit = ['sub_9769.csv','sub_9776.csv']

for file in submission_file_names:
    print file
    temp = pd.read_csv(file)
    temp.columns = ['click_id',file.split('.')[0]]
    if file == 'sub_9695.csv':
        final = temp.copy()
    else:
        final = final.merge(temp,on=['click_id'])

final[[file.split('.')[0] for file in to_submit]].corr()
final['is_attributed'] = final[[file.split('.')[0] for file in to_submit]].mean(axis=1)
final[['click_id','is_attributed']].to_csv('sub_ensemble_4.csv',index = False)

overall = overall.groupby('click_id').sum()
overall.to_csv('sub_lgb_18.csv')
