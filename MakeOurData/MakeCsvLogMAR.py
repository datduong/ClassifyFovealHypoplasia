
import os,sys,re,pickle 

import numpy as np
import pandas as pd 


# ! read in categorical, replace logMAR as "label" col. 
os.chdir('/data/duongdb/FH_OCT_08172021')
df = 'FH_OCTs_label_6fold.csv'
df = pd.read_csv(df,na_values="-")

df['label'] = list ( np.round( df['logMAR'].values , 4 ) )  # * 100
df = df.drop(columns=['logMAR'])

df.to_csv('FH_OCTs_label_6fold_logMAR.csv',index=None)
