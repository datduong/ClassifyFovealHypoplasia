import os,sys,re,pickle
import pandas as pd 
import numpy as np

# ! 
def MakeSoftLabelFromSingle (labelstring,diagnosis2idx): 
  this_label = [0]*len(diagnosis2idx)
  this_label [ diagnosis2idx[labelstring] ] = 1 
  labelstring = ';'.join(str(s) for s in this_label)
  return labelstring


#
os.chdir('/data/duongdb/FH_OCT_08172021')

# labelset = '1,2,3,4'.split(',') # ! what labels to use

labelset = 'A,B,C'.split(',') # ! what labels to use

diagnosis2idx = {val:index for index,val in enumerate(labelset)}
 
df_original = pd.read_csv('FH_OCTs_label_test_input_match_train_col_driving.csv',dtype=str)

temp_ = list ( df_original['label'] ) 
temp_ = [MakeSoftLabelFromSingle(s,diagnosis2idx) for s in temp_]
df_original['softlabel'] = temp_

df_original.to_csv('FH_OCTs_label_test_input_match_train_col_driving_soft.csv', index=False)

