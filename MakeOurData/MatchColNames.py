
import os,sys,re,pickle 
import pandas as pd 
import numpy as np 

# ! cols in train not found in test and vice versa. 
# ! remove those ?? 
os.chdir('/data/duongdb/FH_OCT_08172021')

train = pd.read_csv('FH_OCTs_label_train_input_driving.csv')
test = pd.read_csv('FH_OCTs_label_test_input_driving.csv') 

train_col = set ( train.columns ) 
test_col = set ( test.columns ) 

in_train_not_test = train_col - test_col
in_test_not_train = test_col - train_col

# ! add train to test, just put everyone in test not have the column 
for c in in_train_not_test: 
  test[c] = 0 # set to 0 because no test obs have this column

# set to unknown 
for c in in_test_not_train: 
  if c[0]=='g': 
    test.loc[test[c]==1,'gunknown'] = 1
  # else: 
  #   test.loc[test[c]==1,'dunknown'] = 1

# ! 

test.to_csv ('FH_OCTs_label_test_input_match_train_col_driving.csv',index=False)

in_train_not_test
in_test_not_train
