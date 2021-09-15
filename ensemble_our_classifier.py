import sys, os, re, pickle
import pandas as pd
import numpy as np
from glob import glob
import argparse

import OtherMetrics

def read_output(filename,args): 
    df = pd.read_csv(filename) 
    df = df.sort_values(by='name',ignore_index=True) # sort just to be consisent. 
    df = df.reset_index()
    print ('\nread in {} dim {}'.format(filename,df.shape[0]))
    # print (df)
    return df


def rm_lt50_average(prediction_array,num_labels): # @prediction_array is [[model1], [model2]...] for one single observation 
    counter = 0
    ave_array = np.zeros(num_labels)
    for array in prediction_array: 
        if max(array) > 0.2 : # skip if no prediction is over 0.5
            ave_array = ave_array + array 
            counter = counter + 1
    return ave_array/counter 
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--labels', type=str)
    parser.add_argument('--output_name', type=str)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    args.labels = sorted ( args.labels.strip().split(',') ) 
    num_labels = len(args.labels)
    label_col_index = [str(i) for i in range(num_labels)] # ! col names are index, and not label names

    # ! we need to rank a prediction probability for each condition
    outputs = [read_output(csv,args) for csv in sorted(glob(os.path.join(args.model_dir, 'test_on_fold_5_from_fold*csv')))] # read each prediction
    
    final_df = outputs[0] # place holder 
    prediction_np = np.zeros((final_df.shape[0],num_labels))
    
    for index in range(final_df.shape[0]) : # ! for each observation 
        # average ?
        prediction_array = []
        for df1 in outputs: # go over each csv output, and get the row
            array1 = list ( df1.loc[index,label_col_index] ) 
            prediction_array.append ( [float(a) for a in array1] ) # array of array of numbers. [ [1,2,3], [2,3,4] ... ]
        # 
        prediction_np[index] = rm_lt50_average(prediction_array,num_labels) # ensemble over many models for 1 observation 

    print ('\nprediction_np shape ' , prediction_np.shape)
    final_df.loc[:,label_col_index] = prediction_np
    
    # need to recode the truth and prob labels. 
    try: 
        diagnosis2idx = {int(value):index for index,value in enumerate(args.labels)} # ! FH uses 1,2,3,4
    except: 
        diagnosis2idx = {value:index for index,value in enumerate(args.labels)} # ! driving uses A,B,C
    
    print ('diagnosis2idx : ', diagnosis2idx)
    our_label_index = list ( np.arange(len(diagnosis2idx)) )
    
    final_df['true_label_index'] = final_df['label'].map(diagnosis2idx)
    
    # print ("final_df['true_label_index']")
    # print (final_df['true_label_index'])
    
    PROBS = prediction_np.argmax(axis=1)
    final_df['predict_label_index'] = PROBS

    try:
        final_df['average_score'] = np.matmul (prediction_np , np.reshape([1,2,3,4],(4,1))).flatten().tolist()
    except: 
        final_df['average_score'] = np.matmul (prediction_np , np.reshape([1,2,3],(3,1))).flatten().tolist() # ! make sense ? or just pass ? 

    # print ('PROBS : ', PROBS)
    # print (PROBS.shape)
    for p in PROBS: 
        if p not in our_label_index: 
            print ('we predict something outside our list of labels')
            print (p)
    
    TARGETS = np.array ( list (final_df['true_label_index']) ) 
    # print ('TARGETS : ', TARGETS)

    if len(diagnosis2idx) == 3: # ! driving score
        final_df['label'] = final_df['label'].map(diagnosis2idx) # ! remap 
        final_df['label'] = final_df['label'] + 1 
    
    final_df.to_csv(os.path.join(args.model_dir,"final_prediction.csv"),index=False) # writeout
    
    OtherMetrics.plot_confusion_matrix_manual( prediction_np, TARGETS, diagnosis2idx, os.path.join(args.model_dir,'ensem_confusion_matrix' ), our_label_index )
    
    

