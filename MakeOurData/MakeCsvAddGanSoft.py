
import os,sys,re,pickle
import pandas as pd 
import numpy as np 


from copy import deepcopy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fout_path", type=str, default=None)
parser.add_argument("--img_data_path", type=str, default=None)
parser.add_argument("--datatype", type=str, default='train')
parser.add_argument("--suffix", type=str, default='') # ! add in name e.g. Fold1X1
parser.add_argument("--fold", type=int, default=None) # ! fold to skip out
parser.add_argument("--original_train_csv", type=str, default=None)
parser.add_argument("--keep_label_original", type=str, default=None)
# parser.add_argument("--soft-label", type=int, default=1)
# parser.add_argument("--normal_in_gan", type=int, default=0)
parser.add_argument("--labels", type=str, default=None) # ! comma sep like 1,2,3,4

args = parser.parse_args()


# ! simple adding img, we only care about main labels (not 2nd labels like 'age group')

fold = args.fold
soft_img_dir = os.path.join(args.img_data_path,'F'+str(fold)+'X1') 
soft_img = os.listdir(soft_img_dir)

# images can't have fold=@fold, because @fold is the valid set. 
fold_to_use = np.arange(0,5,1) # 5 folds
fold_to_use = np.delete(fold_to_use, fold) # ! delete valid fold
print ('\n\nfold use', fold_to_use)

labelset = args.labels.split(',') # ! what labels to use
diagnosis2idx = {val:index for index,val in enumerate(labelset)}
diagnosis2idxReverse = {index:val for index,val in enumerate(labelset)}

np.random.seed(fold)

soft_img = np.random.permutation(soft_img)

# --------------------------------------------------------------------------------------------------------------

# ! have to follow previous format. 

score_labels = "/data/duongdb/FH_OCT_08172021/FH_OCTs_label_train_original.csv"
datatype='train'
score_labels = pd.read_csv(score_labels)
print (score_labels.shape)
if datatype=='train': 
  # ! we must run the test set first to not get this error. 
  testcsv = pd.read_csv("/data/duongdb/FH_OCT_08172021/FH_OCTs_label_test_input.csv")
  testimg = list ( testcsv['person_id'] ) 
  # ! filter test from train 
  score_labels = score_labels[score_labels['Study ID'].isin(testimg)==False]
  #
  score_labels = score_labels.sample(frac=1,random_state=fold) # ! random shuffle row, only need to shuffle for trainset 
  score_labels = score_labels.reset_index(drop=True)


# ! get list of diagnosis 
diagnosis = {('d'+k).strip():[] for i,k in enumerate(sorted ( list ( set ( score_labels['Clinical Dignosis'].values ) ) ) )}

# ! get list of genes 
genes = {('g'+k).strip():[] for i,k in enumerate(sorted ( list ( set ( score_labels['Gene'].values ) ) ) )}

# go over this label file, add in img name and path. 
images_dict = {'name':[], 
               'path':[], 
               'label': [], 
               'softlabel': [],
               'person_id':[], 
               'fold': [], 
               'eye_position_od': [],
               'eye_position_os': [],
               'machine_type_z': [], 
               'machine_type_hb': [], 
               'age_taken': [],
               'logMAR': [], 
               'FHScore': [],
               'spherical_equivalent': [], 
               'nystagmus': []
               }

#
images_dict.update(diagnosis)      
images_dict.update(genes)


def GetLabel(string, diagnosis2idx, diagnosis2idxReverse): 
  # seed00001014F1C1,4C2,4M0.75T0.82OD.png
  string = string.split('C') # seed00001014F1 1,4 2,4M0.75T0.82OD.png
  mainlabel = int(string[1][0]) # first label 
  minorlabel = int(string[2][0]) # 2nd label 
  try: 
    splitratio = np.round (float ( string[2].split('M')[1][0:4] ),2)
  except: 
    splitratio = np.round (float ( string[2].split('M')[1][0:3] ),2)
  this_label = [0]*len(diagnosis2idx)
  this_label[mainlabel] = splitratio
  this_label[minorlabel] = 1 - splitratio
  labelstring = ';'.join( "{:.2f}".format(s) for s in this_label )
  return diagnosis2idxReverse[mainlabel], diagnosis2idxReverse[minorlabel], labelstring, splitratio
  
for index,img in enumerate(soft_img): 
  images_dict ['name'].append( img )
  images_dict ['path'].append( os.path.join(soft_img_dir,img) )
  images_dict ['person_id'].append ('fake'+str(index))
  images_dict ['fold'].append( fold_to_use [ index % 4 ] ) # ! there are 4 folds left
  # label
  mainlabel, minorlabel, this_label, splitratio = GetLabel(img, diagnosis2idx, diagnosis2idxReverse)
  images_dict ['label'].append(mainlabel)
  images_dict['FHScore'].append(mainlabel)
  images_dict ['softlabel'].append(this_label)
  # fill the rest with 0, won't use them anyway 
  if 'OD' in img: 
    eye = 'OD'
    images_dict['eye_position_od'].append(1)
    images_dict['eye_position_os'].append(0)
  else: 
    eye = 'OS'
    images_dict['eye_position_od'].append(0)
    images_dict['eye_position_os'].append(1)
  #
  for n in ['machine_type_z', 'machine_type_hb', 'age_taken', 'logMAR', 'spherical_equivalent', 'nystagmus']: 
    images_dict[n].append(0)
  for n in diagnosis: 
    images_dict[n].append(0)
  for n in genes: 
    images_dict[n].append(0)
  
# 
# for k in images_dict: 
#   print (k)
#   print (len(images_dict[k]))

#
df = pd.DataFrame.from_dict(images_dict)
df.shape

# append 

def MakeSoftLabelFromSingle (labelstring,diagnosis2idx): 
  this_label = [0]*len(diagnosis2idx)
  this_label [ diagnosis2idx[labelstring] ] = 1 
  labelstring = ';'.join("{:.2f}".format(s) for s in this_label)
  return labelstring
  
if args.original_train_csv is not None: # ! append to original training dataset
  for f in args.original_train_csv.split(','):
    print ('read {}'.format(f))
    df_original = pd.read_csv(f,dtype=str)
    print ('original df size (at read in) {}'.format(df_original.shape))
    if args.keep_label_original is not None: 
      args.keep_label_original = args.keep_label_original.split(',') # keep these labels from original (so also keep the images)
      df_original = df_original[df_original.label.isin(args.keep_label_original)]
    print ('original df size (may filter out some stuffs) {}'.format(df_original.shape))
    # ! create soft label from single label. in the original csv 
    temp_ = list ( df_original['label'] ) 
    temp_ = [MakeSoftLabelFromSingle(s,diagnosis2idx) for s in temp_]
    df_original['softlabel'] = temp_
    df = pd.concat([df,df_original])


print ('final size {}'.format(df.shape))
df.to_csv(os.path.join(args.fout_path,args.datatype+'-'+args.suffix+'.csv'), index=False)

