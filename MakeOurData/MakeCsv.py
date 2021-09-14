import os,sys,re,pickle
from numpy.lib.twodim_base import diag
import pandas as pd 
import numpy as np 

seed = 202106

root_data_path = '/data/duongdb/FH_OCT_08172021' 
os.chdir(root_data_path)

score_labels = "FH_OCTs_label_train_original.csv" # FH_OCTs_label_test
foutname = "FH_OCTs_label_train_input.csv"
if 'test' in score_labels: 
  foutname = "FH_OCTs_label_test_input.csv"
  datatype='test'
else: 
  datatype='train'



# ! get all images first, filter by id names later
img_data_path = os.path.join(root_data_path,'FH_OCT_Images')
images = os.listdir(img_data_path) # all img names

# ! make csv input for the images. 
score_labels = pd.read_csv(score_labels)
print (score_labels.shape)
if datatype=='train': 
  # ! we must run the test set first to not get this error. 
  testcsv = pd.read_csv("FH_OCTs_label_test_input.csv")
  testimg = list ( testcsv['person_id'] ) 
  # ! filter test from train 
  score_labels = score_labels[score_labels['Study ID'].isin(testimg)==False]
  #
  score_labels = score_labels.sample(frac=1,random_state=seed) # ! random shuffle row, only need to shuffle for trainset 
  score_labels = score_labels.reset_index(drop=True)


#

print (score_labels.shape)

# ! get list of diagnosis 
diagnosis = {('d'+k).strip():[] for i,k in enumerate(sorted ( list ( set ( score_labels['Clinical Dignosis'].values ) ) ) )}

# ! get list of genes 
genes = {('g'+k).strip():[] for i,k in enumerate(sorted ( list ( set ( score_labels['Gene'].values ) ) ) )}

# ! go over this label file, add in img name and path. 
images_dict = {'name':[], 
               'path':[], 
               'label': [], 
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


for index,row in score_labels.iterrows(): # Study ID	OD	OS	Zeiss date	Heidelberg date
  img = [i for i in images if row['Study ID'] in i] # ! check pattern FH_001 in the jpg name
  if len(img) == 0: print ('no images for {}'.format(row['Study ID']))
  # 
  if datatype == 'train': 
    fold = index % 5 # ! if we have our testset, then we just do 5 folds ....... ! make 5 fold ? note: use 6 folds, where fold 0 is the specified test set. 
  else: 
    fold = 5 # ! set fold=5 for test images 
  # ! we want to split folds based on the individuals. use both eyes in same fold
  for i in img: # for each image matching with this id number
    if 'OD' in i: 
      eye = 'OD'
      images_dict['eye_position_od'].append(1)
      images_dict['eye_position_os'].append(0)
    else: 
      eye = 'OS'
      images_dict['eye_position_od'].append(0)
      images_dict['eye_position_os'].append(1)
    # machine type
    if '_Z_' in i: 
      machine_type = 'Z'
      images_dict['age_taken'].append(row['Age at the OCT test (Zeiss)'])
      images_dict['machine_type_z'].append(1)
      images_dict['machine_type_hb'].append(0)
    else: 
      machine_type = 'HB'
      images_dict['age_taken'].append(row['Age at the OCT test (Heidelberg)'])
      images_dict['machine_type_z'].append(0)
      images_dict['machine_type_hb'].append(1)
    #
    images_dict['name'].append(i)
    images_dict['path'].append(os.path.join(img_data_path,i))
    images_dict['label'].append(str(row['Grade ' + eye])) # ! take the column with name "Grade OS or OD"
    images_dict['FHScore'].append(str(row['Grade ' + eye])) # ! need this if we want to model "FH grade-->logMAR"
    images_dict['person_id'].append(row['Study ID'])
    images_dict['fold'].append(fold)
    # 
    # LogMAR OD
    # Spherical Equivalent OD
    images_dict['logMAR'].append(row['LogMAR '+eye])
    images_dict['spherical_equivalent'].append(row['Spherical Equivalent '+eye])
    #
    if row['Nystagmus'].strip() == '+' : 
      images_dict['nystagmus'].append(1)
    else: 
      images_dict['nystagmus'].append(0)
    # 
    # ! diagnosis 
    for i in diagnosis: # for each diagnosis, create 1-hot
      if i != 'd'+row['Clinical Dignosis'].strip(): 
        images_dict[i].append(0)
      else: 
        images_dict[i].append(1)
    # ! gene 
    for i in genes: # for each diagnosis, create 1-hot
      if i != 'g'+row['Gene'].strip(): 
        images_dict[i].append(0)
      else: 
        images_dict[i].append(1)
    
#

for k in images_dict: 
  print (k)
  print (len(images_dict[k]))


# 
df = pd.DataFrame.from_dict(images_dict)
df.shape
df.to_csv(foutname, index=False)

# ! look at labels count 
for i in sorted(set(df['label'].values)):
  print ( i, ' ', images_dict['label'].count(str(i))/df.shape[0] ) 


# test
# 1   0.30612244897959184
# 2   0.30612244897959184
# 3   0.20408163265306123
# 4   0.1836734693877551 

# train
# 1   0.22010869565217392
# 2   0.13858695652173914
# 3   0.36684782608695654
# 4   0.27445652173913043
