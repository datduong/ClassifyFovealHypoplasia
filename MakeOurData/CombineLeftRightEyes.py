
import os,sys,re,pickle 
import pandas as pd 
import numpy as np 

import cv2

# ! combine both eyes into the same image 

# read in csv, get image path, concat images
os.chdir('/data/duongdb/FH_OCT_08172021')

newimgdir = '/data/duongdb/FH_OCT_08172021/CombineEyes' 
try: 
  os.mkdir(newimgdir)
except: 
  pass

#
fout = open ('FH_OCTs_label_6fold_combine_eyes.csv','w')

df = 'FH_OCTs_label_6fold+driving.csv'
df = pd.read_csv(df,dtype=str)

# 'eye_position_od,eye_position_os,machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus,dAchromatopsia,dAniridia,dCHS,dErdheim Chester disease,dFH isolated,dHPS,dNanophthalmos,dOA,dOCA,dWaardenburg 2A,gCNGB3,gGPR143,gHPS1,gHPS5,gMITF,gOCA2,gPAX6,gPRSS56,gSLC45A2,gTYR,gunknown'

cols1 = 'label,person_id,fold,machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus,dAchromatopsia,dAniridia,dCHS,dErdheim Chester disease,dFH isolated,dHPS,dNanophthalmos,dOA,dOCA,dWaardenburg 2A,gCNGB3,gGPR143,gHPS1,gHPS5,gMITF,gOCA2,gPAX6,gPRSS56,gSLC45A2,gTYR,gunknown'.split(',')
cols2 = 'machine_type_z,machine_type_hb,age_taken,spherical_equivalent'.split(',')
         
cols2_ = [c+'2' for c in cols2]
fout.write('name,path,label,person_id,fold,'+','.join(cols1+cols2_)+'\n')

allnames = df['name'].values.tolist()
allpaths = df['path'].values.tolist()

eye_od = [e for e in df['name'].values if '_OD_' in e] # get one eye
validpair = {}
for e1 in eye_od: # get the matching pair by replacing "_OS_"
  e2 = re.sub ('_OD_','_OS_',e1)
  i1 = allnames.index(e1)
  p1 = allpaths [ i1 ] # location in the @df, then get path 
  try: 
    i2 = allnames.index(e2)
    p2 = allpaths [ i2 ] 
  except: 
    print ('fail ', p1)
  img1 = cv2.imread(p1)
  img2 = cv2.imread(p2)
  # resize
  dsize = (min(img1.shape[1],img2.shape[1]), min(img1.shape[0],img2.shape[0]))
  img1 = cv2.resize(img1, dsize)
  img2 = cv2.resize(img2, dsize)
  # concat 
  im_h = cv2.hconcat([img1, img2])
  newpath = os.path.join(newimgdir,e1)
  cv2.imwrite(newpath, im_h)
  # ! concat with other info
  newmeta = list(df.iloc[i1][cols1]) + list(df.iloc[i2][cols2]) # location of all the meta-data for the 1st eye 
  fout.write ( e1 + ',' + newpath + ',' + ','.join(newmeta) + '\n')


# 
fout.close() 
