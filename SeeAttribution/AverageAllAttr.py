import os, numpy, PIL 
import re
import numpy as np 
from PIL import Image
import pandas as pd 

# ! because we use 5-fold cv. we can average attribution for each fold. 
# https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil
# Access all PNG files in directory
# allfiles=os.listdir(os.getcwd())
# imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG",".jpeg",".jpg"]]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str) #
parser.add_argument('--output_name', type=str) #
parser.add_argument('--keyword', type=str, default=None) #
parser.add_argument('--label_file', type=str, help='oct image name does not have label... should have added labels to names')

args = parser.parse_args()
    
os.chdir(args.image_path)

label_file = pd.read_csv(args.label_file)
temp_ = list(label_file['name'])
temp_ = [re.sub(r'(\.jpg|\.jpeg)','',i) for i in temp_]
label_of_image = dict(zip(temp_, list(label_file['label'])))
LABELS = set(list(label_file['label']))

print (label_of_image)

imlist = [f for f in os.listdir(args.image_path) if 'byside' in f] 
if args.keyword is not None: 
  imlist = [ f for f in imlist if args.keyword in f ]
else: 
  args.output_name = 'all_img'

#
w,h=Image.open(imlist[0]).size
N=len(imlist)

for label in LABELS: 
  arr=numpy.zeros((h,w,3),numpy.float)

  counter = 0.0
  for im in imlist:

    # need to convert FH_001_OD_Z_2017_igZ_bysideSignAverage.png --> FH_010_OD_HB_2017.jpg # ! remove the jpg
    im2 = im.split('_ig')[0] 
    if label_of_image[im2] == label: 
      counter = counter + 1 
      imarr=numpy.array(Image.open(im),dtype=numpy.float)
      arr=arr+imarr

  # ! end this label 
  # Round values in array and cast as 8-bit integer
  arr=numpy.array(numpy.round(arr/counter),dtype=numpy.uint8)

  # Generate, save and preview final image
  out=Image.fromarray(arr,mode="RGB")
  print ('num img', counter, ' save ',os.path.join(args.image_path,str(label)+args.output_name+'.png'))
  out.save(os.path.join(args.image_path,str(label)+args.output_name+'.png'))



