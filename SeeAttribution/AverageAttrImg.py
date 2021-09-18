
import os, numpy, PIL 
import re
import numpy as np 
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str) # /data/duongdb/WS22qOther_05052021/Classify/b4ns448Wl1ss10lr0.0001dp0.2b48ntest1
parser.add_argument('--fold', type=str, help='something like 1,2,3')
parser.add_argument('--keyword', type=str, default='Sign')

args = parser.parse_args()
    
maindir = os.path.join(args.model_dir,'EvalDev')

# ! because we use 5-fold cv. we can average attribution for each fold. 

for level in ['_test_Occlusion2']: # _test_1 _test_10

  outdir = os.path.join(args.model_dir,'AverageAttr'+level)
  
  if not os.path.exists(outdir): 
    os.mkdir(outdir)

  # 
  fold = [ str(i.strip()) + level for i in args.fold.split(',')]

  os.chdir(maindir)

  imlist_in_1_fold = sorted ( os.listdir(os.path.join(maindir,fold[0])) ) 
  imlist_in_1_fold = [i for i in imlist_in_1_fold if args.keyword in i] # Positive Sign

  # ! 
  # this_img = imlist_in_1_fold[0]
  for this_img in imlist_in_1_fold: 

    try: 
      imlist = [os.path.join(maindir,i,this_img) for i in fold]

      # Assuming all images are the same size, get dimensions of first image
      w,h=Image.open(imlist[0]).size
      N=len(imlist)

      # Create a numpy array of floats to store the average (assume RGB images)
      arr=numpy.zeros((h,w,3),numpy.float)

      # Build up average pixel intensities, casting each image as an array of floats
      for im in imlist:
        imarr=numpy.array(Image.open(im),dtype=numpy.float)
        arr=arr+imarr/N

      # Round values in array and cast as 8-bit integer
      arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

      # Generate, save and preview final image
      out=Image.fromarray(arr,mode="RGB")
      out.save(os.path.join(outdir,re.sub(r"(\.png|\.jpg)","",this_img)) + "Average"+args.keyword+".png")
      # out.show()

    except: # ! may not have all images done?  
      pass 
