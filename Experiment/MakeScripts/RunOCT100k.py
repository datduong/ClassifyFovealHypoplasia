import re,sys,os,pickle
from datetime import datetime
import time

script = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! check model name
weight=WEIGHT
learningrate=LEARNRATE
imagesize=IMAGESIZE
schedulerscaler=ScheduleScaler 
dropout=DROPOUT

batchsize=32 # 64 ... 64 doesn't work with new pytorch 1.7 ?? why ?? we were using 1.6 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=9c_b4ns_$imagesize'_30ep' # ! this is experiment name

# ! where to save
model_folder_name=Oct100kb4ns$imagesize$imagetype'Wl'$weight'ss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest

maindir=/data/duongdb/FH_OCT_08172021/Classify

modeldir=$maindir/$model_folder_name 
mkdir $modeldir

logdir=$maindir/$model_folder_name 
oofdir=$maindir/$model_folder_name/EvalDev 

cd /data/duongdb/ClassifyEyeOct

# ! train

imagecsv=/data/duongdb/OctAndChestXrayImages/CellData/OCT/train_stratify.csv # ! train input 

python train.py --image-csv $imagecsv --kernel-type $kernel_type --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0 --model-dir $modeldir --log-dir $logdir --num-workers 8 --fold 'FOLD' --out-dim 4 --weighted-loss $weight --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest 

"""

path = '/data/duongdb/FH_OCT_08172021'
os.chdir(path)

counter=0
for fold in [0]: # 0,0,1,2,3,4 '0','1','2','3','4'
  for imagesize in [448]: # 448 512 768 640
    for weight in [1]: # 5,10,
      for schedulerscaler in [10]:
        for learn_rate in [0.00001]:  # 0.00001,0.00003  # we used this too, 0.0001
          for dropout in [0.2]:
            script2 = re.sub('WEIGHT',str(weight),script)
            script2 = re.sub('IMAGESIZE',str(imagesize),script2)
            script2 = re.sub('LEARNRATE',str(learn_rate),script2)
            script2 = re.sub('ScheduleScaler',str(schedulerscaler),script2)
            script2 = re.sub('FOLD',str(fold),script2)
            script2 = re.sub('DROPOUT',str(dropout),script2)
            now = datetime.now() # current date and time
            scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
            fout = open(scriptname,'w')
            fout.write(script2)
            fout.close()
            # 
            time.sleep( 1 )
            # os.system('sbatch --partition=gpu --time=16:00:00 --gres=gpu:v100x:1 --mem=12g --cpus-per-task=16 ' + scriptname )
            counter = counter + 1 

#
exit()

