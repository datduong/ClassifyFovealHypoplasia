import re,sys,os,pickle
from datetime import datetime
import time

# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=24
# sbatch --partition=gpu --time=1:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=10g --cpus-per-task=20
# sbatch --time=12:00:00 --mem=100g --cpus-per-task=24
# sinteractive --time=1:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12

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

batchsize=32 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=b4ns_$imagesize'_30ep' # ! this is experiment name

suffix=SUFFIX
fold=FOLD

# ! check if we use 60k or not
model_folder_name=b4ns$imagesize'w'$weight'ss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest'-'$suffix

maindir=/data/duongdb/FH_OCT_08172021/Classify

modeldir=$maindir/$model_folder_name 
mkdir $modeldir

logdir=$maindir/$model_folder_name 
oofdir=$maindir/$model_folder_name/EvalDev 

cd /data/duongdb/ClassifyEyeOct

# ! train

loaded_model=/data/duongdb/FH_OCT_08172021/Classify/Oct100kb4ns448Wl1ss10lr1e-05dp0.2b32ntest1/9c_b4ns_448_30ep_best_all_fold0.pth # ! load pre-trained from 100k oct

imagecsv=/data/duongdb/FH_OCT_08172021/FH_OCTs_label_train_input.csv # ! train input 

python train.py --image-csv $imagecsv --kernel-type $kernel_type --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0 --model-dir $modeldir --log-dir $logdir --num-workers 4 --fold 'FOLD' --out-dim 4 --weighted-loss $weight --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest --loaded-model $loaded_model

# ! eval

imagecsv=/data/duongdb/FH_OCT_08172021/FH_OCTs_label_test_input.csv # ! test input

python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 32 --num-workers 4 --fold 'FOLD' --out-dim 4 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n-test $ntest # ! actual test set

# ! look at pixels

for condition in Z HB
do
python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 1 --num-workers 8 --fold 'FOLD' --out-dim 4 --dropout $dropout --do_test --n-test $ntest --attribution_keyword $condition --outlier_perc 2 --attribution_model Occlusion
done

"""


path = '/data/duongdb/FH_OCT_08172021'
os.chdir(path)

SUFFIX = 'Img+6F'

# LABELUP = '22q11DS,Controls,WS'

METAFEAT = 'eye_position_od,eye_position_os,machine_type_z,machine_type_hb,age_taken,logMAR,spherical_equivalent,nystagmus,dAchromatopsia,dAniridia,dCHS,dErdheim Chester disease,dFH isolated,dHPS,dNanophthalmos,dOA,dOCA,dWaardenburg 2A,gCNGB3,gGPR143,gHPS1,gHPS5,gMITF,gOCA2,gPAX6,gPRSS56,gSLC45A2,gTYR,gunknown'
METADIM = '128,128'

counter=0
for fold in [0,1,2,3,4]: 
  for imagesize in [448]: # 
    for weight in [1]: # 
      for schedulerscaler in [10]:
        for learn_rate in [0.00001]:  # 0.00001,0.00003  # we used this too, 0.0001
          for dropout in [0.2]:
            script2 = re.sub('WEIGHT',str(weight),script)
            # script2 = re.sub('LABELUP',str(LABELUP),script2)
            script2 = re.sub('IMAGESIZE',str(imagesize),script2)
            script2 = re.sub('METAFEAT',str(METAFEAT),script2)
            script2 = re.sub('METADIM',str(METADIM),script2)
            script2 = re.sub('SUFFIX',str(SUFFIX),script2)
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
            time.sleep( 4 )
            # os.system('sbatch --partition=gpu --time=4:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8 ' + scriptname )
            # os.system('sbatch --partition=gpu --time=00:10:00 --gres=gpu:p100:1 --mem=4g --cpus-per-task=4 ' + scriptname )
            # os.system('sbatch --time=24:00:00 --mem=12g --cpus-per-task=20 ' + scriptname )
            counter = counter + 1 

#

exit()


