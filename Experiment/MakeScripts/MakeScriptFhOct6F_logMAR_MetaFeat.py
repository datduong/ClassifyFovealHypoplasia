import re,sys,os,pickle
from datetime import datetime
import time

# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=24
# sbatch --partition=gpu --time=1:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=10g --cpus-per-task=20
# sbatch --time=12:00:00 --mem=100g --cpus-per-task=24
# sinteractive --time=4:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12

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

suffix=SUFFIX
fold=FOLD

# metafeat='METAFEAT'
metadim='METADIM'

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

python3 train_logmar.py --image-csv $imagecsv --kernel-type $kernel_type --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0 --model-dir $modeldir --log-dir $logdir --num-workers 4 --fold 'FOLD' --out-dim 1 --weighted-loss $weight --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest --loaded-model $loaded_model --linear-loss --meta-features 'METAFEAT' --n-meta-dim $metadim --new-label logMAR

# --label-upweigh LABELUP 

# ! eval

imagecsv=/data/duongdb/FH_OCT_08172021/FH_OCTs_label_test_input_match_train_col.csv # ! test input, same as train, we just take fold 0 ? we don't need to, we just need a pre-trained model

python evaluate_logmar.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 32 --num-workers 4 --fold 'FOLD' --out-dim 1 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n-test $ntest --linear-loss --meta-features 'METAFEAT' --n-meta-dim $metadim --new-label logMAR # ! actual test set

# python evaluate_logmar.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 32 --num-workers 4 --fold 'FOLD' --out-dim 1 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --n-test $ntest --linear-loss --meta-features 'METAFEAT' --n-meta-dim $metadim # ! on the left out fold


"""

path = '/data/duongdb/FH_OCT_08172021'
os.chdir(path)

SUFFIX = 'Img+Meta2+FHScore+6F+logMAR'

# LABELUP = '22q11DS,Controls,WS'

METAFEAT = 'machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus,FHScore'

# METAFEAT = 'eye_position_od,eye_position_os,machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus,dAniridia,dCHS,dErdheim Chester disease,dFH isolated,dHPS,dNanophthalmos,dOA,dOCA,dWaardenburg 2A,gGPR143,gHPS1,gMITF,gOCA2,gPAX6,gPRSS56,gSLC45A2,gTYR,gunknown' # ! not ok with logMAR here 
METADIM = '128,128'

# {'gCNGB3', 'gHPS5', 'dAchromatopsia'} # in test no train

METADIM = '128,128'

counter=0
for fold in [0,1,2,3,4]: 
  for imagesize in [448]: # 448 512 768 640
    for weight in [1]: # 5,10,
      for schedulerscaler in [5]:
        for learn_rate in [0.0005]:  # 0.00001,0.00003  # we used this too, 0.0001
          for dropout in [0.2]: # 0.2
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
            time.sleep( 2 )
            os.system('sbatch --partition=gpu --time=4:00:00 --gres=gpu:p100:1 --mem=10g --cpus-per-task=8 ' + scriptname )
            # os.system('sbatch --partition=gpu --time=00:10:00 --gres=gpu:p100:1 --mem=4g --cpus-per-task=4 ' + scriptname )
            # os.system('sbatch --time=16:00:00 --mem=96g --cpus-per-task=20 ' + scriptname )
            counter = counter + 1 

#
exit()
