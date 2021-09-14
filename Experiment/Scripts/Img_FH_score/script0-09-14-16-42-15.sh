#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! check model name
weight=1
learningrate=1e-05
imagesize=448
schedulerscaler=10 
dropout=0.2

batchsize=32 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=b4ns_$imagesize'_30ep' # ! this is experiment name

suffix=Img+6F
fold=0

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

python train.py --image-csv $imagecsv --kernel-type $kernel_type --image-size $imagesize --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0 --model-dir $modeldir --log-dir $logdir --num-workers 4 --fold '0' --out-dim 4 --weighted-loss $weight --n-epochs 30 --batch-size $batchsize --init-lr $learningrate --scheduler-scaler $schedulerscaler --dropout $dropout --n-test $ntest --loaded-model $loaded_model

# ! eval

imagecsv=/data/duongdb/FH_OCT_08172021/FH_OCTs_label_test_input.csv # ! test input

python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 32 --num-workers 4 --fold '0' --out-dim 4 --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n-test $ntest # ! actual test set

# ! look at pixels

for condition in Z HB
do
python evaluate.py --image-csv $imagecsv --kernel-type $kernel_type --model-dir $modeldir --log-dir $logdir --image-size $imagesize --enet-type tf_efficientnet_b4_ns --oof-dir $oofdir --batch-size 1 --num-workers 8 --fold '0' --out-dim 4 --dropout $dropout --do_test --n-test $ntest --attribution_keyword $condition --outlier_perc 2 --attribution_model Occlusion
done

