
# ----------------------------------------------------------------------------------------------------------------

# ! ensemble 

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyEyeOct

for modelname in b4ns448w2-4ss10lr1e-05dp0.2b32ntest1-Img+Meta2+6F+driving b4ns448w2-4ss10lr1e-05dp0.2b32ntest1-Img+Meta2+logMAR+6F+driving b4ns448w2-4ss10lr1e-05dp0.2b32ntest1-Img+Meta2+logMAR+FHScore+6F+driving
do
cd /data/duongdb/ClassifyEyeOct
modeldir="/data/duongdb/FH_OCT_08172021/Classify/"$modelname
labels='A,B,C' #'A,B,C'  '1,2,3,4'
python3 ensemble_our_classifier.py --model-dir $modeldir --labels $labels
done 
cd $modeldir


# ----------------------------------------------------------------------------------------------------------------

# ! copy images to local pc

mkdir /cygdrive/c/Users/duongdb/Documents/FH_OCT_08172021/Classify
for modelname in b4ns448w2ss10lr1e-05dp0.2b32ntest1-Img+Meta+6F+driving
do
mkdir /cygdrive/c/Users/duongdb/Documents/FH_OCT_08172021/Classify/$modelname
cd /cygdrive/c/Users/duongdb/Documents/FH_OCT_08172021/Classify/$modelname
scp -r $biowulf:/data/duongdb/FH_OCT_08172021/Classify/$modelname/*png .
scp -r $biowulf:/data/duongdb/FH_OCT_08172021/Classify/$modelname/*csv .
done 


# ----------------------------------------------------------------------------------------------------------------

# ! ensemble DECISION TREE

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyEyeOct

for modelname in DecisionTreeDriving DecisionTreeDrivingMeta2 DecisionTreeDrivingMeta2+logMAR DecisionTreeDrivingMeta2+logMAR+FH

do
cd /data/duongdb/ClassifyEyeOct
modeldir="/data/duongdb/FH_OCT_08172021/"$modelname
labels='1,2,3' #'1,2,3'  '1,2,3,4'
python3 ensemble_our_classifier.py --model-dir $modeldir --labels $labels
done 

for modelname in DecisionTreeDriving DecisionTreeDrivingMeta2 DecisionTreeDrivingMeta2+logMAR DecisionTreeDrivingMeta2+logMAR+FH 
do
mkdir /cygdrive/c/Users/duongdb/Documents/FH_OCT_08172021/$modelname
cd /cygdrive/c/Users/duongdb/Documents/FH_OCT_08172021/$modelname
scp -r $biowulf:/data/duongdb/FH_OCT_08172021/$modelname/*png .
scp -r $biowulf:/data/duongdb/FH_OCT_08172021/$modelname/*csv .
done 


# ----------------------------------------------------------------------------------------------------------------

# ! look at best eval for each model

for mod in b4ns448w2ss5lr0.0005dp0.5b32ntest1-Img+6F+driving b4ns448w2ss5lr0.0005dp0.2b32ntest1-Img+6F+driving b4ns448w2ss5lr0.0005dp0.5b32ntest1-Img+Meta+6F+driving b4ns448w2ss5lr0.0005dp0.2b32ntest1-Img+Meta+6F+driving
do
echo $mod
cat /data/duongdb/FH_OCT_08172021/Classify/$mod/log_9c_b4ns_448_30ep_eval.txt
done

