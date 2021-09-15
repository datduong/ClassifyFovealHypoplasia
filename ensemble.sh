
# ----------------------------------------------------------------------------------------------------------------

# ! ensemble 

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyEyeOct # ! change to your own path

for modelname in b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+6F b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+6F+SoftM0.9T0.8
do
  cd /data/duongdb/ClassifyEyeOct
  modeldir="/data/duongdb/FH_OCT_08172021/Classify/"$modelname
  labels='1,2,3,4' # ! FH score uses '1,2,3,4' whereas driving uses 'A,B,C' 
  python3 ensemble_our_classifier.py --model-dir $modeldir --labels $labels
done 
cd $modeldir

