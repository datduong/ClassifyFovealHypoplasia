
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

maindir=/data/duongdb/FH_OCT_08172021/

for fold in 0 1 2 3 4 
do

labels='1,2,3,4'

mix=M0.9T0.8

img_data_path=$maindir/Classify/Soft+FH+Eye$mix

fout_path=$maindir
datatype=train
suffix=FHGanSoftF$fold$mix

original_train_csv=$maindir/FH_OCTs_label_train_input.csv

cd /data/duongdb/ClassifyEyeOct/MakeOurData
python3 MakeCsvAddGanSoft.py --fout_path $fout_path --img_data_path $img_data_path --datatype $datatype --suffix $suffix --fold $fold --original_train_csv $original_train_csv --labels $labels

done 
cd $fout_path

# --------------------------------------------

# ! driving data 


source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

maindir=/data/duongdb/FH_OCT_08172021/

mix=M0.9T0.8

for fold in 0 1 2 3 4 
do

labels='A,B,C'

img_data_path=$maindir/Classify/Soft+Driving+Eye$mix  

fout_path=$maindir
datatype=train
suffix=DrivingGanSoftF$fold$mix

original_train_csv=$maindir/FH_OCTs_label_train_input_driving.csv

cd /data/duongdb/ClassifyEyeOct/MakeOurData
python3 MakeCsvAddGanSoft.py --fout_path $fout_path --img_data_path $img_data_path --datatype $datatype --suffix $suffix --fold $fold --original_train_csv $original_train_csv --labels $labels

done 
cd $fout_path



