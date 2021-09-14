

cd /data/duongdb/ClassifyEyeOct/SeeAttribution

modeldir=/data/duongdb/FH_OCT_08172021/Classify/b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+6F+SoftM0.9T0.8
python3 AverageAttrImg.py --model-dir $modeldir --fold 0,1,2,3,4



# average all images 

cd /data/duongdb/ClassifyEyeOct/SeeAttribution

image_path=/data/duongdb/FH_OCT_08172021/Classify/b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+6F+SoftM0.9T0.8/AverageAttr_test_Occlusion2
label_file=/data/duongdb/FH_OCT_08172021/FH_OCTs_label_test_input_match_train_col.csv
# for keyword in OS_Z
# do
output_name='average_' # $keyword
python3 AverageAllAttr.py --image_path $image_path --output_name $output_name --label_file $label_file
# done 
cd $image_path
