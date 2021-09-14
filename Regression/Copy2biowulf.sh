
for modelname in DecisionTreeOutputDriving  
do 
modelpath=/cygdrive/c/Users/duongdb/Documents/FH_OCT_08172021
scp -r $modelpath/$modelname $biowulf:$datadir/FH_OCT_08172021
done

