import enum
import os,sys,re,pickle
import numpy as np 
import pandas as pd 

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import metrics
from scipy.stats import pearsonr

dfcsv = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_train_input_driving.csv'
dftestname = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_test_input_match_train_col_driving.csv'

MODEL = {
  'DecisionTreeDriving':'eye_position_od,eye_position_os,machine_type_z,machine_type_hb,age_taken,logMAR,spherical_equivalent,nystagmus,dAniridia,dCHS,dErdheim Chester disease,dFH isolated,dHPS,dNanophthalmos,dOA,dOCA,dWaardenburg 2A,gGPR143,gHPS1,gMITF,gOCA2,gPAX6,gPRSS56,gSLC45A2,gTYR,gunknown'.split(','),

  'DecisionTreeDrivingMeta2':'machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus'.split(','),

  'DecisionTreeDrivingMeta2+logMAR':'machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus,logMAR'.split(','),

  'DecisionTreeDrivingMeta2+logMAR+FH':'machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus,logMAR,FHScore'.split(',')
}


for modelname in MODEL: 

  cols = MODEL[modelname]

  outputfolder = 'C:/Users/duongdb/Documents/FH_OCT_08172021/'+modelname
  if not os.path.exists(outputfolder): 
    os.mkdir(outputfolder)

  all_output = None

  bestparams = { k:{'max_depth':0} for k in [0,1,2,3,4] }

  diagnosis2idx = {val:(i+1) for i,val in enumerate(['A','B','C'])}

  for fold in [0,1,2,3,4]: 

    print ('\nfold :', fold)

    df = pd.read_csv(dfcsv,na_values="-")
    df = df.dropna()

    # ! remove fold 0, which is designed as test set. 
    df = df [df['fold'] != 5]
    df['label'] = df['label'].map(diagnosis2idx) # rename labels 
    
    dftrain = df [df['fold'] != fold]
    dfvalid = df [df['fold'] == fold]

    print('dftrain/test:', dftrain.shape[0], dfvalid.shape[0])

    X = dftrain[cols].to_numpy(dtype=float)
    Y = dftrain['label'].to_numpy(dtype=float)

    BEST_ACC = 0 
    BEST_COR = 0 

    max = len (cols)
    if max > 10: 
      max = 10
      
    for max_depth in np.arange(4,max): 
      
      clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=fold*10) # print(clf.tree_.max_depth) , class_weight={1:2,2:1,3:2}
      clf = clf.fit(X, Y)
      # tree.plot_tree(clf) 

      Xtest = dfvalid[cols].to_numpy(dtype=float)
      Ytest = dfvalid['label'].to_numpy(dtype=float)
      Ytest_pred = clf.predict(Xtest)

      acc = metrics.accuracy_score(Ytest, Ytest_pred)
      # average_acc = average_acc + acc
      # print("Accuracy:", acc)

      cor, _ = pearsonr(Ytest, Ytest_pred)
      # average_cor = average_cor + cor
      # print("Cor:",cor)

      if acc > BEST_ACC: 
        bestparams[fold]['max_depth'] = max_depth
        BEST_ACC = acc
        BEST_COR = cor
        record_best = 1
      elif acc == BEST_ACC: 
        if cor > BEST_COR: 
          bestparams[fold]['max_depth'] = max_depth
          BEST_ACC = acc
          BEST_COR = cor
          record_best = 1
      else: 
        record_best = 0 

      if record_best == 1: 
        
        Ytest_pred_score = np.matmul (clf.predict_proba(Xtest), np.reshape([1,2,3],(3,1))).flatten()
    
        # ! write out as csv. 
        dfout = {'average_score':Ytest_pred_score.tolist(), 
                'label':Ytest.tolist(),
                'fold':[fold]*dfvalid.shape[0] }

        dfout = pd.DataFrame.from_dict(dfout)

        dfout.to_csv(os.path.join(outputfolder,'DecisionTreeOutputFold'+str(fold)+'.csv'),index=False)

        # ! features 
        feature_score = dict(zip(cols, clf.feature_importances_))
        feature_order = sorted(feature_score, key=feature_score.get)
        feature_order.reverse()
        # print (feature_order)
        
        text_representation = tree.export_text(clf)
        with open(os.path.join(outputfolder,"decistion_tree"+str(fold)+".log"), "w") as fout:
          fout.write(','.join(feature_order)+'\n\n')
          fout.write(text_representation)

        pickle.dump(clf, open(os.path.join(outputfolder,"decistion_tree"+str(fold)+".pickle"), "wb") )

    # ! see best
    print ('\n\nbest fold, acc ', fold , ' ', BEST_ACC)


  # ! on test 

  print ('\n\n',bestparams)

  df = pd.read_csv(dftestname,na_values="-")
  df = df.dropna()
  df['label'] = df['label'].map(diagnosis2idx) # rename labels 

  diagnosis2idx = {val:(i+1) for i,val in enumerate(['A','B','C'])}

  for fold in [0,1,2,3,4]: 

    print ('\nfold :', fold)

    # df = pd.read_csv(dfcsv,na_values="-")
    # df = df.dropna()
    # df['label'] = df['label'].map(diagnosis2idx) # rename labels 

    dfrealtest = df[df['fold'] == 5] # ! just to be sure
    dfrealtest.reset_index(inplace = True, drop=True)

    # ! load model 
    clf = pickle.load(open(os.path.join(outputfolder,"decistion_tree"+str(fold)+".pickle"), "rb") )

    Xtest = dfrealtest[cols].to_numpy(dtype=float)
    Ytest = dfrealtest['label'].to_numpy(dtype=float)
    Ytest_pred = clf.predict(Xtest)

    acc = metrics.accuracy_score(Ytest, Ytest_pred)
    # average_acc = average_acc + acc
    print("Accuracy:", acc)

    cor, _ = pearsonr(Ytest, Ytest_pred)
    # average_cor = average_cor + cor
    print("Cor:",cor)

    PROBS = clf.predict_proba(Xtest)
    prob_df = pd.DataFrame( PROBS, columns=[str(i) for i in np.arange(3)]) # @PROBS is shape=(1, obs, labelsize)

    Ytest_pred_score = np.matmul (clf.predict_proba(Xtest), np.reshape([1,2,3],(3,1))).flatten()

    # ! write out as csv. 
    dfout = {'average_score':Ytest_pred_score.tolist(), 
            'label':Ytest.tolist(),
            'fold':[fold]*dfrealtest.shape[0] }

    dfout = pd.DataFrame.from_dict(dfout)

    dfout = pd.concat([dfrealtest,dfout, prob_df], axis=1)

    dfout.to_csv(os.path.join(outputfolder,'test_on_fold_5_from_fold'+str(fold)+'.csv'),index=False)

