import os,sys,re,pickle
import numpy as np 
import pandas as pd 

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import metrics

from sklearn.linear_model import ElasticNet

from scipy.stats import pearsonr

dfcsv = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_train_input.csv'
dftestname = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_test_input_match_train_col.csv'

MODEL = {
  'ElasticNetLogMAR':'eye_position_od,eye_position_os,machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus,dAniridia,dCHS,dErdheim Chester disease,dFH isolated,dHPS,dNanophthalmos,dOA,dOCA,dWaardenburg 2A,gGPR143,gHPS1,gMITF,gOCA2,gPAX6,gPRSS56,gSLC45A2,gTYR,gunknown'.split(','),

  'ElasticNetLogMARMeta2':'machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus'.split(','),

  'ElasticNetLogMARMeta2+FH':'machine_type_z,machine_type_hb,age_taken,spherical_equivalent,nystagmus,FHScore'.split(','), 

  'ElasticNetLogMAR+FH':'FHScore'.split(',')
}

for modelname in MODEL: 

  cols = MODEL[modelname]

  outputfolder = 'C:/Users/duongdb/Documents/FH_OCT_08172021/'+modelname
  if not os.path.exists(outputfolder): 
    os.mkdir(outputfolder)

  bestparams = { k:{'alpha':0} for k in [0,1,2,3,4] }

  for fold in [0,1,2,3,4]: 

    print ('\nfold :', fold)

    df = pd.read_csv(dfcsv,na_values="-")
    df = df.dropna()

    # ! remove fold 0, which is designed as test set. 
    df = df [df['fold'] != 5]
    
    dftrain = df [df['fold'] != fold]
    dfvalid = df [df['fold'] == fold]

    print('dftrain/test:', dftrain.shape[0], dfvalid.shape[0])

    X = dftrain[cols].to_numpy(dtype=float)
    Y = dftrain['logMAR'].to_numpy(dtype=float)

    BEST_ACC = -1 
    BEST_COR = -1 

    thisrange = np.arange(0,.01,step=.001)
    if len (cols) == 1: 
      thisrange = [0.0]
    
    for alpha in thisrange: 
      
      clf = ElasticNet(alpha=alpha, l1_ratio=0, max_iter=20000, tol=0.001, random_state=fold*10) # print(clf.tree_.alpha) l1_ratio=alpha, random_state=fold*10
      clf = clf.fit(X, Y)

      Xtest = dfvalid[cols].to_numpy(dtype=float)
      Ytest = dfvalid['logMAR'].to_numpy(dtype=float)
      Ytest_pred = clf.predict(Xtest)

      acc = clf.score(Xtest, Ytest)
      # print("R2 score:", acc)

      cor, _ = pearsonr(Ytest, Ytest_pred)
      # print("Cor:",cor)

      if acc > BEST_ACC: 
        bestparams[fold]['alpha'] = alpha
        BEST_ACC = acc
        BEST_COR = cor
        record_best = 1
      elif acc == BEST_ACC: 
        if cor > BEST_COR: 
          bestparams[fold]['alpha'] = alpha
          BEST_ACC = acc
          BEST_COR = cor
          record_best = 1
      else: 
        record_best = 0 

      if record_best == 1: 
        
        # ! write out as csv. 
        dfout = {'average_score':Ytest_pred.tolist(), 
                'label':Ytest.tolist(),
                'fold':[fold]*dfvalid.shape[0] }

        dfout = pd.DataFrame.from_dict(dfout)

        dfout.to_csv(os.path.join(outputfolder,'DecisionTreeOutputFold'+str(fold)+'.csv'),index=False)

        # ! features 
        feature_score = dict(zip(cols, clf.coef_))
        feature_order = sorted(feature_score, key=feature_score.get)
        feature_order.reverse()
        # print (feature_order)
        
        with open(os.path.join(outputfolder,"decistion_tree"+str(fold)+".log"), "w") as fout:
          fout.write(','.join(feature_order)+'\n\n')

        pickle.dump(clf, open(os.path.join(outputfolder,"decistion_tree"+str(fold)+".pickle"), "wb") )

    # ! see best
    # print ('best fold, R2 ', fold , ' ', BEST_ACC)

  # ! best params
  print ('\n\n')
  print (bestparams)

  # ! run on test set. 
  df = pd.read_csv(dftestname,na_values="-")
  df = df.dropna()

  for fold in [0,1,2,3,4]: 

    print ('\nfold :', fold)

    # df = pd.read_csv(dfcsv,na_values="-")
    # df = df.dropna()

    dfrealtest = df[df['fold'] == 5]
    dfrealtest.reset_index(inplace = True, drop=True)

    # ! load model 
    clf = pickle.load(open(os.path.join(outputfolder,"decistion_tree"+str(fold)+".pickle"), "rb") )

    Xtest = dfrealtest[cols].to_numpy(dtype=float)
    Ytest = dfrealtest['logMAR'].to_numpy(dtype=float)
    Ytest_pred = clf.predict(Xtest)

    acc = clf.score(Xtest, Ytest)
    print("R2 score:", acc)

    cor, _ = pearsonr(Ytest, Ytest_pred)
    print("Cor:",cor)

    # ! write out as csv. 
    dfout = {'average_score':Ytest_pred.tolist(), 
            'label':Ytest.tolist(),
            'fold':[fold]*dfrealtest.shape[0] }

    dfout = pd.DataFrame.from_dict(dfout)

    dfout = pd.concat([dfrealtest,dfout], axis=1)

    dfout.to_csv(os.path.join(outputfolder,'test_on_fold_5_from_fold'+str(fold)+'.csv'),index=False)


