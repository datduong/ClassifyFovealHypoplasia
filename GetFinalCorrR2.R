
library('ggplot2')

# ---------------------------------------------------------------------------------------------------------

mods = c('b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+6F+SoftM0.9T0.8', 'b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+6F+Soft', 'b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+Meta2+6F' , 'b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+Meta2+logMAR+6F')

for (model in mods) {
  print (model)
  cor_ = c() 
  r2 = c() 

  df_ = read.csv(paste0('C:/Users/duongdb/Documents/FH_OCT_08172021/Classify/',model,'/final_prediction.csv'))
  cor_ = c(cor_,cor(df_$label, df_$average_score, method = c("spearman"))) # pearson spearman
  lmmod = lm(df_$label ~ df_$average_score)
  r2 = c(r2, summary(lmmod)$r.squared)
  
  print (summary(cor_))
  print (summary(r2))
  print (table(df_$label))
}


# ---------------------------------------------------------------------------------------------------------

# ! decision tree... same code as neural net can be used here. 
mods = c('DecisionTreeDriving', 'DecisionTreeDrivingMeta2', 'DecisionTreeDrivingMeta2+logMAR', 'DecisionTreeDrivingMeta2+logMAR+FH')

for (model in mods) {
  print (model)
  cor_ = c() 
  r2 = c() 

  df_ = read.csv(paste0('C:/Users/duongdb/Documents/FH_OCT_08172021/',model,'/final_prediction.csv'))
  cor_ = c(cor_,cor(df_$label, df_$average_score, method = c("spearman"))) # pearson spearman
  lmmod = lm(df_$label ~ df_$average_score)
  r2 = c(r2, summary(lmmod)$r.squared)
  
  print (summary(cor_))
  print (summary(r2))
  print (table(df_$label))
}

# ---------------------------------------------------------------------------------------------------------


# ! predict logMAR

mods = c('b4ns448w1ss5lr0.0005dp0.2b32ntest1-Img+6F+logMAR', 'b4ns448w1ss5lr0.0005dp0.2b32ntest1-Img+Meta+6F+logMAR', 'b4ns448w1ss5lr0.0005dp0.2b32ntest1-Img+Meta2+6F+logMAR', 'b4ns448w1ss5lr0.0005dp0.2b32ntest1-Img+Meta2+FHScore+6F+logMAR')

for (model in mods) {
  print (model)
  cor_ = c() 
  r2 = c() 
  average_score = NULL 
  for (fold in c(0,1,2,3,4)){
    df_ = read.csv(paste0('C:/Users/duongdb/Documents/FH_OCT_08172021/Classify/',model,'/test_on_fold_0_from_fold',fold,'.csv'))
    if (is.null(average_score)){ 
      average_score = unlist(df_['X0']) 
    } else { 
      average_score = unlist(df_['X0']) + average_score 
    }
  }
  average_score = average_score / 5
  df_$average_score = average_score
  cor_ = c(cor_,cor(df_$logMAR, df_$average_score, method = c("spearman"))) # pearson spearman
  lmmod = lm(df_$logMAR ~ df_$average_score)
  r2 = c(r2, summary(lmmod)$r.squared)
  print (summary(cor_))
  print (summary(r2))
  # print (summary(lmmod))
}


# ---------------------------------------------------------------------------------------------------------

# ! regression predict logMAR 

mods = c('ElasticNetLogMAR','ElasticNetLogMARMeta2','ElasticNetLogMARMeta2+FH','ElasticNetLogMAR+FH')

# mods = c('ElasticNetLogMAR+FH')
for (model in mods) {
  print (model)
  cor_ = c() 
  r2 = c() 
  average_score = NULL 
  for (fold in c(0,1,2,3,4)){
    df_ = read.csv(paste0('C:/Users/duongdb/Documents/FH_OCT_08172021/',model,'/test_on_fold_5_from_fold',fold,'.csv'))
    if (is.null(average_score)){ 
      average_score = unlist(df_['average_score']) 
    } else { 
      average_score = unlist(df_['average_score']) + average_score 
    }
  }
  average_score = average_score / 5
  df_$average_score = average_score
  cor_ = c(cor_,cor(df_$logMAR, df_$average_score, method = c("spearman"))) # ! call @logMAR instead of @label 
  lmmod = lm(df_$logMAR ~ df_$average_score)
  r2 = c(r2, summary(lmmod)$r.squared)
  print (summary(cor_))
  print (summary(r2))
  # print (summary(lmmod))
}


# ---------------------------------------------------------------------------------------------------------

mods = c('b4ns448w2-4ss10lr1e-05dp0.2b32ntest1-Img+Meta2+6F+driving','b4ns448w2-4ss10lr1e-05dp0.2b32ntest1-Img+Meta2+logMAR+6F+driving','b4ns448w2-4ss10lr1e-05dp0.2b32ntest1-Img+Meta2+logMAR+FHScore+6F+driving')
for (model in mods) {
  print (model)
  cor_ = c() 
  r2 = c() 

  df_ = read.csv(paste0('C:/Users/duongdb/Documents/FH_OCT_08172021/Classify/',model,'/final_prediction.csv'))
  cor_ = c(cor_,cor(df_$label, df_$average_score, method = c("spearman"))) # pearson spearman
  lmmod = lm(df_$label ~ df_$average_score)
  r2 = c(r2, summary(lmmod)$r.squared)
  
  print (summary(cor_))
  print (summary(r2))
  print (table(df_$label))
  plot (df_$label,df_$average_score)
  boxplot (df_$average_score~df_$label, xlab='True label', ylab='Average score', ylim=c(1,3))
}
