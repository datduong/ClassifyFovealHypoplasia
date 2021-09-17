
library('ggplot2')

# ---------------------------------------------------------------------------------------------------------

# ! get correlation and regression R2 on neural network models that predict FH scores or Driving score. 

mods = c('b4ns448w1ss10lr1e-05dp0.2b32ntest1-Img+6F') # ! put your model name here. 

for (model in mods) { # ! go over each model name
  print (model)
  cor_ = c() 
  r2 = c() 

  df_ = read.csv(paste0('C:/Users/duongdb/Documents/FH_OCT_08172021/Classify/',model,'/final_prediction.csv')) # ! read in final prediction (change csv name if needed.)
  cor_ = c(cor_,cor(df_$label, df_$average_score, method = c("spearman"))) # pearson spearman
  lmmod = lm(df_$label ~ df_$average_score) # linear regression R2
  r2 = c(r2, summary(lmmod)$r.squared)
  
  print (summary(cor_))
  print (summary(r2))
  print (table(df_$label))
}



# ---------------------------------------------------------------------------------------------------------

# ! get correlation and regression R2 on neural network models that predict logMAR. 

mods = c('b4ns448w1ss5lr0.0005dp0.2b32ntest1-Img+6F+logMAR') # ! put your model name here. 

for (model in mods) { # ! go over each model name
  print (model)
  cor_ = c() 
  r2 = c() 
  average_score = NULL 
  for (fold in c(0,1,2,3,4)){ # ! go over each fold.
    df_ = read.csv(paste0('C:/Users/duongdb/Documents/FH_OCT_08172021/Classify/',model,'/test_on_fold_0_from_fold',fold,'.csv')) # ! take average logMAR over 5 folds
    if (is.null(average_score)){ 
      average_score = unlist(df_['X0']) 
    } else { 
      average_score = unlist(df_['X0']) + average_score 
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

# ! get correlation and regression R2 on ELASTIC NET ( NOT NEURAL NETWORK ) models that predict logMAR. 

mods = c('ElasticNetLogMAR') # ! put your model name here. 

for (model in mods) { # ! go over each model name
  print (model)
  cor_ = c() 
  r2 = c() 
  average_score = NULL 
  for (fold in c(0,1,2,3,4)){
    df_ = read.csv(paste0('C:/Users/duongdb/Documents/FH_OCT_08172021/',model,'/test_on_fold_5_from_fold',fold,'.csv')) # ! take average logMAR over 5 folds
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

