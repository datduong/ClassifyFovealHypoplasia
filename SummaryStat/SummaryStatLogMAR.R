library(Hmisc)
library(ggplot2)

fin = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_test_input.csv'
fin = read.csv(fin)

fin = fin[fin['fold']==5,]

fin$logMAR = as.numeric(unlist(fin['logMAR']))
summary ( fin$logMAR )

fin$FHScore = as.numeric(unlist(fin['FHScore']))

hist(fin$logMAR,30,xlab='logMAR',main='Histogram logMAR')

summary(lm(fin$logMAR~fin$FHScore))

plot(fin$logMAR~fin$FHScore, xlab='FHScore', ylab='logMAR', main='logMAR v.s. FHScore', cex=1, pch=16)

cor(fin$logMAR, fin$FHScore, use="complete.obs")


p<-ggplot(fin, aes(x=FHScore, y=logMAR)) + 
  geom_jitter(position=position_jitter(0.2)) + 
  geom_smooth(aes(x=FHScore, y=logMAR),method='lm') +
  theme(text = element_text(size = 15))   

p


