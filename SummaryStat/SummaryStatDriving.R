library(Hmisc)
library(ggplot2)

fin = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_train_input.csv'
fin = read.csv(fin)
table ( fin$label ) 
fin = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_test_input.csv'
fin = read.csv(fin)
table (fin$label )

x = as.numeric(unlist(fin$Grade.OS))

colors=1:3
names(colors)=c('A','B','C')
fin$Driving.Index=colors[fin$label]

fin = fin[fin$fold==5,] # ! cheat and look at real test set. 
dim(fin)


x = as.numeric(unlist(fin$logMAR))
y = as.numeric(unlist(fin$Driving.Index))

reg1 <- lm(y~x)
boxplot ( x, y, pch=16, cex=1, xlab=paste('logMAR',round ( summary(reg1)$r.squared, 4 )), ylab='', cex.lab = 1.25)
abline(reg1, col="blue", lwd=2)
print (summary(reg1))



p<-ggplot(fin, aes(x=logMAR, y=Driving.Index)) + 
  geom_jitter(position=position_jitter(0.2)) + 
  geom_smooth(aes(x=logMAR, y=Driving.Index),method='lm') +
  theme(text = element_text(size = 15))   

p



