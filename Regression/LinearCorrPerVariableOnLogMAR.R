
df = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_6fold.csv'
df = read.csv(df, stringsAsFactors=FALSE)

cols = c ('age_taken', 'spherical_equivalent', 'nystagmus' )

for (c in cols) {
  print (c)
  print (cor( as.numeric( unlist(df[c]) ) , as.numeric( unlist(df['logMAR']) ) , use="complete.obs" ) )
}

par(mfrow = c(1, 3))
for (c in cols) {
  x = as.numeric( unlist( df[c] ) )
  y = as.numeric( unlist(df['logMAR']) )
  reg1 <- lm(y~x)
  plot ( x, y, pch=16, cex=1, xlab=paste(c,round ( summary(reg1)$r.squared, 4 )), ylab='logMAR', cex.lab = 1.25)
  abline(reg1, col="blue", lwd=2)
  print (summary(reg1))
}



df = 'C:/Users/duongdb/Documents/FH_OCT_08172021/FH_OCTs_label_6fold+driving.csv'
df = read.csv(df, stringsAsFactors=FALSE)

