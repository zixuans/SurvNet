set.seed(1234)
library(knockoff)
library(corpcor)
library(RSpectra)
library(Rdsdp)
library(gtools)
for(i in 1:25) {
    X<-read.table(paste("data/X_64_",as.character(i),".txt",sep=""))
    X<-scale(X)
    X1<-X[1:8000,] #exclude the test sets
    X_k<-create.second_order(X1,shrink=T)
    write.table(X_k,file=paste("data/Xk_64_",as.character(i),".txt",sep=""),quote=F,row.names=F,col.names=F)
}