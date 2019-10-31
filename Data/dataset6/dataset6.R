# load the gene expression matrix
chen <- get(load('GSE87544_1443737Cells.Expresssion.Matrix.log_tpm+1_.renamed.RData'))
head(chen)
dim(chen)

# load the labels of the cells
labels <- read.csv('GSE87544_1443737Cells.SVM.cluster.identity.renamed.csv')
head(labels)
str(labels)

# check if the orders of the cells in two data matrices are consistent
sum(!labels[,1]==colnames(chen))

# extract OPC and MO cells
Y<-labels[,3]
index <- Y=='MO'|Y=='OPC'
X<-chen[,index]
Y<-Y[index]

# keep the genes that can be detected in more than 30% of the cells (OPC and MO cells)
perc <- rowMeans(X > 0)
keep <- (perc > 0.3)
X <- X[keep, ]
nrow(X)

# save the names of the genes, the new expression matrix, and the new labels
gene_names<-rownames(X)
write.table(gene_names,file='gene_names.txt',quote=F,row.names=F,col.names=F)

rownames(X)<-colnames(X)<-NULL
write.table(X,file='chen_X.txt',quote=F,row.names=F,col.names=F)

Ynew<-matrix(0,ncol(X),2)
Ynew[Y=='MO',1]<-1
Ynew[Y=='OPC',2]<-1
Ynew<-as.data.frame(Ynew)
write.table(Ynew,file='chen_Y.txt',quote=F,row.names=F,col.names=F)