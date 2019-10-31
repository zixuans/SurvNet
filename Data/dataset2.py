import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing

from tensorflow.examples.tutorials.mnist import input_data

seed=1 # set a seed

# load the MNIST dataset
mnist=input_data.read_data_sets("../MNIST_data/",one_hot=True)
train_X,train_Y,test_X,test_Y,val_X,val_Y=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,mnist.validation.images,mnist.validation.labels

# extract the data of digit 0
train_index=train_Y[:,0]==1
train_X=train_X[train_index]
test_index=test_Y[:,0]==1
test_X=test_X[test_index]
val_index=val_Y[:,0]==1
val_X=val_X[val_index]
p0=train_X.shape[1] # the number of original variables

# permute (shuffle) the data
n1=train_X.shape[0]
n2=test_X.shape[0]
n3=val_X.shape[0]
t1=np.random.RandomState(seed).permutation(n1)
t2=np.random.RandomState(seed).permutation(n2)
t3=np.random.RandomState(seed).permutation(n3)
train_X=train_X[t1]
test_X=test_X[t2]
val_X=val_X[t3]

random.seed(seed)
art=np.array(random.sample(range(p0),32)) # the indexes of significant variables

# make the variables significant by shifting their values at half of the samples
np.random.seed(seed)
sign=np.random.choice([-1,1],len(art))
u=np.random.uniform(0.1,0.3,len(art))
u=u*sign
mat1=np.reshape(np.tile(u,int(0.5*n1)),(int(0.5*n1),len(art)))
mat2=np.reshape(np.tile(u,int(0.5*n2)),(int(0.5*n2),len(art)))
mat3=np.reshape(np.tile(u,int(0.5*n3)),(int(0.5*n3),len(art)))
train_X[:int(0.5*n1),art]=train_X[:int(0.5*n1),art]+mat1
test_X[:int(0.5*n2),art]=test_X[:int(0.5*n2),art]+mat2
val_X[:int(0.5*n3),art]=val_X[:int(0.5*n3),art]+mat3

# define labels and permute (shuffle) the training set again
train_Y=np.zeros((n1,2))
train_Y[:int(0.5*n1),0]=1
train_Y[int(0.5*n1):,1]=1
test_Y=np.zeros((n2,2))
test_Y[:int(0.5*n2),0]=1
test_Y[int(0.5*n2):,1]=1
val_Y=np.zeros((n3,2))
val_Y[:int(0.5*n3),0]=1
val_Y[int(0.5*n3):,1]=1
train_X=train_X[t1]
train_Y=train_Y[t1]

# standardize the data
whole_X=np.row_stack((train_X,test_X,val_X))
whole_X=preprocessing.scale(whole_X)
train_X=whole_X[:n1]
test_X=whole_X[n1:(n1+n2)]
val_X=whole_X[-n3:]