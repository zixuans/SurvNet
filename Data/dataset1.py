#import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing

seed=1 # set a seed

np.random.seed(seed)
whole_X=np.random.uniform(0,1,(10000,28*28))
n=whole_X.shape[0]
p0=whole_X.shape[1] # the number of original variables

random.seed(seed)
art=np.array(random.sample(range(p0),32)) # the indexes of significant variables

# make the variables significant by shifting their values at half of the samples
np.random.seed(seed)
sign=np.random.choice([-1,1],len(art))
u=np.random.uniform(0.1,0.3,len(art))
u=u*sign
mat=np.reshape(np.tile(u,int(0.5*n)),(int(0.5*n),len(art)))
whole_X[:int(0.5*n),art]=whole_X[:int(0.5*n),art]+mat

# define labels
whole_Y=np.zeros((n,2))
whole_Y[:int(0.5*n),0]=1
whole_Y[int(0.5*n):,1]=1

# permute (shuffle) and standardize the data
t=np.random.RandomState(seed).permutation(n)
whole_X=whole_X[t]
whole_Y=whole_Y[t]
whole_X=preprocessing.scale(whole_X)

# data splitting
train_X=whole_X[:int(0.8*0.7*n)]
train_Y=whole_Y[:int(0.8*0.7*n)]
val_X=whole_X[int(0.8*0.7*n):int(0.8*n)]
val_Y=whole_Y[int(0.8*0.7*n):int(0.8*n)]
test_X=whole_X[int(0.8*n):]
test_Y=whole_Y[int(0.8*n):]