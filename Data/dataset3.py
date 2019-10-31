#import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing

seed=1 # set a seed

lb,ub=0.8,1 # set a lower bound and an upper bound for variation values

np.random.seed(seed)
whole_X=np.random.uniform(0,1,(10000,28*28))
n=whole_X.shape[0]
p0=whole_X.shape[1] # the number of original variables

art=np.random.choice(p0,32,replace=False) # the indexes of significant variables

# make the variables significant by either raising or lowering their values over all samples
# （inflating their variances without changing their means）
sign=np.random.choice([-1,1],(int(0.5*n),len(art)))
u=np.random.uniform(lb,ub,(int(0.5*n),len(art)))
u=u*sign
whole_X[:int(0.5*n),art]=whole_X[:int(0.5*n),art]+u

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