#import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing

seed=44 # set a seed
np.random.seed(seed)

# load and standardize the data
whole_X=np.loadtxt('chen_X.txt')
whole_Y=np.loadtxt('chen_Y.txt')
whole_X=whole_X.T
whole_X=preprocessing.scale(whole_X)

# permute (shuffle) the data
n=whole_X.shape[0]
t=np.random.permutation(n)
whole_X=whole_X[t]
whole_Y=whole_Y[t]
p0=train_X.shape[1] # the number of original variables

# data splitting
train_X=whole_X[:int(0.8*0.7*n)]
train_Y=whole_Y[:int(0.8*0.7*n)]
val_X=whole_X[int(0.8*0.7*n):int(0.8*n)]
val_Y=whole_Y[int(0.8*0.7*n):int(0.8*n)]
test_X=whole_X[int(0.8*n):]
test_Y=whole_Y[int(0.8*n):]