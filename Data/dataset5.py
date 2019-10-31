import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing

from tensorflow.examples.tutorials.mnist import input_data

# load the MNIST dataset
mnist=input_data.read_data_sets("../MNIST_data/",one_hot=True)
train_X_,train_Y_,test_X_,test_Y_,val_X_,val_Y_=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,mnist.validation.images,mnist.validation.labels

# extract the data of digits 4 and 9
def binary(m,n):
    train_index=(train_Y_[:,m]==1)|(train_Y_[:,n]==1)
    train_X=train_X_[train_index]
    train_Y=train_Y_[train_index][:,[m,n]]
    test_index=(test_Y_[:,m]==1)|(test_Y_[:,n]==1)
    test_X=test_X_[test_index]
    test_Y=test_Y_[test_index][:,[m,n]]
    val_index=(val_Y_[:,m]==1)|(val_Y_[:,n]==1)
    val_X=val_X_[val_index]
    val_Y=val_Y_[val_index][:,[m,n]]
    return train_X,train_Y,test_X,test_Y,val_X,val_Y

train_X,train_Y,test_X,test_Y,val_X,val_Y=binary(4,9)
p0=train_X.shape[1] # the number of original variables

# standardize the data
whole_X=np.row_stack((train_X,test_X,val_X))
whole_X=preprocessing.scale(whole_X)
train_X=whole_X[:train_X.shape[0]]
test_X=whole_X[train_X.shape[0]:train_X.shape[0]+test_X.shape[0]]
val_X=whole_X[-val_X.shape[0]:]