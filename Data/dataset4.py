#import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing

seed=1 # set a seed

np.random.seed(seed)
whole_X=np.random.uniform(-1,1,(10000,784))
n=whole_X.shape[0]
p0=whole_X.shape[1] # the number of original variables

random.seed(seed)
art=np.array(random.sample(range(p0),64)) # the indexes of significant variables

np.random.seed(seed)

# define regression coefficients and error terms
sign=np.random.choice([-1,1],len(art))
beta=np.random.uniform(1,3,len(art))
beta=beta*sign
epsilon=np.random.randn(n)

wholenew_X=np.zeros((n,len(art)))
wholenew_X[:,:16]=whole_X[:,art[:16]]
wholenew_X[:,16:32]=np.sin(whole_X[:,art[16:32]])
wholenew_X[:,32:48]=np.exp(whole_X[:,art[32:48]])
wholenew_X[:,48:]=np.maximum(whole_X[:,art[48:]],np.zeros((n,16)))

# define regression coefficients for 4 interaction terms
sign2=np.random.choice([-1,1],4)
beta2=np.random.uniform(1,3,4)
beta2=beta2*sign2

wholenew_X2=np.zeros((n,4))
wholenew_X2[:,0]=whole_X[:,art[14]]*whole_X[:,art[15]]
wholenew_X2[:,1]=whole_X[:,art[30]]*whole_X[:,art[31]]
wholenew_X2[:,2]=whole_X[:,art[46]]*whole_X[:,art[47]]
wholenew_X2[:,3]=whole_X[:,art[62]]*whole_X[:,art[63]]

# define responses
whole_Y=np.dot(wholenew_X,beta)+np.dot(wholenew_X2,beta2)+epsilon
whole_Y=np.reshape(whole_Y,(n,1))

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