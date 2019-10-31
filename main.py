import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing

from initial import IN
from SurvNet import FN



#----------------------------------------------------------------------------------------------------
# Generate/load the data
# (here we take dataset 1 as a demo, you can substitute it with your own data)
#----------------------------------------------------------------------------------------------------

seed=1 # set a seed

np.random.seed(seed)
whole_X=np.random.uniform(0,1,(10000,28*28))
n=whole_X.shape[0]
p0=whole_X.shape[1] # the number of original variables

random.seed(seed)
art=np.array(random.sample(range(p0),64)) # the indexes of significant variables

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



#----------------------------------------------------------------------------------------------------
# Define the network struture and parameters
#----------------------------------------------------------------------------------------------------

n_classes=2
n_hidden1=40
n_hidden2=20
learning_rate=0.05
epochs=10
batch_size=50
num_batches=train_X.shape[0]/batch_size
dropout=0.8
alpha=0.03 # used for a GL_alpha stopping criterion



#----------------------------------------------------------------------------------------------------
# Define the number of surrogate variables, a prespecified threshold for FDR, an elimination rate, 
# and salience: "abs" or "squ", which means using either absolute values or squares of partial derivatives 
# to measure variable importance
#----------------------------------------------------------------------------------------------------
 
q0=p0
eta=0.1
elimination_rate=1
salience="squ"



#----------------------------------------------------------------------------------------------------
# Print some important parameters
# Note: 'art' is not applicable to real datasets with unknown significant variables
#----------------------------------------------------------------------------------------------------

print('seed:',seed)
print('# of significant variables:',len(art),'FDR cutoff:',eta,'elimination rate:',elimination_rate,'salience:',salience,'\n')



#----------------------------------------------------------------------------------------------------
# Run initial results without variable selection
#----------------------------------------------------------------------------------------------------

initial=IN(seed, 
           train_X,train_Y,val_X,val_Y,test_X,test_Y, 
           n_classes,n_hidden1,n_hidden2, 
           learning_rate,epochs,batch_size,num_batches,dropout,alpha)



#----------------------------------------------------------------------------------------------------
# Run SurvNet
# Note: 'PP' and 'AFDR' are not applicable to real datasets with unknown significant variables
#----------------------------------------------------------------------------------------------------

final,No_org,sval_org,P,Q,PP,train_Loss,train_Acc,val_Loss,val_Acc,EFDR,AFDR=FN(seed, 
                                                                                train_X,train_Y,val_X,val_Y,test_X,test_Y, 
                                                                                n_classes,n_hidden1,n_hidden2, learning_rate,epochs,batch_size,num_batches,dropout,alpha, 
                                                                                p0,q0,art,eta,elimination_rate,salience)



#----------------------------------------------------------------------------------------------------
# Output results
#----------------------------------------------------------------------------------------------------
initnfinal=np.concatenate((initial,final))
np.savetxt('initnfinal.txt',initnfinal)

P,Q,PP,train_Loss,train_Acc,val_Loss,val_Acc,EFDR,AFDR=np.array(P),np.array(Q),np.array(PP),np.array(train_Loss),np.array(train_Acc),np.array(val_Loss),np.array(val_Acc),np.array(EFDR),np.array(AFDR)
step=np.column_stack((P,Q,PP,train_Loss,train_Acc,val_Loss,val_Acc,EFDR,AFDR))
np.savetxt('step.txt',step)

result=np.column_stack((No_org,sval_org))
np.savetxt('result.txt',result)