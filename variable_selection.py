### variable selection in one step

import tensorflow as tf
import numpy as np
import math

# enter the current training set, the current validation set, the column indexes of current remaining variables, the weights and biases of the last network (for a warm start), and the current number of original variables and surrogate variables
def VS(seed, 
       train_X,train_Y,val_X,val_Y,test_X,test_Y, 
       n_classes,n_hidden1,n_hidden2, 
       learning_rate,epochs,batch_size,num_batches,dropout,alpha, 
       p0,q0,art,eta,elimination_rate,salience, 
       No,W1,W2,W3,B1,B2,B3,p,q, 
       P,Q,PP,train_Loss,train_Acc,val_Loss,val_Acc,EFDR,AFDR):
    
    ### the network performance using current remaining variables (before variable elimination in this step)
    
    # get the number of current remaining variables
    n_dim=train_X.shape[1]
    
    X=tf.placeholder(tf.float32,[None,n_dim])
    Y=tf.placeholder(tf.float32,[None,n_classes])
    keep_prob=tf.placeholder(tf.float32)
    
    # initialize weights and biases using the values of weights and biases in the last trained network
    w1=tf.Variable(W1,name="weights1")
    b1=tf.Variable(B1,name="biases1")
    w2=tf.Variable(W2,name="weights2")
    b2=tf.Variable(B2,name="biases2")
    w3=tf.Variable(W3,name="weights3")
    b3=tf.Variable(B3,name="biases3")
    
    out1=tf.nn.relu(tf.matmul(X,w1)+b1)
    out1=tf.nn.dropout(out1,keep_prob,seed=seed)
    out2=tf.nn.relu(tf.matmul(out1,w2)+b2)
    out2=tf.nn.dropout(out2,keep_prob,seed=seed)
    out3=tf.matmul(out2,w3)+b3
    
    cost=tf.nn.softmax_cross_entropy_with_logits(logits=out3,labels=Y)
    loss=tf.reduce_mean(cost)
    op_train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    grad=tf.gradients(cost,X) # define the partial derivatives

    init=tf.global_variables_initializer()
    loss_val_trace=[]
    accuracy_val_trace=[]
    
    # training using a GL_alpha stopping criterion
    with tf.Session() as sess:
        sess.run(init)
        i=1
        for j in range(math.ceil(num_batches)):
            sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
        loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
        pred_val=np.argmax(sess.run(out3,feed_dict={X:val_X,Y:val_Y,keep_prob:1.}),axis=1)
        accuracy_val=np.mean(pred_val==np.argmax(val_Y,axis=1))
        loss_val_trace.append(loss_val)
        accuracy_val_trace.append(accuracy_val)
        
        while i<epochs and loss_val/min(loss_val_trace)<1+alpha:
            i+=1
            for j in range(math.ceil(num_batches)):
                sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
            loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
            pred_val=np.argmax(sess.run(out3,feed_dict={X:val_X,Y:val_Y,keep_prob:1.}),axis=1)
            accuracy_val=np.mean(pred_val==np.argmax(val_Y,axis=1))
            loss_val_trace.append(loss_val)
            accuracy_val_trace.append(accuracy_val)
        
        # save the loss and accuracy on the training set
        loss_train=sess.run(loss,feed_dict={X:train_X,Y:train_Y,keep_prob:1.})
        pred_train=np.argmax(sess.run(out3,feed_dict={X:train_X,Y:train_Y,keep_prob:1.}),axis=1)
        accuracy_train=np.mean(pred_train==np.argmax(train_Y,axis=1))
        train_Loss.append(loss_train)
        train_Acc.append(accuracy_train)
        
        # save the loss and accuracy on the validation set
        val_Loss.append(loss_val)
        val_Acc.append(accuracy_val)
        print('epochs:',i)
        print('validation accuracy:',accuracy_val,'\n')
        
        
        ### variable elimination in this step
        
        # get the partial derivatives of the loss at each sample with respect to each variable
        vals=sess.run(grad,feed_dict={X:train_X,Y:train_Y,keep_prob:1.})[0]
        
        # compute the mean of absolute values or squares of the partial derivatives as the variable importance
        if salience=="abs":
            temp=np.abs(vals)
            s=np.mean(temp,axis=0)
        if salience=="squ":
            temp=vals**2
            s=np.mean(temp,axis=0)
        
        # the number of variables to be left after variable elimination in this step
        num=n_dim-math.ceil(elimination_rate*(q-eta*p*q0/p0))
        
        # get the indexes of the variables to be kept by sorting the importance of variables
        index=np.argsort(s)[-num:]
        
        # eliminate the other variables, update the training set, the validation set, and the column indexes of remaining variables, and record the importance of remaining variables
        train_X=train_X[:,index]
        val_X=val_X[:,index]
        No=No[index]
        s=s[index]
        
        # count how many original variables, surrogate variables, and significant variables (only for simulated datasets) there are
        q=sum(No>p0-1)
        p=num-q
        count=[]
        for i in range(len(art)):
            count.append(art[i] in No)
        
        # calculate the estimated FDR and the actual FDR (only for simulated datasets)
        eFDR=q/p*p0/q0
        aFDR=(p-sum(count))/p
        
        # save the statistics above
        P.append(p)
        Q.append(q)
        PP.append(sum(count))
        EFDR.append(eFDR)
        AFDR.append(aFDR)
        print('# of original variables:',p,'# of surrogate variables:',q,'# of significant variables:',sum(count))
        
        # save the values of weights and biases
        weights1=sess.run(w1)
        W1=weights1[index,:]
        W2=sess.run(w2)
        W3=sess.run(w3)
        B1=sess.run(b1)
        B2=sess.run(b2)
        B3=sess.run(b3)
    
    # return the updated training set, the updated validation set, the column indexes of remaining variables, the weights and biases of the current network, the numbers of remaining original variables and surrogate variables, the estimated FDR, and the importance of remaining variables
    return train_X,val_X,No,W1,W2,W3,B1,B2,B3,p,q,eFDR,s