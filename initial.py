### the network performance without variable selection (for comparison with the performance using SurvNet)

import tensorflow as tf
import numpy as np
import math

def IN(seed, 
       train_X,train_Y,val_X,val_Y,test_X,test_Y, 
       n_classes,n_hidden1,n_hidden2, 
       learning_rate,epochs,batch_size,num_batches,dropout,alpha):
    
    # get the number of variables
    n_dim=train_X.shape[1]
    
    # create placeholders, initialize weights and biases, and define the loss function
    X=tf.placeholder(tf.float32,[None,n_dim])
    Y=tf.placeholder(tf.float32,[None,n_classes])
    keep_prob=tf.placeholder(tf.float32)
    
    W1=0.01*np.random.randn(n_dim,n_hidden1).astype(np.float32)
    B1=np.random.randn(n_hidden1).astype(np.float32)
    W2=0.01*np.random.randn(n_hidden1,n_hidden2).astype(np.float32)
    B2=np.random.randn(n_hidden2).astype(np.float32)
    W3=0.01*np.random.randn(n_hidden2,n_classes).astype(np.float32)
    B3=np.random.randn(n_classes).astype(np.float32)
    
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
            
        # get the initial loss and accuracy on the test set (with all original variables)
        loss_test=sess.run(loss,feed_dict={X:test_X,Y:test_Y,keep_prob:1.})
        pred_test=np.argmax(sess.run(out3,feed_dict={X:test_X,Y:test_Y,keep_prob:1.}),axis=1)
        accuracy_test=np.mean(pred_test==np.argmax(test_Y,axis=1))
        
        initial=[loss_test,accuracy_test]
        print('epochs:',i)
        print('initial test loss:',loss_test,'initial test accuracy:',accuracy_test,'\n')

    # return the initial loss and accuracy on the test set
    return initial