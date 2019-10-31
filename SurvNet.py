### the complete proposed variable selection procedure (SurvNet)

import tensorflow as tf
import numpy as np
import math

from variable_selection import VS

def FN(seed, 
       train_X,train_Y,val_X,val_Y,test_X,test_Y, 
       n_classes,n_hidden1,n_hidden2, 
       learning_rate,epochs,batch_size,num_batches,dropout,alpha, 
       p0,q0,art,eta,elimination_rate,salience):
    
    # define lists to save statistics at each step of variable selection (p0 and q0 are defined in main.py, art is only defined for simulated datasets)
    P=[p0]
    Q=[q0]
    PP=[len(art)]
    train_Loss=[]
    train_Acc=[]
    val_Loss=[]
    val_Acc=[]
    EFDR=[1]
    AFDR=[(p0-len(art))/p0]
    
    # generate q0 surrogate variables for the training set and the validation set
    tv_X=np.row_stack((train_X,val_X))
    pool_X=tv_X.flatten()
    rand=np.random.choice(len(pool_X),(tv_X.shape[0],q0))
    new_X=pool_X[rand]
    train_X=np.column_stack((train_X,new_X[:train_X.shape[0]]))
    val_X=np.column_stack((val_X,new_X[-val_X.shape[0]:]))
    
    # initialize weights and biases for the network with all original variables and surrogate variables 
    n_dim=train_X.shape[1]
    W1=0.01*np.random.randn(n_dim,n_hidden1).astype(np.float32)
    B1=np.random.randn(n_hidden1).astype(np.float32)
    W2=0.01*np.random.randn(n_hidden1,n_hidden2).astype(np.float32)
    B2=np.random.randn(n_hidden2).astype(np.float32)
    W3=0.01*np.random.randn(n_hidden2,n_classes).astype(np.float32)
    B3=np.random.randn(n_classes).astype(np.float32)
    
    # the column indexes of all variables, the initial values of the number of original variables, the number of surrogate variables, and the estimated FDR
    No=np.array(range(n_dim))
    p=p0
    q=q0
    eFDR=1
    print('# of original variables:',p0,'# of surrogate variables:',q0,'# of significant variables:',len(art))
    
    # select variables step by step until the estimated FDR is no greater than eta (the prespecified threshold) 
    while eFDR>eta:
        train_X,val_X,No,W1,W2,W3,B1,B2,B3,p,q,eFDR,sval=VS(seed, 
                                                            train_X,train_Y,val_X,val_Y,test_X,test_Y, 
                                                            n_classes,n_hidden1,n_hidden2, learning_rate,epochs,batch_size,num_batches,dropout,alpha, 
                                                            p0,q0,art,eta,elimination_rate,salience, 
                                                            No,W1,W2,W3,B1,B2,B3,p,q, 
                                                            P,Q,PP,train_Loss,train_Acc,val_Loss,val_Acc,EFDR,AFDR)
    
    # define the network with selected variables
    n_dim=train_X.shape[1]
    
    X=tf.placeholder(tf.float32,[None,n_dim])
    Y=tf.placeholder(tf.float32,[None,n_classes])
    keep_prob=tf.placeholder(tf.float32)
    
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
            
        loss_train=sess.run(loss,feed_dict={X:train_X,Y:train_Y,keep_prob:1.})
        pred_train=np.argmax(sess.run(out3,feed_dict={X:train_X,Y:train_Y,keep_prob:1.}),axis=1)
        accuracy_train=np.mean(pred_train==np.argmax(train_Y,axis=1))
        train_Loss.append(loss_train)
        train_Acc.append(accuracy_train)
        
        val_Loss.append(loss_val)
        val_Acc.append(accuracy_val)
        print('epochs:',i)
        print('validation accuracy:',accuracy_val,'\n')
        
        # get the indexes of remaining original variables
        index_org=No<=p0-1
        No_org=No[index_org]
        sval_org=sval[index_org]
        
        ## check if there are any surrogate variables left:
        # if no, get the final loss and accuracy on the test set using only the selected variables;
        if len(No_org)==len(No):
            loss_test=sess.run(loss,feed_dict={X:test_X[:,No_org],Y:test_Y,keep_prob:1.})
            pred_test=np.argmax(sess.run(out3,feed_dict={X:test_X[:,No_org],Y:test_Y,keep_prob:1.}),axis=1)
            accuracy_test=np.mean(pred_test==np.argmax(test_Y,axis=1))
            final=[loss_test,accuracy_test]
            print('final test loss:',loss_test,'final test accuracy:',accuracy_test)
        
        # if yes, remove the remaining surrogate variables and train the network containing only the selected variables that are original with a warm start using a GL_alpha stopping criterion.
        else:
            train_X=train_X[:,index_org]
            val_X=val_X[:,index_org]
            
            weights1=sess.run(w1)
            W1=weights1[index_org,:]
            W2=sess.run(w2)
            W3=sess.run(w3)
            B1=sess.run(b1)
            B2=sess.run(b2)
            B3=sess.run(b3)

            n_dim=train_X.shape[1]
            
            X=tf.placeholder(tf.float32,[None,n_dim])
            Y=tf.placeholder(tf.float32,[None,n_classes])
            keep_prob=tf.placeholder(tf.float32)
            
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
                
                # get the final loss and accuracy on the test set
                loss_test=sess.run(loss,feed_dict={X:test_X[:,No_org],Y:test_Y,keep_prob:1.})
                pred_test=np.argmax(sess.run(out3,feed_dict={X:test_X[:,No_org],Y:test_Y,keep_prob:1.}),axis=1)
                accuracy_test=np.mean(pred_test==np.argmax(test_Y,axis=1))
                final=[loss_test,accuracy_test]
                print('epochs:',i)
                print('final test loss:',loss_test,'final test accuracy:',accuracy_test)
                
    # return the final loss and accuracy on the test set, the column indexes of the selected (original) variables and their importance, and the lists that save the statistics
    return final,No_org,sval_org,P,Q,PP,train_Loss,train_Acc,val_Loss,val_Acc,EFDR,AFDR