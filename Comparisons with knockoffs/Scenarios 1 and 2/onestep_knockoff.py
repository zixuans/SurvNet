import tensorflow as tf
import numpy as np
import math
from sklearn import preprocessing

eta=0.1
#elimination_rate=1
#print('squ',64,eta,elimination_rate,'\n')



def IN():
    n_dim=train_X.shape[1]
    W1=0.01*np.random.randn(n_dim,n_hidden1).astype(np.float32)
    B1=np.random.randn(n_hidden1).astype(np.float32)
    W2=0.01*np.random.randn(n_hidden1,n_hidden2).astype(np.float32)
    B2=np.random.randn(n_hidden2).astype(np.float32)
    W3=0.01*np.random.randn(n_hidden2,n_classes).astype(np.float32)
    B3=np.random.randn(n_classes).astype(np.float32)
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
        loss_test=sess.run(loss,feed_dict={X:test_X,Y:test_Y,keep_prob:1.})
        pred_test=np.argmax(sess.run(out3,feed_dict={X:test_X,Y:test_Y,keep_prob:1.}),axis=1)
        accuracy_test=np.mean(pred_test==np.argmax(test_Y,axis=1))
        initnfinal=[loss_test,accuracy_test]
        print('epochs:',i)
        print('initial test loss:',loss_test,'initial test accuracy:',accuracy_test,'\n')

    return initnfinal

'''
def FS(train_X,val_X,No,W1,W2,W3,B1,B2,B3,p,q,salience):
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
    grad=tf.gradients(cost,X)

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
        print('epochs:',i)
        print('validation accuracy:',accuracy_val,'\n')
        
        vals=sess.run(grad,feed_dict={X:train_X,Y:train_Y,keep_prob:1.})[0]
        #std=np.std(train_X,axis=0)
        #var=std**2
        if salience=="abs":
            temp=np.abs(vals)
            s=np.sum(temp,axis=0)
            #if standardized==False:
                #s=s*std
        if salience=="squ":
            temp=vals**2
            s=np.sum(temp,axis=0)
            #if standardized==False:
                #s=s*var
        num=n_dim-math.ceil(elimination_rate*(q-eta*p*q0/p0))
        index=np.argsort(s)[-num:]
        train_X=train_X[:,index]
        val_X=val_X[:,index]
        No=No[index]
        q=sum(No>p0-1)
        p=num-q
        count=[]
        for i in range(len(sig)):
            count.append(sig[i] in No)
        eFDR=q/p*p0/q0
        aFDR=(p-sum(count))/p
        P.append(p)
        PP.append(sum(count))
        EFDR.append(eFDR)
        AFDR.append(aFDR)
        print(p,q,sum(count))
        
        weights1=sess.run(w1)
        W1=weights1[index,:]
        W2=sess.run(w2)
        W3=sess.run(w3)
        B1=sess.run(b1)
        B2=sess.run(b2)
        B3=sess.run(b3)
    
    return train_X,val_X,No,W1,W2,W3,B1,B2,B3,p,q,eFDR
'''

def main(train_X,val_X,salience):
    new_X=whole_Xk
    train_X=np.column_stack((train_X,new_X[:train_X.shape[0]]))
    val_X=np.column_stack((val_X,new_X[-val_X.shape[0]:]))
    
    n_dim=train_X.shape[1]
    No=np.array(range(n_dim))
    W1=0.01*np.random.randn(n_dim,n_hidden1).astype(np.float32)
    B1=np.random.randn(n_hidden1).astype(np.float32)
    W2=0.01*np.random.randn(n_hidden1,n_hidden2).astype(np.float32)
    B2=np.random.randn(n_hidden2).astype(np.float32)
    W3=0.01*np.random.randn(n_hidden2,n_classes).astype(np.float32)
    B3=np.random.randn(n_classes).astype(np.float32)
    #p=p0
    #q=q0
    
    #eFDR=1
    print(p0,len(sig))
    
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
    grad=tf.gradients(cost,X)

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
        
        vals=sess.run(grad,feed_dict={X:train_X,Y:train_Y,keep_prob:1.})[0]
        #std=np.std(train_X,axis=0)
        #var=std**2
        if salience=="abs":
            temp=np.abs(vals)
            s=np.sum(temp,axis=0)
            #if standardized==False:
                #s=s*std
        if salience=="squ":
            temp=vals**2
            s=np.sum(temp,axis=0)
            #if standardized==False:
                #s=s*var
        
        snew=s[:p0]-s[p0:]
        posval=snew[snew>0]
        sort=np.argsort(posval)
        for i in range(len(posval)):
            t=posval[sort[i]]
            num1=sum(snew>=t)
            num2=sum(snew<=-t)
            if num2/num1<=eta:
                break
        
        index=snew>=t
        p=sum(index)
        count=snew[sig]>=t
        eFDR=sum(snew<=-t)/p
        aFDR=(p-sum(count))/p
        P.append(p)
        PP.append(sum(count))
        EFDR.append(eFDR)
        AFDR.append(aFDR)
        print(p,sum(count))
        
        train_X=train_X[:,:p0]
        train_X=train_X[:,index]
        val_X=val_X[:,:p0]
        val_X=val_X[:,index]
        weights1=sess.run(w1)
        W1=weights1[:p0,:]
        W1=W1[index,:]
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
        loss_test=sess.run(loss,feed_dict={X:test_X[:,index],Y:test_Y,keep_prob:1.})
        pred_test=np.argmax(sess.run(out3,feed_dict={X:test_X[:,index],Y:test_Y,keep_prob:1.}),axis=1)
        accuracy_test=np.mean(pred_test==np.argmax(test_Y,axis=1))
        final=[loss_test,accuracy_test]
        print('final test loss:',loss_test,'final test accuracy:',accuracy_test,'\n')
        
    return final
                


N_runs=25
result1=np.zeros((N_runs,4))
result2=np.zeros((N_runs,4))
for i in range(N_runs):
    seed=i+1
    print('seed:',seed,'\n')
    
    whole_X=np.loadtxt('data/X_32_'+str(seed)+'.txt')
    whole_Y=np.loadtxt('data/Y_32_'+str(seed)+'.txt')
    sig=np.loadtxt('data/sig_32_'+str(seed)+'.txt')
    whole_Xk=np.loadtxt('data/Xk_32_'+str(seed)+'.txt')
    sig=sig.astype(int)
    np.random.seed(seed)

    n=whole_X.shape[0]
    p0=whole_X.shape[1]

    whole_X=preprocessing.scale(whole_X)

    train_X=whole_X[:int(0.8*0.7*n)]
    train_Y=whole_Y[:int(0.8*0.7*n)]
    val_X=whole_X[int(0.8*0.7*n):int(0.8*n)]
    val_Y=whole_Y[int(0.8*0.7*n):int(0.8*n)]
    test_X=whole_X[int(0.8*n):]
    test_Y=whole_Y[int(0.8*n):]
    
    n_classes=2
    n_hidden1=40
    n_hidden2=20
    learning_rate=0.05
    epochs=10
    batch_size=50
    num_batches=train_X.shape[0]/batch_size
    dropout=0.8
    alpha=0.03

    #q0=p0

    P=[p0]
    PP=[len(sig)]
    EFDR=[1]
    AFDR=[(p0-len(sig))/p0]


    initnfinal=IN()
    final=main(train_X,val_X,'squ')

    initnfinal=np.append(initnfinal,final)
    result1[i]=initnfinal

    P,PP,EFDR,AFDR=np.array(P),np.array(PP),np.array(EFDR),np.array(AFDR)
    info=np.column_stack((P,PP,EFDR,AFDR))
    result2[i]=info[-1]

    

np.savetxt('result1_32_oneknockoff.txt',result1)
np.savetxt('result2_32_oneknockoff.txt',result2)