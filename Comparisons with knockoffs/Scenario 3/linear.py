import tensorflow as tf
import numpy as np
import math
from sklearn import preprocessing
from scipy.linalg import toeplitz

dim=3000
eta=0.2
elimination_rate=1


def data(dim):
    t=[pow(0.5,i) for i in range(dim)]
    Sigmainv=toeplitz(t)
    Sigma=np.linalg.inv(Sigmainv)
    X=np.random.multivariate_normal(np.zeros(dim),Sigma,1000)
    sig=np.random.choice(dim,30,replace=False)
    beta=np.random.choice([-1.5,1.5],len(sig))
    noise=np.random.randn(1000)
    Y=np.dot(X[:,sig],beta)+noise
    return X,Y,sig



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
    cost=tf.square(Y-out3)
    loss=tf.reduce_mean(cost)
    op_train=tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init=tf.global_variables_initializer()
    loss_val_trace=[]
    with tf.Session() as sess:
        sess.run(init)
        i=1
        for j in range(math.ceil(num_batches)):
            sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
        loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
        loss_val_trace.append(loss_val)
        while i<epochs and loss_val/min(loss_val_trace)<1+alpha:
            i+=1
            for j in range(math.ceil(num_batches)):
                sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
            loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
            loss_val_trace.append(loss_val)
        #loss_test=sess.run(loss,feed_dict={X:test_X,Y:test_Y,keep_prob:1.})
        #initnfinal=[loss_test]
        print('epochs:',i)
        print('initial loss:',loss_val,'\n')
        


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
    cost=tf.square(Y-out3)
    loss=tf.reduce_mean(cost)
    op_train=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    grad=tf.gradients(cost,X)

    init=tf.global_variables_initializer()
    loss_val_trace=[]
    with tf.Session() as sess:
        sess.run(init)
        i=1
        for j in range(math.ceil(num_batches)):
            sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
        loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
        loss_val_trace.append(loss_val)
        while i<epochs and loss_val/min(loss_val_trace)<1+alpha:
            i+=1
            for j in range(math.ceil(num_batches)):
                sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
            loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
            loss_val_trace.append(loss_val)
        print('epochs:',i)
        print('validation loss:',loss_val,'\n')
        
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


def main(train_X,val_X,salience):
    tv_X=np.row_stack((train_X,val_X))
    pool_X=tv_X.flatten()
    rand=np.random.choice(len(pool_X),(tv_X.shape[0],q0))
    new_X=pool_X[rand]
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
    p=p0
    q=q0
    
    eFDR=1
    print(p0,q0,len(sig))
    while eFDR>eta:
        train_X,val_X,No,W1,W2,W3,B1,B2,B3,p,q,eFDR=FS(train_X,val_X,No,W1,W2,W3,B1,B2,B3,p,q,salience)
    
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
    cost=tf.square(Y-out3)
    loss=tf.reduce_mean(cost)
    op_train=tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init=tf.global_variables_initializer()
    loss_val_trace=[]
    with tf.Session() as sess:
        sess.run(init)
        i=1
        for j in range(math.ceil(num_batches)):
            sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
        loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
        loss_val_trace.append(loss_val)
        while i<epochs and loss_val/min(loss_val_trace)<1+alpha:
            i+=1
            for j in range(math.ceil(num_batches)):
                sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
            loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
            loss_val_trace.append(loss_val)
        print('epochs:',i)
        print('validation loss:',loss_val,'\n')
'''
        index_org=No<=p0-1
        No_org=No[index_org]

        if len(No_org)==len(No):
            loss_test=sess.run(loss,feed_dict={X:test_X[:,No_org],Y:test_Y,keep_prob:1.})
            final=[loss_test]
            print('final test loss:',loss_test,'\n')
        
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
            cost=tf.square(Y-out3)
            loss=tf.reduce_mean(cost)
            op_train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            init=tf.global_variables_initializer()
            loss_val_trace=[]
            with tf.Session() as sess:
                sess.run(init)
                i=1
                for j in range(math.ceil(num_batches)):
                    sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
                loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
                loss_val_trace.append(loss_val)
                while i<epochs and loss_val/min(loss_val_trace)<1+alpha:
                    i+=1
                    for j in range(math.ceil(num_batches)):
                        sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
                    loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
                    loss_val_trace.append(loss_val)
                loss_test=sess.run(loss,feed_dict={X:test_X[:,No_org],Y:test_Y,keep_prob:1.})
                final=[loss_test]
                print('epochs:',i)
                print('final test loss:',loss_test,'\n')
'''



N_runs=15
result=np.zeros((N_runs,4))
for i in range(N_runs):
    seed=i+11
    print('seed:',seed,'\n')
    
    np.random.seed(seed)
    whole_X,whole_Y,sig=data(dim)

    n=whole_X.shape[0]
    p0=whole_X.shape[1]

    whole_X=preprocessing.scale(whole_X)
    whole_Y=np.reshape(whole_Y,(n,1))

    train_X=whole_X[:int(0.9*n)]
    train_Y=whole_Y[:int(0.9*n)]
    val_X=whole_X[int(0.9*n):]
    val_Y=whole_Y[int(0.9*n):]
    
    n_classes=1
    n_hidden1=40
    n_hidden2=20
    learning_rate=0.002
    epochs=100
    batch_size=30
    num_batches=train_X.shape[0]/batch_size
    dropout=0.8
    alpha=1

    q0=int(p0/4)

    P=[p0]
    PP=[len(sig)]
    EFDR=[1]
    AFDR=[(p0-len(sig))/p0]


    IN()
    main(train_X,val_X,'squ')

    P,PP,EFDR,AFDR=np.array(P),np.array(PP),np.array(EFDR),np.array(AFDR)
    info=np.column_stack((P,PP,EFDR,AFDR))
    result[i]=info[-1]

print(learning_rate,epochs,batch_size,alpha)
np.savetxt('linear_'+str(dim)+'_elim1_pt2.txt',result)