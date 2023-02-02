import numpy as np

def sigmoid(z):
    
    g=1/(1+np.exp(-z))

    return g

def compute_cost(X,y,w,b,lambda_=1):
    m,n=X.shape
    cost=0.
    loss=0.
    for i in range(m):
        z=np.dot(w,X[i])+b
        f_wb=sigmoid(z)
        loss=-y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
        cost+=loss

    cost/=m

    return cost    

def compute_gradient(X,y,w,b,lambda_=None):
    m,n=X.shape

    dj_dw=np.zeros(n)
    dj_db=0.

    for i in range(m):
        z=np.dot(X[i],w)+b
        f_wb=sigmoid(z)
        err=f_wb-y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i][j]
        dj_db+=err
        
    dj_dw/=m
    dj_db/=m

    return dj_db,dj_dw

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters,lambda_=None):

    m,n=X.shape
    
    w=w_in
    b=b_in

    dj_dw=np.zeros(n)
    dj_db=0.

    one_tenth_num_iters=num_iters/10
    cost_history=np.zeros(10)
    counter=0

    for i in range(num_iters):
        
        
        if(i%one_tenth_num_iters==0):
            cost_history[counter]=cost_function(X,y,w,b)
            print(cost_history[counter])
            counter=counter+1
        dj_db,dj_dw=gradient_function(X,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db

    return w,b,cost_history


def prediction(w,b,X):
    m,n=X.shape
    y=np.zeros(m)
    for i in range(m):
        z=np.dot(X[i],w)+b
        value=sigmoid(z)
        if(value>0.5):
            y[i]=1
        else:
            y[i]=0
    return y