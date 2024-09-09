
import argparse
import logging
import numpy as np
import math
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)

def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(arg1,arg2):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    shape=(arg1,arg2)
    np.random.seed(np.prod(shape))
    # Implement random initialization here
    retar=np.random.uniform(-0.1,0.1,shape)
    retar[:,0]=np.zeros((arg1))
    return retar

class nn:
    def __init__ (self, X_tr, y_tr, X_te, y_te, n_hid,alpha,beta, n_epochs, lr,out_metrics):
        self.X_tr = X_tr
        self.y_tr =y_tr
        self.X_te= X_te
        self.y_te =  y_te
        self.n_hid=n_hid
        self.alpha=alpha
        self.beta=beta
        self.n_epochs = n_epochs
        self.lr = lr
        self.out_metrics=out_metrics
        self.trainer = 0.0
        self.tester = 0.0
        self.cetrainavg = np.zeros((n_epochs,1))
        self.cetestavg = np.zeros((n_epochs,1))
        self.sal=np.zeros((n_hid,self.X_tr.shape[1]))
        self.sbeta=np.zeros((10,n_hid+1))

def forward(nn, X):
   a = np.dot(nn.alpha,X) #check
   z = 1/(1+np.exp(-a))
   z = np.append(1,z)
   b = np.dot(nn.beta,z)
   yh =np.divide(np.exp(b),np.sum(np.exp(b)))
   return X,a,z,b,yh

def backward(nn, x, y, resultx, resulta,resultz,resultb,resulty):
   x,a,z,b,yh =  resultx,resulta,resultz,resultb,resulty
   gyh =-np.divide(y,yh)
   gyh=gyh.reshape(-1,1)
   gyht=gyh.T
   y_h=yh.reshape(-1,1)
   i=np.dot(y_h,y_h.T)
   subd=np.subtract(np.diag(yh),i)
   gb=np.dot(gyht,subd)
   z_p=z.reshape(-1,1)
   gbeta=np.dot(gb.T,z_p.T)
   gz=np.dot(nn.beta.T,gb.T)
   ga=np.multiply(gz.flatten(), np.multiply(z,1-z))[1:]
   x=x.reshape(-1,1)
   ga = np.reshape(ga, (1,ga.shape[0]))
   galpha=np.dot(ga.T,x.T)
   gx=np.dot(nn.alpha.T,ga.T)
   return galpha, gbeta

def train(nn):
        for i in range(0, nn.n_epochs):
         X_tr,y_tr=shuffle(nn.X_tr,nn.y_tr, i)
         for j in range(0, nn.X_tr.shape[0]):
               x,ytemp = X_tr[j,:], y_tr[j,]
               y=np.eye(10)[ytemp]
               xr,ar,zr,br,yhr = forward(nn,x)
               gralpha, grbeta = backward(nn,x,y,xr,ar,zr,br,yhr)
               nn.sal+=(gralpha)**2
               nn.sbeta+=(grbeta)**2
               update_a=(nn.lr)*(gralpha)*(1/np.sqrt(1e-5+nn.sal))
               update_b=(nn.lr)*(grbeta)*(1/np.sqrt(1e-5+nn.sbeta))
               nn.beta = nn.beta - update_b
               nn.alpha= nn.alpha - update_a
         trainces=list()
         for k in range(0, nn.X_tr.shape[0]):
               x,ytemp = nn.X_tr[k,:], nn.y_tr[k,]
               y=np.eye(10)[ytemp]
               x,a,z,b,yh= forward(nn,x)
               crossentrop=-np.dot(y.T,np.log(yh))
               trainces.append(crossentrop)
         testces=list()
         for m in range(0, nn.X_te.shape[0]):
               x,ytemp = nn.X_te[m,:], nn.y_te[m,]
               y=np.eye(10)[ytemp]
               x,a,z,b,yh,= forward(nn,x)
               crossentrop=-np.dot(y.T,np.log(yh))
               testces.append(crossentrop)
         nn.cetrainavg[i] = sum(trainces)/len(trainces)
         nn.cetestavg[i] = sum(testces)/len(testces)


def predict_train(nn):
      preds=[]
      length= nn.X_tr.shape[0]
      for i in range(0, length):
            x,ytemp = nn.X_tr[i,:], nn.y_tr[i]
            y=np.eye(10)[ytemp]
            x,a,z,b,yh = forward(nn,x)
            preds.append(np.argmax(yh))
            if np.argmax(y)!=np.argmax(yh):
               nn.trainer+=1
      nn.trainer=nn.trainer/length
      return preds
def predict_test(nn):
      preds=[]
      length= nn.X_te.shape[0]
      for i in range(0, length):
            x,ytemp = nn.X_te[i,:], nn.y_te[i]
            y=np.eye(10)[ytemp]
            x,a,z,b,yh = forward(nn,x)
            preds.append(np.argmax(yh))
            if np.argmax(y)!=np.argmax(yh):
               nn.tester+=1
      nn.tester=nn.tester/length
      return preds


def write(nn):
    st=''
    for w in range(0, nn.n_epochs):
        st = st+ "epoch=" + str(w+1) +" crossentropy(train): " + str(nn.cetrainavg[w][0]) + "\n"
        st = st + "epoch=" + str(w+1) + " crossentropy(validation): " + str(nn.cetestavg[w][0]) + "\n"
    st = st + "error(train): " + str(nn.trainer) + "\n"
    st = st + "error(validation): " + str(nn.tester)
    metrix=open(nn.out_metrics,'w')
    print(st)
    metrix.write(st)
