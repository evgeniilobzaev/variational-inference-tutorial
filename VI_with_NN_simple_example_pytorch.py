#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.distributions import Normal
from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt

### to generate data
def load_dataset(w0,b0,x_range,n=150,seed=0):
    np.random.seed(seed)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g**2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    y = (y - y.mean()) / y.std()
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return y[:, None], x[:, None]

# MLE model
class MLE(nn.Module):
    def __init__(self,hidden_size = 10):
        super(MLE, self).__init__()

        # Standard MLP
        self.layers = nn.Sequential(
                                        nn.Linear(1,hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size,hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size,1)
                                   )
    def forward(self,X):
        out = self.layers(X)
        return out

# sum of squares (for MLE)
def sse(y_true,y_pred):
    return torch.sum((y_true-y_pred)**2)

# VI model
class VI(nn.Module):
    def __init__(self,hidden_size=10):
        super(VI,self).__init__()

        # for reparametrisation
        self.q_mu = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self,X):
        mu      = self.q_mu(X)
        log_var = self.q_log_var(X)

        # now sample
        sigma = torch.exp(0.5 * log_var)
        eps   = torch.randn_like(sigma)

        # and obtain sample from variational posterior
        out   = mu + eps*sigma

        return out,mu,log_var

# for VI loss
def log_gaussian(y,loc,scale):
    dist = Normal(loc,scale) # instantiate object
    logp = dist.log_prob(y)        # compute logprobability of this distribution
    return logp

# main VI loss calculation
def loss(Y,Y_variational,mu,log_var):
    # prior
    log_p = log_gaussian(
                            Y_variational,
                            torch.zeros_like(mu),
                            torch.ones_like(log_var)
                        )

    # variational posterior
    log_q = log_gaussian(
                            Y_variational,
                            mu,
                            torch.exp(0.5 * log_var)
                        )
    # likelihood
    log_D = log_gaussian(
                            Y,
                            mu,
                            torch.exp(0.5 * log_var)
                        )
    
    # compute elbo
    sum_elbo = log_D + log_p - log_q
    elbo     = sum_elbo.mean()
    return -1.0 * elbo

#### to train MLE essentially
def train_MLE(x,y,epochs):
    # create tensors
    X = torch.tensor(x,dtype = torch.float32)
    Y = torch.tensor(y,dtype = torch.float32)   

    # create model
    model     = MLE(hidden_size=15)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0)
    for epoch in range(epochs):
        # get predictions
        Y_pred = model(X)

        
        # get loss
        loss = sse(Y,Y_pred)/Y.size(0)

        # gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(f"Epoch:{epoch}, MSE:{loss.item():0.4f}")
    
    # get predictions
    y_final_pred       = model(X)
    y_final_pred_numpy = y_final_pred.detach().numpy()

    return y_final_pred_numpy

#### to train VI essentially
def train_VI(x,y,epochs,n_samples_prediction=1000):
    # create tensors
    X = torch.tensor(x,dtype = torch.float32)
    Y = torch.tensor(y,dtype = torch.float32)   

    model     = VI(hidden_size=15)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0)
    for epoch in range(epochs):
        # forward pass
        variational_Y,mu,log_var = model(X)

        # loss
        elbo_loss = loss(Y,variational_Y,mu,log_var) 

        # do gradient step
        optimizer.zero_grad()
        elbo_loss.backward()
        optimizer.step()
        # print every 500 epochs
        if epoch % 500 == 0:
            print(f"Epoch:{epoch}, ELBO:{elbo_loss.item():0.4f}")

    
    # get final predictions
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples_prediction):
            Y_pred,_,_ = model(X)
            predictions.append(Y_pred)

    # now need to concat
    Y_samples = torch.cat(predictions,dim=1).numpy()

    # compute quantiles
    q1, mu, q2 = np.quantile(Y_samples, [0.05, 0.5, 0.95], axis=1)

    return q1,mu,q2

# main
def VI_simple_demo_NN():
    # generate data for 500 datapoints
    # some parameters
    w0                = 0.325
    b0                = 3.7
    x_range           = [-30, 60] 
    y,x               = load_dataset(w0,b0,x_range,n=500,seed=0)
    # train MLE for 100 epochs
    y_pred_MLE        = train_MLE(x,y,100)
    # train VI for 5000 epochs and generate 5000 samples per datapoint
    q1_VI,mu_VI,q2_VI = train_VI(x,y,5000,n_samples_prediction=5000)

    # plot all the results
    fig,ax = plt.subplots(figsize=(10,5))
    ax.scatter(x,y,label="data") # data
    ax.plot(x,y_pred_MLE,color="green",label="MLE") # MLE
    ax.plot(x,mu_VI,color="red",label="VI") # VI
    ax.fill_between(x.flatten(), q1_VI, q2_VI, alpha=0.2)
    ax.legend()


