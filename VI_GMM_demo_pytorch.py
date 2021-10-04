#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.distributions
import matplotlib as plt
import seaborn as sns
import numpy as np

"""
phi: K x N matrix of cluster assignments
mu:  K x 1 vector of cluster means
s2:  K x 1 vector of cluster variances
x:   N x 1 vector of data points
"""

def update_phi(m, s2, x):
    """
    Variational update for the mixture assignments c_i
    """
    a = torch.ger(x, m)
    b = (s2+m**2)*.5
    return torch.transpose(torch.exp(a-b), 0, 1)/torch.exp(a-b).sum(dim = 1)

def update_m(x, phi, sigma_sq):
    """
    Variational update for the mean of the mixture mean
    distribution mu
    """
    num = torch.matmul(phi, x)
    denom = 1/sigma_sq + phi.sum(dim = 1)
    return num/denom

def update_s2(phi, sigma_sq):
    """
    Variational update for the variance of the mixture mean
    distribution mu
    """
    return (1/sigma_sq + phi.sum(dim = 1))**(-1)

def compute_elbo(phi, m, s2, x, sigma_sq):
    # The ELBO
    t1 = -(2*sigma_sq)**(-1)*(m**2 + s2).sum() + .5*torch.log(s2).sum()
    t2 = -.5 * torch.matmul(phi, x**2).sum() + (phi*torch.ger(m, x)).sum() \
            -.5*(torch.transpose(phi, 0, 1)*(m**2 + s2)).sum()
    t3 = torch.log(phi)
    t3[t3 == float("-Inf")] = 0 # Prevent underflow
    t3 = - (phi*t3).sum()
    return t1 + t2 + t3

def fit(data, k, sigma_sq, num_iter = 2000):
    n = len(data)

    # Randomly initialize the parameters

    # sample m from N(0,1) independently
    m   = torch.distributions.MultivariateNormal(torch.zeros(k), torch.eye(k)).sample()
    # sample s2 from Exponential
    s2  = torch.tensor([torch.distributions.Exponential(5).sample() for _ in range(0,k)])
    
    # set phi to all zeros for now
    phi = torch.zeros((k,n), dtype=torch.float32)
    
    # assign 0 to elbo
    elbo = torch.zeros(num_iter)	
    
    # initialize for each datapoint
    # initializing with Dirichlet ensures phi[:,i] sums up to 1
    for i in range(0, n):
        phi[:,i] = torch.distributions.Dirichlet(torch.from_numpy(np.repeat(1.0,k))).sample().float()

    # update stuff
    for j in range(0, num_iter):
        phi     = update_phi(m, s2, data)
        m       = update_m(data, phi, sigma_sq)
        s2      = update_s2(phi, sigma_sq)
        elbo[j] = compute_elbo(phi, m, s2, data, sigma_sq)
    return (phi, m, s2, elbo)

def VI_GMM_demo():
    # generate data
    datapoints          = torch.zeros(1000)
    datapoints[0:333]   = torch.distributions.Normal(-10, 1).sample((333,))
    datapoints[333:666] = torch.distributions.Normal(.25, 1).sample((333,))
    datapoints[666:]    = torch.distributions.Normal(5, 1).sample((334,))

    # fit 
    out = fit(
                datapoints, 
                3,               # number of components
                10,              # sigma^2 is known
                num_iter = 2000  # number of iterations
                )
                
    # plot results
    sns.distplot(list(datapoints[0:333]), kde=False, bins=50, norm_hist=True)
    sns.distplot(list(datapoints[333:666]), kde=False, bins=50, norm_hist=True)
    sns.distplot(list(datapoints[666:]), kde=False, bins=50, norm_hist=True)
    sns.distplot(list(torch.distributions.Normal(loc=out[1][0], scale=1).sample((1000,))), kde=True, hist=False)
    sns.distplot(list(torch.distributions.Normal(loc=out[1][1], scale=1).sample((1000,))), kde=True, hist=False)
    sns.distplot(list(torch.distributions.Normal(loc=out[1][2], scale=1).sample((1000,))), kde=True, hist=False)
