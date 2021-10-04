#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from scipy.special import expit as sigmoid
from scipy.stats import multivariate_normal

import copy

# to load data
def load_data(file_):
    data = Image.open(file_)
    img = np.double(data)
    img_mean = np.mean(img)
    img_binary = +1*(img>img_mean) + -1*(img<img_mean)
    return img_binary

# compute log-likelihood of data under variational distribution
def log_like(mu,J):
    # obtain shape of mu
    M,N      = mu.shape
    # accumulate loglike
    log_like = 0.0
    for ix in range(N):
        for iy in range(M):
            pos = iy + M*ix
            neighborhood = pos + np.array([-1,1,-M,M])         
            boundary_idx = [iy!=0,iy!=M-1,ix!=0,ix!=N-1]
            pos = iy + M*ix
            neighborhood = pos + np.array([-1,1,-M,M])         
            boundary_idx = [iy!=0,iy!=M-1,ix!=0,ix!=N-1]
            neighborhood = neighborhood[np.where(boundary_idx)[0]]  
            xx, yy = np.unravel_index(pos, (M,N), order='F')          # index of current pixel
            nx, ny = np.unravel_index(neighborhood, (M,N), order='F') # indices of pixel's neighbors
            # compute loglike
            log_like += mu[xx,yy] * J * np.sum(mu[nx,ny])
    
    return log_like

# fixed point algorithm to compute mu(t)
def update_mu(mu_prev,J,logodds):
    # get shape of mu_prev -> should be M-x-N
    M,N = mu_prev.shape

    # loop
    mu_next = np.zeros(shape=(M,N)) # mu_next is mu(t) and mu_prev is mu(t-1)
    
    for ix in range(N):
        for iy in range(M):
            pos = iy + M*ix
            neighborhood = pos + np.array([-1,1,-M,M])         
            boundary_idx = [iy!=0,iy!=M-1,ix!=0,ix!=N-1]
            neighborhood = neighborhood[np.where(boundary_idx)[0]]  
            
            # index of current pixel in 2D coordinate system
            xx, yy       = np.unravel_index(pos, (M,N), order='F')

            # index of current pixel neighbors in 2D coordinatr system
            nx, ny       = np.unravel_index(neighborhood, (M,N), order='F')
            
            # find sum of W(ij)*mu(j)^(t-1)
            mu_pos       = J*np.sum(mu_prev[nx,ny])

            # compute mu(i)^(t)       
            mu_next[xx,yy] = np.tanh(mu_pos + 0.5*logodds[xx,yy])
    
    return mu_next

# compute a, using updated mu
def compute_a(mu,J,logodds):
    # get shape
    M,N = mu.shape

    a = np.zeros(shape=(M,N)) # empty array for a
    
    # loop
    for ix in range(N):
        for iy in range(M):
            pos          = iy + M*ix
            neighborhood = pos + np.array([-1,1,-M,M])         
            boundary_idx = [iy!=0,iy!=M-1,ix!=0,ix!=N-1]
            neighborhood = neighborhood[np.where(boundary_idx)[0]]  
            
            # index of current pixel in 2D coordinate system
            xx, yy       = np.unravel_index(pos, (M,N), order='F')

            # index of current pixel neighbors in 2D coordinatr system
            nx, ny       = np.unravel_index(neighborhood, (M,N), order='F')
            
            # find sum of W(ij)*mu(j)^(t-1)
            mu_pos       = J*np.sum(mu[nx,ny])

            # compute a
            a[xx,yy]     = mu_pos + 0.5*logodds[xx,yy]
    
    return a

# to do VI
def doVI(y, J = 1.0, sigma = 1.0, n_iter = 10):
    # set up some variables to store results at each iteration
    ELBO     = np.zeros(n_iter) 

    # get shape of y -> our noisy image
    M,N = y.shape

    # log-odds: L(plus) - L(minus) in textbook
    logodds = multivariate_normal.logpdf(y.flatten(), mean=+1, cov=sigma**2) - multivariate_normal.logpdf(y.flatten(), mean=-1, cov=sigma**2)
    
    # log-propbs separately
    logp1 = np.reshape(multivariate_normal.logpdf(y.flatten(), mean=+1, cov=sigma**2), (M, N)) # L(plus)
    logm1 = np.reshape(multivariate_normal.logpdf(y.flatten(), mean=-1, cov=sigma**2), (M, N)) # L(minus)
    
    # shape it back, will be useful later
    logodds = np.reshape(logodds, (M, N))

    # initial mu array: we set to 0 everything
    mu_prev  = np.zeros(shape=(M,N))

    # start updating mu
    for i in tqdm(range(n_iter)):
        # update mu as fixed point algorithm, no damping!
        mu_next = update_mu(mu_prev,J,logodds)

        # compute a_next
        a_next  = compute_a(mu_next,J,logodds)
        qxp1    = sigmoid(2*a_next)  # q(x=+1)
        qxm1    = sigmoid(-2*a_next) # q(x=-1)

        # compute entropy
        Hx   = -qxm1*np.log(qxm1+1e-10) - qxp1*np.log(qxp1+1e-10)

        # compute ELBO
        ELBO[i] = log_like(mu_next,J) + np.sum(qxp1*logp1 + qxm1*logm1) + np.sum(Hx) 

        # set mu_next to my_prev
        mu_prev = mu_next
    
    return mu_next, ELBO

# main        
def VI_Ising_Model_demo(noise_level=2.0,J=1.0,n_iter=10):
    # set seed
    np.random.seed(0)

    # load and noise the data
    file_      = "./figures/bayes.bmp"
    img_binary = load_data(file_)
    M, N       = img_binary.shape
    print(f"Size of image: {M}-by-{N}") 
    y          = img_binary + noise_level*np.random.randn(M, N) #y_i ~ N(x_i; sigma^2);
    plt.figure()
    plt.imshow(y,cmap='Greys', interpolation='nearest')

    # no do VI
    mu_VI, ELBO = doVI(
                        y,
                        J      = J,
                        sigma  = noise_level,
                        n_iter = n_iter
                      )

    # plot denoised image
    plt.figure()
    plt.imshow(mu_VI,cmap='Greys', interpolation='nearest')

    # plot ELBO
    plt.figure()
    plt.plot(ELBO, color='b', lw=2.0, label='ELBO')
    plt.title('Variational Inference for Ising Model')
    plt.xlabel('iterations'); plt.ylabel('ELBO objective')
    plt.legend(loc='upper left')

