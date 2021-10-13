#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.distributions import Normal
from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt

# sample from Gumbel
def sample_gumbel(n,k):
    unif = torch.distributions.Uniform(0,1).sample((n,k))
    g    = -torch.log(-torch.log(unif))
    return g

# provide vector of categorical probabilities (pi), number of samples (n) and temperature (T) to obtain gumbel-softmax samples
def sample_gumbel_softmax(pi,n,T):
    # number of categories
    k = len(pi)
    # obtain Gumbel samples
    g = sample_gumbel(n, k)
    # start doing anneled softmax
    h     = (g + torch.log(pi))/T
    h_max = h.max(dim=1, keepdim=True)[0]
    h     = h - h_max
    h_exp = torch.exp(h)
    out   = h_exp/h_exp.sum(dim=1,keepdim=True)
    return out

# main
def gumbel_softmax_demo(seed = 0,n_samples = 1000,k=7,temperatures = [0.1,1.0]):
    torch.manual_seed(seed)
    
    # create a probability vector pi
    pi = torch.randint(high=100, size=(k,), dtype=torch.float)
    pi = pi/pi.sum()

    print(f"Our probabilities vector with {k} categories: {pi.numpy()}")

    # lets do many samples from Gumbel Softmax and everage accros categories to have a distribution
    #temperatures = [0.01,0.1,1.0,10.0]

    fig,axs      = plt.subplots(
                                    nrows   = len(temperatures)+1, # for all temperatures plus true categorical
                                    ncols   = 1,
                                    figsize = (5,20)
                               )

    # loop pver temperatures, start with 1 (idx 0 is for true categorical)
    for idx in range(len(temperatures)):
        # get temperature
        T = temperatures[idx]

        # obtain samples-> these samples are not one-hot encoded, but pretty close
        z = sample_gumbel_softmax(
                                    pi,        # vector of probabilities
                                    n_samples, # number of samples
                                    T          # annealing temperature
                                  )
        # compute average across number of samples (dim=0)
        z_mean = z.mean(dim=0).numpy()

        axs[idx+1].bar(
                            np.arange(k)+1, 
                            z_mean
                        )
        axs[idx+1].set_title(f"Gumbel-Softmax approximation with temperature {T}")
    
    # do true categorical
    true_categorical = torch.distributions.Categorical(probs=pi)
    true_samples     = true_categorical.sample((n_samples,))

    # need to convert to one-hot
    one_hot                                       = torch.zeros(n_samples,k)
    one_hot[range(n_samples),true_samples.long()] = 1

    # compute mean
    one_hot_mean = one_hot.mean(dim=0)

    
    # plot
    axs[0].bar(
                            np.arange(k)+1, 
                            one_hot_mean
                        )
    axs[0].set_title("True Categorical Distribution")
    


    





