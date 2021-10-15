#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import kl_divergence
import numpy as np

# Discrete VAE class
class DiscreteVAE(nn.Module):
    def __init__(self,N,K):
        super(DiscreteVAE,self).__init__()
        
        # keep our K and N
        self.K = K
        self.N = N

        # create encoder -> logits are not here yet!
        self.encoder = self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.K*self.N)
        )

        # create decoder -> since our values are 0/1 pixels, we have LogSigmoid at the end
        self.decoder = nn.Sequential(
            nn.Linear(self.N*self.K, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.LogSigmoid()
        )
    
    def forward(self,X,temperature = 1.0,hard=False):
        # encode
        X              = self.encoder(X)                      # pass through encoder
        unnorm_logits  = X.view(-1, self.N, self.K)           # get shape [batch_size,N,K]
        norm_logits    = F.log_softmax(unnorm_logits, dim=-1) # obtain logits across last dimension (K)

        # sampling from Gumbel Softmax
        z       = F.gumbel_softmax(
                                    unnorm_logits,
                                    tau    = temperature,
                                    hard   = hard,
                                    dim    = -1           # to perform softmax along last dimension (which is K)
                                  )
        # decode back
        z_reshaped = z.view(-1,self.N*self.K) # change from 3D to 2D: [batch_size, N*K]
        out_logits = self.decoder(z_reshaped)

        # return:
        # 1. norm_logits: normalized output of encoder -> need for KL
        # 2. z:           latent RVs (not really needed but just in case)
        # 3. out_logits: logits to get Bernoulli RV (need to reconstruction error)
        return norm_logits,z,out_logits
    
    # decode z
    def decode(self,z):
        z_reshaped  = z.view(-1,self.N*self.K)
        out         = self.decoder(z_reshaped)
        return out

# function to compute ELBO
def loss_for_discrete_vae(y,encoder_logits,decoder_logits,K):
    # instantiate our p(y|z)
    py_z = torch.distributions.Bernoulli(logits=decoder_logits)
    # compute logpprob of data (y)
    reconstruction_error = py_z.log_prob(y).sum(dim=-1)

    # compute KL
    # variational posterior categorical distribution
    posterior_dist = torch.distributions.Categorical(logits=encoder_logits)
    # prior distribution with p=1/K for all categories
    prior_dist     = torch.distributions.Categorical(probs=torch.ones_like(encoder_logits)/K)
    # compute KL divergence
    KL             = kl_divergence(posterior_dist, prior_dist).sum(-1)

    # combine as ELBO = reconstruction - KL
    total_loss = reconstruction_error - KL
    
    # change sign and average over batch size
    out = -total_loss.mean()

    return out

# function for sampling from the prior
def sample_from_prior(model,N,K,n_samples = 1):
    # create our one-hot vectors
    M                                    = n_samples*N
    np_z                                 = np.zeros((M,K))
    np_z[range(M),np.random.choice(K,M)] = 1
    # reshape so we have shape [n_samples,N,K]
    np_z                                 = np.reshape(np_z,[n_samples,N,K])
    # convert to tensor
    z_tensor                             = torch.from_numpy(np_z).float()

    with torch.no_grad():
        logp        = model.decode(z_tensor)
        dist_x      = torch.distributions.Bernoulli(logits=logp)
        sampled_img = dist_x.sample((1,))
    return sampled_img.view(-1,28,28)

# to plot
def plot_mnist_image(img,lbl=None):
    fig = plt.figure()
    plt.imshow(img, cmap='gray',interpolation='nearest')
    if lbl is not None:
        plt.title(lbl)
    plt.show()

# main function
def discrete_vae_demo(
                        K             = 10,   # number of classes (known from the problem)
                        N             = 70,   # number of categorical distributions (that's our choice)
                        NUM_EPOCHS    = 70,
                        freq          = 10,   
                        init_temp     = 1.0,
                        ANNEAL_RATE   = 0.000003,
                        MIN_TEMP      = 0.15,
                     ):
    ### seeds
    torch.manual_seed(0)
    np.random.seed(0)

    ### load data
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST-data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
        batch_size=128, shuffle=True)


    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./MNIST-data', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                        ])),
        batch_size=1, shuffle=True)
    
    ### plot one
    for X_test,y_lbl in test_loader:
        X_np = X_test.numpy()[0,0,:,:] # remove all unnecessary dimensions
        y_np = y_lbl.numpy()[0]
        plot_mnist_image(X_np,lbl=f"True label:{y_np}")
        break

    ### instantiate model and optimizer
    model     = DiscreteVAE(N,K)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0)

    iter_counter  = 0

    ### training
    for epoch in range(NUM_EPOCHS):
        
        loss_accumulator        = []
        temperature_accumulator = []
        
        for batch_idx, (X_train,_) in enumerate(train_loader):
        
            # change dimensionality of X_train
            data = X_train.view(-1, 28*28)
            
            # set temperature
            temperature = np.maximum(
                                        init_temp * np.exp(-ANNEAL_RATE * iter_counter), 
                                        MIN_TEMP
                                    )

            # forward pass
            encoder_logits,z,decoder_logits = model(
                                                        data,
                                                        temperature = temperature,
                                                        hard        = False
                                                    )
            
            # compute loss
            elbo = loss_for_discrete_vae(data,encoder_logits,decoder_logits,K)

            # clean pprevious gradients and do a step
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            # increase iter counter
            iter_counter +=1
            
            # cache losses and temperature
            loss_accumulator.append(elbo.item())
            temperature_accumulator.append(temperature)
        
        avg_loss = sum(loss_accumulator)/len(loss_accumulator)
        avg_temp = sum(temperature_accumulator)/len(temperature_accumulator)
        
        print(f"Epoch:{epoch} (# of batch updates:{iter_counter}), ELBO:{avg_loss:0.4f}, Temperature:{avg_temp:0.4f}")
            
            
        # show batch loss some every number of iterations
        if epoch%freq == 0:
            fig,axs       = plt.subplots(nrows=1,ncols=5,figsize=(15,7))
            prior_samples = sample_from_prior(model,N,K,n_samples = 5).numpy()
            for i in range(5):
                axs[i].imshow(
                                prior_samples[i,:,:], 
                                cmap          ='gray',
                                interpolation ='nearest'
                            )
            fig.suptitle(f"Samples from prior after epoch:{epoch}")
            plt.show()
    
    ### post-training sampling
    print("Post-training samples")
    n_samples     = 20
    prior_samples = sample_from_prior(model,N,K,n_samples = n_samples)
    prior_samples = prior_samples.numpy()
    fig,axs = plt.subplots(nrows=4,ncols=5,figsize=(20,10))

    for i in range(n_samples):
        row = i//5
        col = i%5
        
        # select one sample
        sample        = prior_samples[i,:,:]
        
        # plot
        axs[row,col].imshow(
                        sample, 
                        cmap          ='gray',
                        interpolation ='nearest'
                    )
        fig.suptitle("Post-training examples")


