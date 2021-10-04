#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# GMM class
# most important: logpdf evaluation
class GaussianMixture1D:
    def __init__(self, mixture_probs, means, stds):
        self.num_mixtures = len(mixture_probs)
        self.mixture_probs = mixture_probs
        self.means = means
        self.stds = stds

    def sample(self, num_samples=1):
        mixture_ids = np.random.choice(self.num_mixtures, size=num_samples, p=self.mixture_probs)
        result = np.zeros([num_samples])
        for sample_idx in range(num_samples):
            result[sample_idx] = np.random.normal(
                loc=self.means[mixture_ids[sample_idx]],
                scale=self.stds[mixture_ids[sample_idx]]
            )
        return result

    def logpdf(self, samples):
        ### computes log of mixture through log sum exp (a)
        ### a = log(mixture_probs) + logpdf_of_mixture_components
        mixture_logpdfs = np.zeros([len(samples), self.num_mixtures])
        for mixture_idx in range(self.num_mixtures):
            mixture_logpdfs[:, mixture_idx] = scipy.stats.norm.logpdf(
                samples,
                loc=self.means[mixture_idx],
                scale=self.stds[mixture_idx]
            )
        return logsumexp(mixture_logpdfs + np.log(self.mixture_probs), axis=1)

    def pdf(self, samples):
        ### exponentiate logp
        return np.exp(self.logpdf(samples))

# compute KL through numerical integration
def approx_kl(gmm_1, gmm_2, xs):
    ### approximate integral using trapezoid rule
    ### x is given
    ### f(x) is computed f(x) = p(x) * [log p(x) - log q(x)]
    ### then np.trapz is called
    ys = gmm_1.pdf(xs) * (gmm_1.logpdf(xs) - gmm_2.logpdf(xs))
    return np.trapz(ys, xs)

# minimize KL(p||q) through iterating over possible solutions
def minimize_pq(p, xs, q_means, q_stds):
    q_mean_best  = None
    q_std_best   = None
    kl_best     = np.inf
    # loop over all combinations of variational means and standard deviations
    # to find the one that minimizes KL(p||q) -> spread out
    for q_mean in q_means:
        for q_std in q_stds:
            q = GaussianMixture1D(np.array([1]), np.array([q_mean]), np.array([q_std]))
            kl = approx_kl(p, q, xs)
            if kl < kl_best:
                kl_best = kl
                q_mean_best = q_mean
                q_std_best = q_std

    q_best = GaussianMixture1D(np.array([1]), np.array([q_mean_best]), np.array([q_std_best]))
    return q_best, kl_best

# minimize KL(q||p) through iterating over possible solutions
def minimize_qp(p, xs, q_means, q_stds):
    q_mean_best = None
    q_std_best = None
    kl_best = np.inf
    # loop over all combinations of variational means and standard deviations
    # to find the one that minimizes KL(q||p) -> mode seeking
    for q_mean in q_means:
        for q_std in q_stds:
            q = GaussianMixture1D(np.array([1]), np.array([q_mean]), np.array([q_std]))
            kl = approx_kl(q, p, xs)
            if kl < kl_best:
                kl_best = kl
                q_mean_best = q_mean
                q_std_best = q_std

    q_best = GaussianMixture1D(np.array([1]), np.array([q_mean_best]), np.array([q_std_best]))
    return q_best, kl_best

# main function
def KL_calculation_demo():
    p_second_means_min = 0
    p_second_means_max = 20
    num_p_second_means = 4
    p_second_mean_list = np.linspace(p_second_means_min, p_second_means_max, num_p_second_means)

    # mixture of Gaussians
    p               = [None] * num_p_second_means

    # KL(p||q)-> spread out
    q_best_forward  = [None] * num_p_second_means
    kl_best_forward = [None] * num_p_second_means

    # KL(q||p)-> mode seeking
    q_best_reverse  = [None] * num_p_second_means
    kl_best_reverse = [None] * num_p_second_means

    ## loop over means
    for p_second_mean_idx, p_second_mean in enumerate(p_second_mean_list):
        # fix mixture pprobabilities as 0.5 and 0.5
        p_mixture_probs = np.array([0.5, 0.5])
        
        # first mean is always 0, second mean is different
        p_means = np.array([0, p_second_mean])

        # standard deviations kept at 1.0
        p_stds  = np.array([1, 1])

        # fix GMM as element of array p
        p[p_second_mean_idx] = GaussianMixture1D(p_mixture_probs, p_means, p_stds)

        q_means_min = np.min(p_means) - 1
        q_means_max = np.max(p_means) + 1
        num_q_means = 20
        q_means     = np.linspace(q_means_min, q_means_max, num_q_means)

        q_stds_min = 0.1
        q_stds_max = 5
        num_q_stds = 20
        q_stds = np.linspace(q_stds_min, q_stds_max, num_q_stds)

        # basically do -+ 3 sigmas from the smallest/largest center on x-axis: generate 1000 points
        trapz_xs_min     = np.min(np.append(p_means, q_means_min)) - 3 * np.max(np.append(p_stds, q_stds_max))
        trapz_xs_max     = np.max(np.append(p_means, q_means_min)) + 3 * np.max(np.append(p_stds, q_stds_max))
        num_trapz_points = 1000
        trapz_xs         = np.linspace(trapz_xs_min, trapz_xs_max, num_trapz_points)

        # here find the best variational distribution (single component) for KL(p||q)->spread out
        q_best_forward[p_second_mean_idx], kl_best_forward[p_second_mean_idx] = minimize_pq(
            p[p_second_mean_idx], trapz_xs, q_means, q_stds
        )
        # here find the best variational distribution (single component) for KL(q||p)->mode seeking
        q_best_reverse[p_second_mean_idx], kl_best_reverse[p_second_mean_idx] = minimize_qp(
            p[p_second_mean_idx], trapz_xs, q_means, q_stds
        )

    # plotting
    fig, axs = plt.subplots(nrows=1, ncols=num_p_second_means, sharex=True, sharey=True)
    fig.set_size_inches(12.0, 8.0)
    for p_second_mean_idx, p_second_mean in enumerate(p_second_mean_list):
        xs_min          = -5
        xs_max          = 30
        num_plot_points = 1000
        xs              = np.linspace(xs_min, xs_max, num_plot_points)
        
        # plot p(x)
        axs[p_second_mean_idx].plot(xs, p[p_second_mean_idx].pdf(xs), label='$p$\n(mixture)', color='blue')
        # plot q(x) from optimising KL(p||q)
        axs[p_second_mean_idx].plot(xs, q_best_forward[p_second_mean_idx].pdf(xs), label='$\mathrm{argmin}_q \,\mathrm{KL}(p || q)$\n(global fit)', color='red', linestyle='dashed')
        # plot q(x) from optimising KL(q||p)
        axs[p_second_mean_idx].plot(xs, q_best_reverse[p_second_mean_idx].pdf(xs), label='$\mathrm{argmin}_q \,\mathrm{KL}(q || p)$\n(mode seeking)', color='green', linestyle='dotted')

        axs[p_second_mean_idx].spines['right'].set_visible(False)
        axs[p_second_mean_idx].spines['top'].set_visible(False)
        axs[p_second_mean_idx].set_yticks([])
        axs[p_second_mean_idx].set_xticks([])

    axs[2].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0), fontsize='small')

