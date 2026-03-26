import numpy as np
import torch


def calc_ln_mu_sigma(mean, var):
    """
    Helper function: given a desired mean and var of a lognormal dist 
    (the func arguments) calculates and returns the underlying mu and sigma
    for the normal distribution that underlies the desired log normal dist.
    """
    mu_ln = np.log(mean**2 / np.sqrt(mean**2 + var))
    sigma_ln = np.sqrt(np.log(1 + (var /mean**2)))
    return mu_ln, sigma_ln

def excitatory_weight(Wex, numerator=2, k=1):
    ne, n_input = Wex.shape
    target_std_wex = np.sqrt(numerator*ne/(n_input*(ne-1)))
    mu, sigma = calc_ln_mu_sigma(target_std_wex*k,target_std_wex**2)
    Wex_np = np.random.lognormal(mu, sigma, size=(ne, n_input))
    Wex.data = torch.from_numpy(Wex_np).float()
    Wex.clamp = True

def subtractive_excitatory_inhibitory_weight(Wix, Wex):
    ni, _ = Wix.shape
    Wex_np = Wex.detach().numpy()
    Wix_np = np.ones(shape=(ni,1))*Wex_np.mean(axis=0,keepdims=True)
    Wix.data = torch.from_numpy(Wix_np).float()
    Wix.clamp = True

def subtractive_inhibitory_excitatory_weight(Wex, Wei, numerator=2, k=1):
    ne, n_input = Wex.shape
    ne, ni = Wei.shape
    target_std_wex = np.sqrt(numerator*ne/(n_input*(ne-1)))
    mu, sigma = calc_ln_mu_sigma(target_std_wex*k,target_std_wex**2)
    Wei_np = np.random.lognormal(mu, sigma, size=(ne, ni))
    
    Wei_np /= Wei_np.sum(axis=1, keepdims=True)
    Wei.data = torch.from_numpy(Wei_np).float()
    Wei.clamp = True

def divisive_excitatory_inhibitory_weight(Wei, Wex, Wix, Bix):
    ne, n_input = Wex.shape
    Wei_np = Wei.detach().numpy()
    Wex_np = Wex.detach().numpy()
    Wix_np = Wix.detach().numpy()

    W = Wex_np - Wei_np@Wix_np

    _, S, V_T = np.linalg.svd(W)
    V = V_T[:ne].T

    Bix_np = np.diag(S) @ V.T
    Bix.data = torch.from_numpy(Bix_np).float()
    Bix.clamp = False

def divisive_inhibitory_excitatory_weight(Wex, Bei):
    ne, _ = Wex.shape
    Bei_np = np.ones((ne,ne))/ne
    Bei.data = torch.from_numpy(Bei_np).float()
    Bei.clamp = True