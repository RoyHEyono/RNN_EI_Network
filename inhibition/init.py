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


def parametrized_ln_mean_weight(mean_linear, W_XE, W_EE, bias=None):
    """Init the mean predictor as the exact per-sample mean of the pre-activation.

    Mirrors the subtractive-inhibition step of the notebook's divisive-recurrent
    scheme. With ``feat = cat([x_t, h_prev])`` and pre-activation
    ``Z = W_XE x + W_EE h + b``, the mean over hidden units is linear in ``feat``:
    ``mean_dim(Z) = mean_row([W_XE | W_EE]) @ feat + mean(b)``. So a single
    ``Linear(feat_dim, 1)`` reproduces the LayerNorm mean exactly.
    """
    W_XE_np = W_XE.detach().cpu().numpy()
    W_EE_np = W_EE.detach().cpu().numpy()
    W_full = np.concatenate([W_XE_np, W_EE_np], axis=1)  # (n_h, n_in + n_h)
    mean_row = W_full.mean(axis=0, keepdims=True)  # (1, feat_dim)
    mean_linear.weight.data = torch.from_numpy(mean_row).float()
    bias_mean = float(bias.detach().cpu().mean()) if bias is not None else 0.0
    mean_linear.bias.data = torch.tensor([bias_mean], dtype=torch.float32)
    mean_linear.weight.clamp = True
    mean_linear.bias.clamp = True


def parametrized_ln_var_weight(var_first, var_second, W_XE, W_EE, bias=None, n_div=None):
    """Init the variance predictor as the exact pre-activation variance.

    Mirrors the divisive-inhibition step of the notebook's recurrent scheme.
    Concatenate ``W_full = [W_XE | W_EE | b]`` (bias as the constant-1 input
    column), center it (subtractive inhibition), then SVD ``W_eff = U S Vᵀ`` and
    set ``B_IX = diag(S) Vᵀ`` and ``B_EI = 1/n_h``. Then
    ``var = B_EI @ (B_IX @ [feat, 1])²`` equals ``var(Z)`` exactly.

    ``var_first`` is a ``Linear(feat_dim, n_div)`` holding ``B_IX`` (its bias
    absorbs the constant-1 / bias column). ``var_second`` is a
    ``Linear(n_div, 1)`` performing the ``B_EI`` averaging.
    """
    ne, n_in = W_XE.shape
    _, n_h = W_EE.shape  # ne == n_h
    feat_dim = n_in + n_h

    W_XE_np = W_XE.detach().cpu().numpy()
    W_EE_np = W_EE.detach().cpu().numpy()
    if bias is not None:
        bias_col = bias.detach().cpu().numpy().reshape(-1, 1)
    else:
        bias_col = np.zeros((ne, 1))
    W_full = np.concatenate([W_XE_np, W_EE_np, bias_col], axis=1)  # (n_h, feat_dim + 1)

    W_IX = W_full.mean(axis=0, keepdims=True)  # (1, feat_dim + 1)
    W_eff = W_full - np.ones((ne, 1)) @ W_IX

    _, S, V_T = np.linalg.svd(W_eff)
    if n_div is None:
        n_div = ne
    k = min(n_div, S.shape[0])
    B_IX = np.diag(S[:k]) @ V_T[:k, :]  # (k, feat_dim + 1)

    B_IX_w = B_IX[:, :feat_dim]  # (k, feat_dim)
    B_IX_b = B_IX[:, feat_dim]   # (k,) constant-1 / bias column
    var_first.weight.data = torch.from_numpy(B_IX_w).float()
    var_first.bias.data = torch.from_numpy(B_IX_b).float()
    var_first.weight.clamp = False
    var_first.bias.clamp = False

    B_EI = np.ones((1, k)) / ne
    var_second.weight.data = torch.from_numpy(B_EI).float()
    var_second.bias.data = torch.zeros(1, dtype=torch.float32)
    var_second.weight.clamp = True
    var_second.bias.clamp = True