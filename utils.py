import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

def plot_ratemaps(x, y, p, n = 5, bins = 32, figsize = (2,2)):
    fig, ax = plt.subplots(n, n, figsize = figsize)
    axs = ax.ravel()
    hists = binned_statistic_2d(x,y, p.T, bins = bins)[0]

    for i in range(n**2):
        axs[i].imshow(hists[i])
        axs[i].axis("off")
    return hists

def construct_exponential(sA, sB, Gr, s = 1):
    R = torch.matrix_exp(s*Gr)
    A = R.T@sA@R
    B = R.T@sB@R
    return A, B, R

def rectify(x): # relu
    return np.maximum(x, np.mean(x))

def block_diag(fill_vals):
    # Helper function to create block-diagonal skew-symmetric matrices
    blocks = []
    for i in range(len(fill_vals)):
        block = torch.zeros((2,2))
        
        block[0,1] = fill_vals[i]
        block[1,0] = -block[0,1]
        blocks.append(block)
    return torch.block_diag(*blocks)

def batch_inference(model, r, p0):
    p_ = []
    for ri in tqdm(r):
        p_.append(model(ri, p0).detach().cpu()) # run
    p_ = np.concatenate(p_)
    return p_

def interleave(a, b):
    return torch.cat([torch.tensor([ai, bi]) for (ai, bi) in zip(a,b)])