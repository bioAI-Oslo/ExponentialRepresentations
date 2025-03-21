import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

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