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

class SimpleDatasetMaker(object):
    """
    Simple test dataset maker; square box + bounce off walls
    """

    def __init__(self, chamber_size=1, stddev=4*np.pi, rayleigh_scale=0.025):
        self.chamber_size = chamber_size
        self.stddev = stddev
        self.rayleigh_scale = rayleigh_scale
        
    def bounce(self, r, v):
        outside = np.abs(r + v) >= self.chamber_size
        v[outside] = -v[outside]
        return v

    def generate_dataset(self, samples, timesteps, device = "cpu"):
        r = np.zeros((samples, timesteps, 2))

        s = np.random.rayleigh(self.rayleigh_scale, (samples, timesteps))
        prev_hd = np.random.uniform(0, 2*np.pi, samples)
        r[:, 0] = np.random.uniform(-self.chamber_size, self.chamber_size, (samples, 2))

        for i in range(timesteps - 1):
            hd = np.random.vonmises(prev_hd, self.stddev, samples)
            prop_v = s[:, i, None] * np.stack((np.cos(hd), np.sin(hd)), axis=-1)
            v = self.bounce(r[:, i], prop_v)
            prev_hd = np.arctan2(v[:,1], v[:,0])
            r[:, i+1] = r[:, i] + v # dt = 1

        v = np.diff(r, axis = 1)
        return torch.tensor(r.astype('float32'), device = device), torch.tensor(v.astype('float32'), device = device)



def ratemap_collage(ratemaps, cols=5, figsize=(5, 5), cmap="viridis", vmin=None, vmax=None, **kwargs):
    """ plot collage of ratemaps

    Args:
        ratemaps (np ndarray): array of shape (N, bins, bins). 
            Should contain N ratemaps, each of shape (bins, bins)
        cols (int, optional): Number of units in each row. Defaults to 5.
        figsize (tuple, optional): Size of figure. Defaults to (5,5).
        cmap (str, optional): Matplotlib colormap. Defaults to "viridis".

    """
    ratio = len(ratemaps) // cols 
    rows = ratio if ((len(ratemaps) % cols) == 0) else ratio + 1
    fig, axs = plt.subplots(cols, rows, figsize=figsize, **kwargs)

    for i in range(len(ratemaps)):
        axs[i // cols, i % cols].imshow(ratemaps[i].T, origin="lower", interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i // cols, i % cols].axis("off")
    
    return fig, axs
