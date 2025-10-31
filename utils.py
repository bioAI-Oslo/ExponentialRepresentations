import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

def compute_ratemaps(x, y, p, bins = 32):
    # Compute ratemaps for each unit given positions (x,y) and rates p 
    hists = scipy.stats.binned_statistic_2d(x,y, p.T, bins = bins)[0]
    return hists

def random_orthogonal_matrix(n):
    # Create random nxn orthogonal matrix R 
    R = torch.tensor(scipy.stats.ortho_group.rvs(n))
    R = R.to(torch.float32)
    return R

def starting_state(R):
    p0 = R.T@torch.ones(len(R), dtype=torch.float32)/np.sqrt(len(R))
    p0 = p0.to(torch.float32)
    return p0

def rectify(x): # relu
    return np.maximum(x, 0)

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


def ratemap_collage(ratemaps, rows = 5, cols=5, figsize=(5, 5), cmap="viridis", vmin=None, vmax=None, **kwargs):
    """ plot collage of ratemaps

    Args:
        ratemaps (np ndarray): array of shape (N, bins, bins). 
            Should contain N ratemaps, each of shape (bins, bins)
        cols (int, optional): Number of units in each row. Defaults to 5.
        figsize (tuple, optional): Size of figure. Defaults to (5,5).
        cmap (str, optional): Matplotlib colormap. Defaults to "viridis".

    """
    fig, axs = plt.subplots(cols, rows, figsize=figsize, **kwargs)

    count = 0
    for row in range(rows):
        for col in range(cols):
            axs[col, row].axis("off")
            if count < len(ratemaps):
                axs[col, row].imshow(ratemaps[count], cmap=cmap, vmin=vmin, vmax=vmax)
            count += 1
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axs


def orthogonal_basis(q0):
    # find basis orthogonal to q0 using Gram-Schmidt orthonormalization
    # q0 is the first vector of the basis

    basis = []
    basis.append(q0)
    
    for i in range(len(q0) - 1):
        
        q = np.random.uniform(-1, 1, len(q0)) # i-th column of A
        for j in range(len(basis)):
            z = basis[j]
            q = q - np.dot(q, z) * z
        
        # normalize q
        q = q / np.sqrt(np.dot(q, q))
        basis.append(q)
    
    return np.array(basis)