import numpy as np
import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
torch.set_default_dtype(torch.float64)

class ExponentialRepresentation(object):
    def __init__(self, A, B):
        """ Generate spatial representations using matrix exponential

        Args:
            A : Generator matrix in x direction
            B : Generator matrix in y direction
        """
        self.A = A 
        self.B = B 
  
    def __call__(self, v, p0):
        """ Generate spatial representations along 
            trajectory spanned by v from starting state p0

        Args:
            v (torch.tensor): Trajectory velocities, shape (timesteps)
            p0 (torch.tensor): Starting state, shape (n_units)

        Returns:
            p: Generated representation, shape (timesteps, n_units)
        """
        
        p = torch.zeros((len(v), len(p0))).to(torch.float32)
        p[0] = p0 

        for t in tqdm(range(len(v)-1)):
            p[t+1] = torch.matrix_exp(v[t,0]*self.A + v[t,1]*self.B)@p[t]
        return p
