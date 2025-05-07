import numpy as np
import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
torch.set_default_dtype(torch.float64)

class ExponentialRepresentation(torch.nn.Module):
    def __init__(self, A, B):
        """ Generate spatial representations using matrix exponential

        Args:
            A : Generator matrix in x direction
            B : Generator matrix in y direction
        """
        super().__init__()
        self.A = A 
        self.B = B 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = torch.tensor(A, device=self.device, requires_grad=False)
        self.B = torch.tensor(B, device=self.device, requires_grad=False)
  
    def forward(self, v, p0):
        """ Generate spatial representations along 
            trajectory spanned by v from starting state p0

        Args:
            v (torch.tensor): Trajectory velocities, shape (batch size,2)
            p0 (torch.tensor): Starting state, shape (n_units)

        Returns:
            p: Generated representation, shape (batch size, n_units)
        """
        v = v.to(self.device)
        p0 = p0.to(self.device)

        Q = torch.matrix_exp(v[:,0,None,None]*self.A + v[:,1,None,None]*self.B) 

        p = torch.tensordot(Q, p0, dims = [(-1,), (-1,)]) # gpu goes vromm
        return p
