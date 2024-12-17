import numpy as np
import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
torch.set_default_dtype(torch.float64)


class ExponentialRepresentation(object):
    def __init__(self, A, B):
        """_summary_

        Args:
            A (_type_): _description_
            B (_type_): _description_
        """
        self.A = A 
        self.B = B 
        # Add exception for non-skew A, B?

        commutation = self.check_commutation(self.A, self.B)
        assert commutation, "A and B do not (numerically) commute."

        self.QA, self.QB = self.construct_exponential()    
        # Add exception when this fails

    def construct_exponential(self):
        # construct component transformations using eigenvalue decomposition
        lA, VA = torch.linalg.eig(self.A)
        lB, VB = torch.linalg.eig(self.B)

        VA_inv = torch.linalg.inv(VA)
        VB_inv = torch.linalg.inv(VB)

        QA = lambda vx: VA@torch.diag(torch.exp(vx*lA))@VA_inv
        QB = lambda vy: VB@torch.diag(torch.exp(vy*lB))@VB_inv
        return QA, QB 

    def check_commutation(self, X, Y, eps = 1e-11):
        commutator = X@Y - Y@X 
        print(torch.max(commutator))
        if torch.max(commutator) < eps:
            commute = True
        else:
            commute = False
        return commute
    
    def Q(self, v):
        # construct full transformation matrix
        return self.QA(v[...,0])@self.QB(v[...,1])
    
    def __call__(self, v, f0):
        """ Generate spatial representations along 
            trajectory spanned by v from starting state f0

        Args:
            v (torch.tensor): Trajectory velocities, shape (timesteps)
            f0 (torch.tensor): Starting state, shape (n_units)

        Returns:
            f: Generated representation, shape (timesteps, n_units)
        """
        
        f = torch.zeros((len(v), len(f0)))
        f[0] = f0 

        for t in tqdm(range(len(v)-1)):
            # print(self.Q(v[t]))
            f[t+1] = torch.real(self.Q(v[t]))@f[t]
        return f

