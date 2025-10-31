import torch

class ExponentialRepresentation(torch.nn.Module):
    def __init__(self, A, B):
        """Generate spatial representations using matrix exponential

        Args:
            A : Generator matrix in x direction
            B : Generator matrix in y direction
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = torch.clone(A).detach().to(self.device)
        self.B = torch.clone(B).detach().to(self.device)

    def forward(self, v, p0):
        """Generate spatial representations along
            trajectory spanned by v from starting state p0

        Args:
            v (torch.tensor): Trajectory velocities, shape (batch size,2)
            p0 (torch.tensor): Starting state, shape (n_units)

        Returns:
            p: Generated representation, shape (batch size, n_units)
        """
        v = v.to(self.device)
        p0 = p0.to(self.device)

        Q = torch.matrix_exp(
            v[:, 0, None, None] * self.A + v[:, 1, None, None] * self.B
        )
        p = torch.tensordot(Q, p0, dims=[(-1,), (-1,)])  # gpu goes vromm
        return p


class RemappingModel(ExponentialRepresentation):
    def __init__(self, A, B, C):
        """Generate spatial representations using matrix exponential

        Args:
            A : Generator matrix in x direction
            B : Generator matrix in y direction
            C : Generator matrix for context
        """
        super().__init__(A, B)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = torch.clone(C).detach().to(self.device)

    def forward(self, v, p0):
        """Generate spatial representations along
            trajectory spanned by v from starting state p0

        Args:
            v (torch.tensor): Trajectory velocities and context, shape (batch size,3)
            p0 (torch.tensor): Starting state, shape (n_units)

        Returns:
            p: Generated representation, shape (batch size, n_units)
        """
        v = v.to(self.device)
        p0 = p0.to(self.device)

        Q = torch.matrix_exp(
            v[:, 0, None, None] * self.A
            + v[:, 1, None, None] * self.B
            + v[:, 2, None, None] * self.C
        )

        p = torch.tensordot(Q, p0, dims=[(-1,), (-1,)])  # gpu goes vromm
        return p
