"""
Defines the firn PDE: coefficients, grids, IC/BC, and discrete residual operator.
No PyTorch logic here. Pure physics + discretization.
"""

class FirnProblem:
    def __init__(self, z, t, Da, v0, params):
        self.z = z
        self.t = t
        self.Da = Da
        self.v0 = v0
        self.params = params

    def ic(self, z):
        """Initial condition V(z, t=0)."""
        pass

    def bc_left(self, t):
        """Left boundary condition at z=0."""
        pass

    def bc_right(self, t):
        """Right boundary condition at z=z_max."""
        pass

    def residual(self, V_pred):
        """Compute discrete PDE residual on the stencil."""
        pass
