"""
Loads MATLAB-generated forward solution and prepares supervised training pairs.
Also constructs FirnProblem with required grids and coefficients.
"""

import scipy.io as sio

def load_firn_forward(path):
    """Load firn_forward.mat and return z, t, V, coefficients."""
    return sio.loadmat(path)

def build_supervised_samples(z, t, V):
    """Return (z_i, t_i, V_i) as tensors for data loss."""
    pass
