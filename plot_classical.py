import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load MATLAB data
data = loadmat("data/raw/firn_forward.mat")
z = data["z"].squeeze()         # shape (129,)
t = data["t"].squeeze()         # shape (129,)
V = data["V"]                   # shape (129, 129)

# Create heatmap
plt.figure(figsize=(7, 5))
plt.imshow(V, aspect='auto', origin='lower',
           extent=[t.min(), t.max(), z.min(), z.max()])
plt.colorbar(label="Concentration")
plt.xlabel("Time")
plt.ylabel("Depth")
plt.title("Classical Solver Output V(z, t)")

plt.tight_layout()
plt.show()
