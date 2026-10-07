"""Minimal example: minimise the 30-D sphere function with DRTO."""
import numpy as np
from drto import drto

d, N, T = 30, 30, 500                       # dimension, population, iterations
lb, ub = -100 * np.ones(d), 100 * np.ones(d)
sphere = lambda X, rng=None: np.sum(X**2, axis=1)   # vectorised: X is (N, d)

rng = np.random.default_rng(1)
X0 = lb + rng.random((N, d)) * (ub - lb)
best, curve = drto(sphere, lb, ub, X0, T, rng)
print(f"best f = {best:.3e} after {N * (T + 1)} evaluations")
