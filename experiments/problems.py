"""Problem sets used in the study.

classical(key)            the 23 classical functions, as in benchmarks.py
shifted_rotated(key, s)   f(M (x - o)) for the scalable functions F1-F13: o is a
                          shift drawn in 0.8 x the box and M a random orthogonal
                          matrix, both fixed by the seed s. Bounds unchanged.
cec2022(i, d)             CEC 2022 single-objective bound-constrained suite via
                          opfunu (shifted and rotated by construction).
Every problem is returned as (f, lb, ub, d, fstar, name) with f(X, rng) -> (N,).
"""
import numpy as np
from benchmarks import FUNCS, bounds


def classical(key):
    name, f, *_ = FUNCS[key]
    lb, ub, d = bounds(key)
    return f, lb, ub, d, FUNCS[key][5], f'{key}'


def rand_orth(d, rng):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))


def shifted_rotated(key, seed=2026):
    name, base, *_ = FUNCS[key]
    lb, ub, d = bounds(key)
    rng = np.random.default_rng([seed, int(key[1:])])
    o = 0.8 * (lb + rng.random(d) * (ub - lb))
    M = rand_orth(d, rng)
    # F8 optimum is not at the origin; shift-rotate keeps the landscape but the
    # optimum moves to o + M^T x*, which may leave the box, so F8 is excluded.
    def f(X, r=None):
        return base((X - o) @ M.T, r)
    return f, lb, ub, d, FUNCS[key][5], f'SR-{key}'


def cec2022(i, d=10):
    import opfunu
    cls = getattr(opfunu.cec_based.cec2022, f'F{i}2022')
    F = cls(ndim=d)
    lb, ub = F.bounds[:, 0].astype(float), F.bounds[:, 1].astype(float)

    def f(X, r=None):
        return np.array([F.evaluate(x) for x in X])
    return f, lb, ub, d, float(F.f_global), f'C22-F{i}'
