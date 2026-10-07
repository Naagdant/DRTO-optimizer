"""Diversity Reference Tracking Optimizer (DRTO), frozen after the development stage.
Budget is counted in function evaluations: N*(T+1), identical to the baselines."""
import numpy as np


def drto(f, lb, ub, X0, T, rng, Fn=0.5, kp=0.2, ki=0.05, Dend=1e-8, p=0.2, CR=0.9,
         control=True, return_trace=False, g=0.3, crmode='learn', lr=0.1,
         act='sigma', kinj=2.0, noise=True):
    N, d = X0.shape
    span = ub - lb; diag = np.linalg.norm(span)
    budget = N * (T + 1)
    X = X0.copy(); fit = f(X, rng); nfe = N
    div = lambda Z: np.mean(np.linalg.norm(Z - Z.mean(0), axis=1)) / diag
    D0 = max(div(X), 1e-12)
    I = 0.0; pcr = 0.5
    hist_n, hist_b = [nfe], [fit.min()]
    tr = []
    idx = np.arange(N); npb = max(2, int(round(p * N)))
    while nfe < budget:
        tau = nfe / budget
        D = max(div(X), 1e-300)
        Dref = D0 * (Dend / D0) ** tau
        e = np.log(Dref) - np.log(D)
        u = 1.0
        if control:
            I = float(np.clip(I + e, -20 / ki, 20 / ki))
            u = float(np.exp(np.clip(kp * e + ki * I, -10, 10)))
        tr.append((tau, D, Dref, u))
        s0 = g * Dref * diag / np.sqrt(d)
        sig = s0 * u if act in ('sigma', 'all') else s0
        npb_t = int(np.clip(round(p * N * u), 2, N)) if act in ('p', 'all') else npb
        m = min(N, budget - nfe)
        Fi = np.clip(Fn * (1 + 0.1 * rng.standard_normal(N)), 0.02, 1.2)[:, None]
        order = np.argsort(fit)
        pb = X[order[rng.integers(0, npb_t, N)]]
        r1 = (idx + rng.integers(1, N, N)) % N
        r2 = (idx + rng.integers(1, N, N)) % N
        Vm = X + Fi * (pb - X) + Fi * (X[r1] - X[r2])
        if noise:
            Vm = Vm + sig * rng.standard_normal((N, d))
        if crmode == 'fixed':
            hic = np.ones(N, bool); CRi = np.full(N, CR)
        else:
            hic = rng.random(N) < pcr
            CRi = np.where(hic, 0.9, 0.1)
        cross = rng.random((N, d)) < CRi[:, None]
        cross[idx, rng.integers(0, d, N)] = True
        U = np.where(cross, Vm, X)
        U = np.where(U < lb, (X + lb) / 2, np.where(U > ub, (X + ub) / 2, U))
        sel = idx[:m]
        uf = np.full(N, np.inf); uf[sel] = f(U[sel], rng); nfe += m
        s = uf <= fit
        if crmode == 'learn':
            gain = np.where(np.isfinite(uf), np.maximum(fit - uf, 0), 0)
            gh, gl = gain[hic].sum(), gain[~hic].sum()
            if gh + gl > 0:
                pcr = float(np.clip((1 - lr) * pcr + lr * gh / (gh + gl), 0.1, 0.9))
        X[s], fit[s] = U[s], uf[s]
        if act in ('inject', 'all') and control and e > 0 and nfe < budget:
            k = int(min(N // 2, round(kinj * e), budget - nfe))
            if k > 0:
                order = np.argsort(fit); w = order[-k:]
                base = X[order[rng.integers(0, npb, k)]]
                Z = np.clip(base + Dref * diag / np.sqrt(d) * rng.standard_normal((k, d)), lb, ub)
                X[w] = Z; fit[w] = f(Z, rng); nfe += k
        hist_n.append(nfe); hist_b.append(min(hist_b[-1], fit.min()))
    grid = N * (np.arange(T + 1) + 1)
    curve = np.array(hist_b)[np.searchsorted(hist_n, grid, side='right') - 1]
    if return_trace:
        return curve[-1], curve, np.array(tr)
    return curve[-1], curve
