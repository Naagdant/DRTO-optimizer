"""Six population-based optimizers with an identical budget:
N agents, one initial evaluation of the shared population X0, then T iterations
of N evaluations each. Every function returns (best value, best-so-far curve of
length T+1). Bounds are enforced by clipping in every algorithm."""
import numpy as np


def _clip(X, lb, ub):
    return np.clip(X, lb, ub)


def pso(f, lb, ub, X0, T, rng, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, vfrac=0.2):
    """Inertia-weight PSO (Shi & Eberhart 1998), linearly decreasing w."""
    N, d = X0.shape
    vmax = vfrac * (ub - lb)
    X = X0.copy(); V = np.zeros_like(X)
    fit = f(X, rng)
    P, pf = X.copy(), fit.copy()
    g = np.argmin(pf); G, gf = P[g].copy(), pf[g]
    curve = np.empty(T + 1); curve[0] = gf
    for t in range(T):
        w = w_max - (w_max - w_min) * t / T
        r1, r2 = rng.random((N, d)), rng.random((N, d))
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
        V = np.clip(V, -vmax, vmax)
        X = _clip(X + V, lb, ub)
        fit = f(X, rng)
        imp = fit < pf
        P[imp], pf[imp] = X[imp], fit[imp]
        g = np.argmin(pf)
        if pf[g] < gf:
            G, gf = P[g].copy(), pf[g]
        curve[t + 1] = gf
    return gf, curve


def ga(f, lb, ub, X0, T, rng, pc=0.9, eta_c=20.0, eta_m=20.0, n_elite=2):
    """Real-coded GA: binary tournament, SBX crossover, polynomial mutation
    (pm = 1/d), elitism of the two best individuals."""
    N, d = X0.shape
    pm = 1.0 / d
    X = X0.copy(); fit = f(X, rng)
    b = np.argmin(fit); best, bf = X[b].copy(), fit[b]
    curve = np.empty(T + 1); curve[0] = bf
    span = ub - lb
    for t in range(T):
        a, c = rng.integers(0, N, N), rng.integers(0, N, N)
        par = np.where((fit[a] <= fit[c])[:, None], X[a], X[c])
        p1, p2 = par[0::2], par[1::2]
        u = rng.random(p1.shape)
        beta = np.where(u <= 0.5, (2 * u)**(1 / (eta_c + 1)), (1 / (2 * (1 - u)))**(1 / (eta_c + 1)))
        do = (rng.random(p1.shape[0]) < pc)[:, None] & (rng.random(p1.shape) < 0.5)
        c1 = np.where(do, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), p1)
        c2 = np.where(do, 0.5 * ((1 - beta) * p1 + (1 + beta) * p2), p2)
        Y = np.vstack([c1, c2])
        u = rng.random(Y.shape)
        dq = np.where(u < 0.5, (2 * u)**(1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u))**(1 / (eta_m + 1)))
        mut = rng.random(Y.shape) < pm
        Y = _clip(Y + mut * dq * span, lb, ub)
        yf = f(Y, rng)
        # elitism: the n_elite best parents replace the worst offspring
        e = np.argsort(fit)[:n_elite]; wst = np.argsort(yf)[-n_elite:]
        Y[wst], yf[wst] = X[e], fit[e]
        X, fit = Y, yf
        b = np.argmin(fit)
        if fit[b] < bf:
            best, bf = X[b].copy(), fit[b]
        curve[t + 1] = bf
    return bf, curve


def de(f, lb, ub, X0, T, rng, F=0.5, CR=0.9):
    """DE/rand/1/bin (Storn & Price 1997)."""
    N, d = X0.shape
    X = X0.copy(); fit = f(X, rng)
    curve = np.empty(T + 1); curve[0] = fit.min()
    idx = np.arange(N)
    for t in range(T):
        r = np.array([rng.choice(np.delete(idx, i), 3, replace=False) for i in idx])
        Vm = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
        cross = rng.random((N, d)) < CR
        cross[idx, rng.integers(0, d, N)] = True
        U = _clip(np.where(cross, Vm, X), lb, ub)
        uf = f(U, rng)
        s = uf <= fit
        X[s], fit[s] = U[s], uf[s]
        curve[t + 1] = min(curve[t], fit.min())
    return curve[-1], curve


def gwo(f, lb, ub, X0, T, rng):
    """Grey Wolf Optimizer (Mirjalili et al. 2014)."""
    N, d = X0.shape
    X = X0.copy(); fit = f(X, rng)
    o = np.argsort(fit)[:3]
    L = X[o].copy(); Lf = fit[o].copy()  # alpha, beta, delta
    curve = np.empty(T + 1); curve[0] = Lf[0]
    for t in range(T):
        a = 2 - 2 * t / T
        Xn = np.zeros_like(X)
        for k in range(3):
            A = 2 * a * rng.random((N, d)) - a
            C = 2 * rng.random((N, d))
            Xn += L[k] - A * np.abs(C * L[k] - X)
        X = _clip(Xn / 3, lb, ub)
        fit = f(X, rng)
        allX = np.vstack([L, X]); allf = np.concatenate([Lf, fit])
        o = np.argsort(allf)
        # keep three distinct leaders (stable sort keeps old leader on ties)
        L, Lf = allX[o[:3]].copy(), allf[o[:3]].copy()
        curve[t + 1] = Lf[0]
    return Lf[0], curve


def woa(f, lb, ub, X0, T, rng, b=1.0):
    """Whale Optimization Algorithm (Mirjalili & Lewis 2016)."""
    N, d = X0.shape
    X = X0.copy(); fit = f(X, rng)
    g = np.argmin(fit); G, gf = X[g].copy(), fit[g]
    curve = np.empty(T + 1); curve[0] = gf
    for t in range(T):
        a = 2 - 2 * t / T
        a2 = -1 - t / T
        Xn = np.empty_like(X)
        for i in range(N):
            A = 2 * a * rng.random() - a
            C = 2 * rng.random()
            p = rng.random()
            l = (a2 - 1) * rng.random() + 1
            if p < 0.5:
                if abs(A) >= 1:
                    Xr = X[rng.integers(N)]
                    Xn[i] = Xr - A * np.abs(C * Xr - X[i])
                else:
                    Xn[i] = G - A * np.abs(C * G - X[i])
            else:
                Xn[i] = np.abs(G - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + G
        X = _clip(Xn, lb, ub)
        fit = f(X, rng)
        g = np.argmin(fit)
        if fit[g] < gf:
            G, gf = X[g].copy(), fit[g]
        curve[t + 1] = gf
    return gf, curve


def sca(f, lb, ub, X0, T, rng, a=2.0):
    """Sine Cosine Algorithm (Mirjalili 2016)."""
    N, d = X0.shape
    X = X0.copy(); fit = f(X, rng)
    g = np.argmin(fit); G, gf = X[g].copy(), fit[g]
    curve = np.empty(T + 1); curve[0] = gf
    for t in range(T):
        r1 = a - t * a / T
        r2 = 2 * np.pi * rng.random((N, d))
        r3 = 2 * rng.random((N, d))
        r4 = rng.random((N, d))
        step = np.abs(r3 * G - X)
        X = np.where(r4 < 0.5, X + r1 * np.sin(r2) * step, X + r1 * np.cos(r2) * step)
        X = _clip(X, lb, ub)
        fit = f(X, rng)
        g = np.argmin(fit)
        if fit[g] < gf:
            G, gf = X[g].copy(), fit[g]
        curve[t + 1] = gf
    return gf, curve


ALGOS = {'PSO': pso, 'GA': ga, 'DE': de, 'GWO': gwo, 'WOA': woa, 'SCA': sca}
