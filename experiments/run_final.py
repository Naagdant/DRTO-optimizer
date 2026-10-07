"""Final experiment (fresh seeds / instances, parameters frozen).
usage: python3 run_final.py <set> where set in classical, sr, cec, stress"""
import sys, json, time, numpy as np
from multiprocessing import Pool
from problems import classical, shifted_rotated, cec2022
from algorithms import ALGOS
from drto import drto

R, N = 30, 30
SET = sys.argv[1]
if SET == 'classical':
    PROBS = [('C', f'F{i}', i - 1) for i in range(1, 24)]; T = 500
elif SET == 'sr':
    PROBS = [('S', k, 100 + j) for j, k in enumerate(['F1', 'F2', 'F3', 'F4', 'F9', 'F10', 'F11'])]; T = 500
elif SET == 'cec':
    PROBS = [('X', i, 200 + i) for i in range(1, 13)]; T = 20000 // N - 1
elif SET == 'stress':
    PROBS = [('C', f'F{i}', i - 1) for i in range(1, 14)] + \
            [('S', k, 100 + j) for j, k in enumerate(['F1', 'F2', 'F3', 'F4', 'F9', 'F10', 'F11'])]; T = 500

MAIN = ['DRTO', 'PSO', 'GA', 'DE', 'GWO', 'WOA', 'SCA']
ALGS = (['DRTO-open'] + MAIN[:1]) if SET == 'stress' else (MAIN + (['DRTO-open'] if SET != 'cec' else []))
OLD = ['PSO', 'GA', 'DE', 'GWO', 'WOA', 'SCA']


def get(p):
    k, key, _ = p
    if k == 'C': return classical(key)
    if k == 'S': return shifted_rotated(key, seed=2027)
    return cec2022(key, 10)


def fn_for(alg):
    if alg in ALGOS: return ALGOS[alg]
    kw = {'DRTO': {}, 'DRTO-open': {'control': False}}[alg]
    if SET == 'stress': kw = dict(kw, g=0.1, Dend=1e-10)
    return lambda *z: drto(*z, **kw)


def job(a):
    p, alg = a
    f, lb, ub, d, fs, name = get(p)
    fi = p[2]
    ai = OLD.index(alg) if alg in OLD else 50 + ['DRTO', 'DRTO-open'].index(alg)
    fin = np.empty(R); cur = []
    t0 = time.time()
    for r in range(R):
        X0 = lb + np.random.default_rng([fi, r]).random((N, d)) * (ub - lb)
        rng = np.random.default_rng([fi, r, 1000 + ai])
        v, c = fn_for(alg)(f, lb, ub, X0, T, rng)
        fin[r] = v - fs; cur.append(c - fs)
    cur = np.array(cur)
    if cur.shape[1] > 501:   # store 501 points for long runs
        cur = cur[:, np.linspace(0, cur.shape[1] - 1, 501).round().astype(int)]
    return name, alg, fin, cur, time.time() - t0


if __name__ == '__main__':
    import os
    part = f'results/{SET}_parts'; os.makedirs(part, exist_ok=True)
    pname = lambda p, a: f"{part}/{get(p)[5]}__{a}.npz"
    tasks = [(p, a) for p in PROBS for a in ALGS if not os.path.exists(pname(p, a))]
    with Pool(2) as pool:
        for name, alg, fin, cur, dt in pool.imap_unordered(job, tasks):
            np.savez(f'{part}/{name}__{alg}.npz', fin=fin, cur=cur)
            print(name, alg, f'{np.median(fin):.3e}', f'{dt:.0f}s', flush=True)
    rows, curves = [], {}
    for p in PROBS:
        for a in ALGS:
            z = np.load(pname(p, a)); name = get(p)[5]
            rows += [f'{name},{a},{r},{float(v)!r}' for r, v in enumerate(z['fin'])]
            curves[f'{name}|{a}'] = z['cur']
    open(f'results/{SET}_err.csv', 'w').write('prob,algo,run,err\n' + '\n'.join(rows) + '\n')
    np.savez_compressed(f'results/{SET}_curves.npz', **curves)
