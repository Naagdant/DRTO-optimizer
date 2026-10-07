"""Statistics for the DRTO study. Reads results/*_err.csv, writes results/stats.json.
Errors are f(x_best) - f*. Two errors closer than tol are treated as equal."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from benchmarks import FUNCS

MAIN = ['DRTO', 'PSO', 'GA', 'DE', 'GWO', 'WOA', 'SCA']
RIV = MAIN[1:]
CEC_FSTAR = {i: v for i, v in zip(range(1, 13), [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700])}


def tol(p):
    if p.startswith('C22'):
        return 1e-8 * CEC_FSTAR[int(p.split('F')[1])]
    k = p.split('-')[-1]
    return 1e-8 * max(1.0, abs(FUNCS[k][5]))


def load(s):
    try:
        return pd.read_csv(f'results/{s}_err.csv')
    except FileNotFoundError:
        return None


def vec(df, p, a):
    return df[(df.prob == p) & (df.algo == a)].sort_values('run').err.to_numpy()


def wil(x, y, tl):
    d = x - y; d[np.abs(d) < tl] = 0
    b, w = int((d < 0).sum()), int((d > 0).sum())
    pv = 1.0 if np.all(d == 0) else wilcoxon(d, zero_method='wilcox').pvalue
    sign = '=' if pv >= 0.05 else ('+' if b > w else '-')
    return dict(p=float(pv), sign=sign, better=b, worse=w)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj


def friedman(med, probs):
    M = np.array([[med[p][a] for a in MAIN] for p in probs])
    Mr = np.array([np.round(row / tol(p)) for row, p in zip(M, probs)])
    ranks = np.array([rankdata(r) for r in Mr])
    st = friedmanchisquare(*Mr.T)
    ph = []
    for j, a in enumerate(RIV, start=1):
        d = Mr[:, 0] - Mr[:, j]
        pv = wilcoxon(d[d != 0]).pvalue if np.any(d != 0) else 1.0
        ph.append([a, float(pv), int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())])
    adj = holm(np.array([x[1] for x in ph]))
    return dict(chi2=float(st.statistic), p=float(st.pvalue), n=len(probs),
                ranks={a: float(ranks[:, j].mean()) for j, a in enumerate(MAIN)},
                posthoc=[dict(rival=x[0], p=x[1], p_holm=float(h), better=x[2], ties=x[3], worse=x[4])
                         for x, h in zip(ph, adj)])


out = {'sets': {}}
allmed, allprobs = {}, []
for s in ['classical', 'sr', 'cec']:
    df = load(s)
    if df is None:
        continue
    probs = list(dict.fromkeys(df.prob))
    algs = [a for a in MAIN + ['DRTO-open'] if a in set(df.algo)]
    desc, w, med = {}, {}, {}
    for p in probs:
        med[p] = {}
        for a in algs:
            v = vec(df, p, a)
            desc[f'{p}|{a}'] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), median=float(np.median(v)))
            med[p][a] = float(np.median(v))
        x = vec(df, p, 'DRTO')
        for a in RIV + (['DRTO-open'] if 'DRTO-open' in algs else []):
            w[f'{p}|{a}'] = wil(x, vec(df, p, a), tol(p))
    counts = {a: {k: sum(1 for p in probs if w[f'{p}|{a}']['sign'] == k) for k in '+=-'}
              for a in RIV + (['DRTO-open'] if 'DRTO-open' in algs else [])}
    out['sets'][s] = dict(probs=probs, desc=desc, wil=w, counts=counts, friedman=friedman(med, probs))
    allmed.update(med); allprobs += probs
out['overall'] = friedman(allmed, allprobs)
dfs = load('stress')
if dfs is not None:
    st = {}
    for p in dict.fromkeys(dfs.prob):
        st[p] = dict(closed=float(np.median(vec(dfs, p, 'DRTO'))), open=float(np.median(vec(dfs, p, 'DRTO-open'))),
                     **wil(vec(dfs, p, 'DRTO'), vec(dfs, p, 'DRTO-open'), tol(p)))
    out['stress'] = st
json.dump(out, open('results/stats.json', 'w'), indent=1)

for s, v in out['sets'].items():
    print(s, 'counts', v['counts'])
    print('  ranks', {a: round(r, 2) for a, r in v['friedman']['ranks'].items()}, 'p=%.1e' % v['friedman']['p'])
    print('  holm', [(x['rival'], round(x['p_holm'], 4)) for x in v['friedman']['posthoc']])
print('overall', {a: round(r, 2) for a, r in out['overall']['ranks'].items()},
      [(x['rival'], round(x['p_holm'], 4)) for x in out['overall']['posthoc']])
if 'stress' in out:
    print('stress', ''.join(v['sign'] for v in out['stress'].values()))
