"""Tables and quoted numbers for the paper, all from results/stats.json -> tables.json"""
import json
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from analysis import tol as TOL
S = json.load(open('results/stats.json'))
AK = ['DRTO', 'PSO', 'GA', 'DE', 'GWO', 'WOA', 'SCA']
RIV = AK[1:]


def e(x):
    if x == 0: return '0'
    m, ex = f'{x:.2E}'.split('E'); return f'{m}E{int(ex):+03d}'.replace('-', '−')


def tol(p):
    return TOL(p)


def fp(x):
    return '<0.0001' if x < 1e-4 else f'{x:.4f}'


T = {}
c = S['sets']['classical']
rows = []
for p in c['probs']:
    means = [c['desc'][f'{p}|{a}']['mean'] for a in AK]
    mn = min(means)
    rows.append([p, 'Mean'] + [{'t': e(m), 'b': m <= mn + tol(p)} for m in means])
    rows.append(['', 'Std'] + [{'t': e(c['desc'][f'{p}|{a}']['std']), 'b': False} for a in AK])
T['classical'] = rows

rows = []
for s in ['sr', 'cec']:
    st = S['sets'][s]
    for p in st['probs']:
        meds = [st['desc'][f'{p}|{a}']['median'] for a in AK]; mn = min(meds)
        r = [p.replace('C22-', 'C')]
        for a, m in zip(AK, meds):
            sg = '' if a == 'DRTO' else ' ' + {'+': '+', '=': '=', '-': '−'}[st['wil'][f'{p}|{a}']['sign']]
            r.append({'t': e(m) + sg, 'b': m <= mn + tol(p)})
        rows.append(r)
T['srcec'] = rows

summ = []
for s, lab in [('classical', 'Classical (23)'), ('sr', 'Shifted-rotated (7)'), ('cec', 'CEC 2022 (12)')]:
    st = S['sets'][s]
    summ.append([lab, f"{st['friedman']['ranks']['DRTO']:.2f}"] +
                [f"{st['counts'][a]['+']}/{st['counts'][a]['=']}/{st['counts'][a]['-']}" for a in RIV])
tot = [sum(S['sets'][s]['counts'][a][k] for s in S['sets']) for a in RIV for k in '+=-']
summ.append(['All (42)', f"{S['overall']['ranks']['DRTO']:.2f}"] +
            [f'{tot[3*i]}/{tot[3*i+1]}/{tot[3*i+2]}' for i in range(6)])
T['summary'] = summ
ph = {x['rival']: x for x in S['overall']['posthoc']}
T['ranks'] = [[a] + [f"{S['sets'][s]['friedman']['ranks'][a]:.2f}" for s in ['classical', 'sr', 'cec']] +
              [f"{S['overall']['ranks'][a]:.2f}", '–' if a == 'DRTO' else fp(ph[a]['p_holm'])] for a in AK]

N = {}
for s in ['classical', 'sr', 'cec']:
    fr = S['sets'][s]['friedman']
    N[f'{s}_chi'] = f"{fr['chi2']:.2f}"; N[f'{s}_p'] = e(fr['p'])
    for a in AK: N[f'{s}_r_{a}'] = f"{fr['ranks'][a]:.2f}"
    for x in fr['posthoc']: N[f"{s}_h_{x['rival']}"] = f"{x['p_holm']:.3f}"
    for a in RIV:
        for k, nm in zip('+=-', ['w', 't', 'l']): N[f'{s}_{nm}_{a}'] = str(S['sets'][s]['counts'][a][k])
    if 'DRTO-open' in S['sets'][s]['counts']:
        for k, nm in zip('+=-', ['w', 't', 'l']): N[f'{s}_{nm}_open'] = str(S['sets'][s]['counts']['DRTO-open'][k])
N['all_chi'] = f"{S['overall']['chi2']:.2f}"; N['all_p'] = e(S['overall']['p'])
for a in AK: N[f'all_r_{a}'] = f"{S['overall']['ranks'][a]:.2f}"
for a in RIV:
    N[f'all_h_{a}'] = fp(ph[a]['p_holm'])
    for i, nm in enumerate(['w', 't', 'l']): N[f'all_{nm}_{a}'] = str(tot[3 * RIV.index(a) + i])
st = S['stress']; sg = [v['sign'] for v in st.values()]
N['stress_w'] = str(sg.count('+')); N['stress_t'] = str(sg.count('=')); N['stress_l'] = str(sg.count('-')); N['stress_n'] = str(len(sg))
N['stress_F1_c'] = e(st['F1']['closed']); N['stress_F1_o'] = e(st['F1']['open'])
N['stress_SRF1_c'] = e(st['SR-F1']['closed']); N['stress_SRF1_o'] = e(st['SR-F1']['open'])
D = lambda s, p, a: e(S['sets'][s]['desc'][f'{p}|{a}']['median'])
for s, p, a in [('classical', 'F1', 'DRTO'), ('classical', 'F1', 'GWO'), ('classical', 'F1', 'WOA'), ('classical', 'F9', 'DRTO'),
                ('classical', 'F9', 'WOA'), ('classical', 'F9', 'GA'), ('classical', 'F8', 'DRTO'), ('classical', 'F8', 'WOA'),
                ('classical', 'F5', 'WOA'), ('classical', 'F5', 'DRTO'), ('classical', 'F13', 'DRTO'), ('classical', 'F13', 'WOA'),
                ('sr', 'SR-F1', 'DRTO'), ('sr', 'SR-F1', 'GWO'), ('sr', 'SR-F1', 'WOA'), ('sr', 'SR-F10', 'DRTO'), ('sr', 'SR-F10', 'WOA'),
                ('sr', 'SR-F3', 'DE'), ('sr', 'SR-F3', 'DRTO'), ('cec', 'C22-F6', 'DRTO'), ('cec', 'C22-F6', 'DE'),
                ('cec', 'C22-F8', 'DRTO'), ('cec', 'C22-F8', 'DE'), ('cec', 'C22-F9', 'DRTO'), ('cec', 'C22-F9', 'DE')]:
    N[f'm_{p}_{a}'.replace('-', '')] = D(s, p, a)
import pandas as pd
d = pd.read_csv('results/cec_err.csv'); g = d[d.prob == 'C22-F9']
for a in ['DRTO', 'DE', 'PSO', 'GA']: N[f'f9hit_{a}'] = str(int((g[g.algo == a].err < 1e-2).sum()))
T['N'] = N
json.dump(T, open('tables.json', 'w'), ensure_ascii=False, indent=0)
print(len(N), 'numbers')
