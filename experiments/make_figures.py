"""Figures for the DRTO paper, generated from results/."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from problems import classical, shifted_rotated
from drto import drto

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8, 'savefig.dpi': 400,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'legend.frameon': False,
                     'mathtext.fontset': 'dejavuserif'})
W1 = 3.42
AK = ['DRTO', 'PSO', 'GA', 'DE', 'GWO', 'WOA', 'SCA']
COL = {'DRTO': '#c0392b', 'PSO': '#e67e22', 'GA': '#7f8c8d', 'DE': '#2e86c1', 'GWO': '#27ae60', 'WOA': '#8e44ad', 'SCA': '#b7950b'}
MK = {'DRTO': 'o', 'PSO': 'P', 'GA': 's', 'DE': '^', 'GWO': 'D', 'WOA': 'v', 'SCA': 'X'}
S = json.load(open('results/stats.json'))
FLOOR = 1e-16


def logticks(ax, lo, hi, n=5):
    lo, hi = int(np.floor(lo)), int(np.ceil(hi)); step = max(1, int(np.ceil((hi - lo) / n)))
    lo = hi - step * int(np.ceil((hi - lo) / step)); e = np.arange(lo, hi + 1, step)
    ax.set_yticks(10.0 ** e); ax.set_yticklabels([f'$10^{{{k}}}$' for k in e])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_ylim(10.0 ** lo, 10.0 ** hi)


# ---- Fig. 1 block diagram
fig, ax = plt.subplots(figsize=(W1, 1.5)); ax.set_xlim(-0.15, 10.05); ax.set_ylim(0, 4.0); ax.axis('off')
def box(x, y, w, h, t):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc='#f4f4f4', ec='k', lw=0.6))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=5.6)
def arr(x1, y1, x2, y2, t=None, dy=0.2):
    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=0.6))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, t, ha='center', fontsize=6.5)
box(0.0, 2.3, 1.55, 1.0, 'Reference\n$D_{ref}(\\tau)$')
ax.add_patch(plt.Circle((2.2, 2.8), 0.22, fc='w', ec='k', lw=0.6)); ax.text(2.2, 2.8, '$\\Sigma$', ha='center', va='center', fontsize=7)
box(2.95, 2.3, 1.55, 1.0, 'PI\ncontroller')
box(5.2, 2.3, 1.95, 1.0, 'Trial generation\nnoise scale $\\sigma$')
box(7.85, 2.3, 2.05, 1.0, 'Population and\ngreedy selection')
box(5.2, 0.3, 1.95, 0.9, 'Diversity\nsensor $D$')
arr(1.6, 2.8, 1.98, 2.8); ax.text(1.75, 3.0, '+', fontsize=7)
arr(2.42, 2.8, 2.9, 2.8, '$e$'); arr(4.55, 2.8, 5.15, 2.8, '$u$'); arr(7.2, 2.8, 7.8, 2.8)
ax.plot([8.88, 8.88, 7.2], [2.25, 0.75, 0.75], 'k', lw=0.6); ax.annotate('', (7.2, 0.75), (7.25, 0.75), arrowprops=dict(arrowstyle='->', lw=0.6))
ax.plot([5.15, 2.2], [0.75, 0.75], 'k', lw=0.6); ax.annotate('', (2.2, 2.58), (2.2, 0.75), arrowprops=dict(arrowstyle='->', lw=0.6))
ax.text(2.3, 2.2, '−', fontsize=8)
fig.tight_layout(pad=0.1); fig.savefig('figs/fig1_loop.png'); plt.close(fig)

# ---- Fig. 3 convergence curves (median error)
sel = [('classical', 'F12'), ('classical', 'F21'), ('sr', 'SR-F1'), ('sr', 'SR-F10'), ('cec', 'C22-F4'), ('cec', 'C22-F10')]
fig, axs = plt.subplots(3, 2, figsize=(W1, 4.3))
for ax, (s, p) in zip(axs.ravel(), sel):
    C = np.load(f'results/{s}_curves.npz')
    lo, hi = 0, -99
    for a in AK:
        c = np.maximum(np.median(C[f'{p}|{a}'], axis=0), FLOOR); x = np.linspace(0, 1, c.size)
        ax.semilogy(x, c, color=COL[a], lw=1.4 if a == 'DRTO' else 0.8, label=a, zorder=5 if a == 'DRTO' else 2)
        lo = min(lo, np.log10(c.min())); hi = max(hi, np.log10(c.max()))
    logticks(ax, lo, hi, 4); ax.set_title(p, fontsize=8); ax.grid(alpha=0.25, lw=0.4)
for ax in axs[2]: ax.set_xlabel('Budget fraction')
for ax in axs[:, 0]: ax.set_ylabel('Median error')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=7, loc='upper center', fontsize=6, handlelength=1.0, columnspacing=0.5)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.3, w_pad=0.3); fig.savefig('figs/fig3_convergence.png'); plt.close(fig)

# ---- Fig. 4 pairwise Wilcoxon outcomes over all problems
riv = AK[1:]
cnt = {k: [sum(S['sets'][s]['counts'][r][k] for s in S['sets']) for r in riv] for k in '+=-'}
fig, ax = plt.subplots(figsize=(W1, 1.6)); left = np.zeros(len(riv)); tot = sum(cnt[k][0] for k in '+=-')
for k, c, lab in [('+', '#c0392b', 'DRTO better'), ('=', '#bdc3c7', 'No significant difference'), ('-', '#34495e', 'DRTO worse')]:
    ax.barh(range(len(riv)), cnt[k], left=left, color=c, height=0.6, label=lab)
    for i, v in enumerate(cnt[k]):
        if v: ax.text(left[i] + v / 2, i, str(v), ha='center', va='center', fontsize=6.5, color='k' if k == '=' else 'w')
    left += cnt[k]
ax.set_yticks(range(len(riv))); ax.set_yticklabels([f'vs {r}' for r in riv]); ax.invert_yaxis()
ax.set_xlim(0, tot); ax.set_xlabel(f'Number of problems (of {tot})')
ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.45, 1.0), fontsize=6, handlelength=1.0, columnspacing=0.6)
fig.tight_layout(pad=0.2); fig.savefig('figs/fig4_wilcoxon.png'); plt.close(fig)

# ---- Fig. 5 Friedman ranks per suite
groups = [(S['sets']['classical']['friedman'], 'Classical\n(23)'), (S['sets']['sr']['friedman'], 'Shifted-\nrotated (7)'),
          (S['sets']['cec']['friedman'], 'CEC 2022\n(12)'), (S['overall'], 'All\n(42)')]
fig, ax = plt.subplots(figsize=(W1, 2.1)); off = np.linspace(-0.3, 0.3, len(AK))
for j, a in enumerate(AK):
    ax.plot(np.arange(4) + off[j], [g[0]['ranks'][a] for g in groups], ls='none', marker=MK[a], color=COL[a],
            ms=4.5 if a == 'DRTO' else 3.5, mec='k' if a == 'DRTO' else COL[a], mew=0.5, label=a)
ax.set_xticks(range(4)); ax.set_xticklabels([g[1] for g in groups]); ax.set_ylim(0.5, 7.5); ax.set_yticks(range(1, 8))
ax.set_ylabel('Average rank (lower is better)'); ax.grid(axis='y', alpha=0.25, lw=0.4)
for x in [0.5, 1.5, 2.5]: ax.axvline(x, color='0.8', lw=0.5)
ax.legend(ncol=7, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=6, handletextpad=0.1, columnspacing=0.5)
fig.tight_layout(pad=0.2); fig.savefig('figs/fig5_ranks.png'); plt.close(fig)

# ---- Fig. 6 stress ablation
st = S['stress']; ps = list(st)
fig, ax = plt.subplots(figsize=(W1, 1.9)); x = np.arange(len(ps))
co = np.maximum([st[p]['closed'] for p in ps], FLOOR); op = np.maximum([st[p]['open'] for p in ps], FLOOR)
ax.vlines(x, np.minimum(co, op), np.maximum(co, op), color='0.75', lw=0.8)
ax.plot(x, op, 'o', mfc='w', mec='0.3', ms=3.5, label='Open loop'); ax.plot(x, co, 's', color=COL['DRTO'], ms=3.2, label='Closed loop')
ax.set_yscale('log'); logticks(ax, np.log10(min(co.min(), op.min())), np.log10(max(co.max(), op.max())), 5)
ax.set_xticks(x); ax.set_xticklabels(ps, rotation=90, fontsize=6); ax.set_ylabel('Median error')
ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0)); ax.grid(axis='y', alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.2); fig.savefig('figs/fig6_stress.png'); plt.close(fig)
print('ok')
