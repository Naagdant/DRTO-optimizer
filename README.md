# DRTO: Diversity Reference Tracking Optimizer

Python implementation of **DRTO**, a population-based metaheuristic in which a
proportional-integral (PI) controller makes the population diversity follow a
prescribed reference trajectory, together with all experiments and raw results
of the paper:

> K. D. Bodha, V. Arun and A. Awasthi, "Diversity Reference Tracking Optimizer: A
> Population-Based Metaheuristic with Closed-Loop PI Control of Search Diversity," 2026.

## How DRTO works

1. Measure the population diversity *D* (mean distance to the centroid, normalised by the box diagonal).
2. Compare log *D* with a reference that decays geometrically from the initial diversity to `Dend` over the evaluation budget.
3. A PI controller turns the tracking error into a multiplier *u* for the perturbation scale σ = g·D_ref·u·‖u−l‖/√d.
4. Trial vectors: current-to-pbest difference move + σ·N(0, I), binomial crossover with CR ∈ {0.1, 0.9} chosen by an online success estimate, greedy selection.

## Quick start

```bash
pip install -r requirements.txt
python example.py
```

```python
from drto import drto
best, curve = drto(f, lb, ub, X0, T, rng)   # f maps an (N, d) array to (N,)
```

Budget: `N * (T + 1)` function evaluations, where `N = X0.shape[0]`.
Default parameters (as in the paper): `kp=0.2, ki=0.05, g=0.3, Dend=1e-8, Fn=0.5, p=0.2, lr=0.1`.

## Reproducing the paper

Run from the repository root:

```bash
export PYTHONPATH=.
python experiments/run_final.py classical   # 23 classical functions
python experiments/run_final.py sr          # 7 shifted-rotated functions
python experiments/run_final.py stress      # open- vs closed-loop ablation
python experiments/run_final.py cec         # CEC 2022, D = 10 (slow)
python experiments/analysis.py              # statistics -> results/stats.json
```

`results/` already contains the raw final errors of all 30 runs for every
algorithm and problem (`*_err.csv`) and the computed statistics (`stats.json`),
so every number in the paper can be checked without rerunning.

| File | Content |
|---|---|
| `drto/drto.py` | the algorithm |
| `experiments/benchmarks.py` | the 23 classical benchmark functions |
| `experiments/problems.py` | shifted-rotated and CEC 2022 problem wrappers |
| `experiments/algorithms.py` | PSO, GA, DE, GWO, WOA and SCA used for comparison |
| `experiments/run_final.py` | experiment runner (30 paired runs, shared initial populations) |
| `experiments/analysis.py` | Wilcoxon, Friedman and Holm tests |

## Citation

If you use this code, please cite the paper above.

## License

MIT
