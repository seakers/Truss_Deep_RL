"""Round 9, experiment IV -- WHY does training flatten the pointer? A variance decomposition.

Rounds 5-8 established the fact: across every actor measured, the pointer head ("which
bar do I edit?") collapses to near-uniform (KL from uniform 0.0000-0.0114), and training
pushes it FURTHER toward uniform, not away. Round 8's stress channel changed which bars it
mildly prefers -- physics-aligned, corr(|util|, p) = -0.73 -- without stopping the
flattening. No experiment so far explains the mechanism.

Here is a candidate mechanism that is testable without touching the network at all.

The action is factored into two heads: WHICH bar (72 options) and WHAT area (50 options).
Both heads receive the same scalar advantage. If the outcome of an edit is governed almost
entirely by the VALUE chosen and barely at all by WHICH bar it was applied to, then the
pointer's gradient is mostly the value head's noise. Under advantage normalisation that
noise has no consistent direction, so the pointer random-walks toward the maximum-entropy
solution -- uniform. The flattening would then not be a bug in the optimiser or the reward;
it would be the correct response to an action space where one head genuinely does not
matter much.

So: measure it. For realistic designs, evaluate a full bar x value grid and decompose the
variance of dPhi into

  - bar main effect     : how much does WHICH bar you touch matter, averaged over values
  - value main effect   : how much does WHAT value you set matter, averaged over bars
  - interaction+residual: the part only explained by the specific combination

If the bar main effect is a few percent of total variance, the pointer is being asked to
learn a distinction that carries almost no return signal, and every "fix the RL" idea in
the backlog is aimed at the wrong thing. If the bar effect is large, the flattening IS a
learning failure and is worth chasing.

This is a claim about the PROBLEM's action space, measured directly, not an inference from
training curves.
"""
import os, sys, json
import numpy as np
import pygad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))

from evaluation.truss_problem import TrussProblem
from optimization.design_repair import design_potential

MV = np.load(os.path.join(HERE, "max_values.npy"))
MO = MV[:3]
MIN_MASK = np.array([True, True, False, True, True])
prob = TrussProblem()
NV, AL = prob.n_elements, len(prob.area_map)

N_BARS, N_VALS, N_DESIGNS = 12, 10, 12
VAL_GRID = np.linspace(0, AL - 1, N_VALS).astype(int)


def evolve(seed, gens):
    def fit(g_, sol, idx):
        ov, isc, cv = prob.evaluate([int(x) for x in sol])
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float); a[~MIN_MASK] = 0.0
            a[-2:] = cv; a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]; return a
        a = np.array(ov + cv, dtype=float); a[MIN_MASK] = -a[MIN_MASK]; return a
    ga = pygad.GA(num_generations=gens, num_parents_mating=32, fitness_func=fit,
                  sol_per_pop=64, num_genes=NV, gene_space=[range(AL)] * NV,
                  parent_selection_type="tournament", K_tournament=3,
                  crossover_type="single_point", mutation_type="random",
                  mutation_num_genes=1, keep_elitism=8, keep_parents=0,
                  save_solutions=False, random_seed=seed)
    ga.run()
    return [[int(x) for x in s] for s in ga.population]


def decompose(G):
    """Two-way ANOVA on the bar x value grid of a single design.

    G[b, v] = dPhi from setting bar b to value v. Returns the fraction of total
    variance attributable to the bar main effect, the value main effect, and the
    remainder (interaction). Standard sum-of-squares decomposition: the effects are
    orthogonal by construction on a complete balanced grid, so the three shares sum to 1.
    """
    gm = G.mean()
    bar_eff = G.mean(axis=1) - gm
    val_eff = G.mean(axis=0) - gm
    ss_tot = ((G - gm) ** 2).sum()
    if ss_tot <= 0:
        return np.nan, np.nan, np.nan
    ss_bar = G.shape[1] * (bar_eff ** 2).sum()
    ss_val = G.shape[0] * (val_eff ** 2).sum()
    return ss_bar / ss_tot, ss_val / ss_tot, 1.0 - (ss_bar + ss_val) / ss_tot


rng = np.random.default_rng(0)
rows = []
for seed in range(3):
    for gens in (0, 100, 400):
        pop = evolve(seed, gens) if gens else [list(rng.integers(0, AL, NV)) for _ in range(64)]
        for d in pop[:N_DESIGNS]:
            o, c, cv, util = prob.evaluate(d, return_utilisation=True)
            phi0 = design_potential(o, cv, MO, MIN_MASK)
            bars = rng.choice(NV, N_BARS, replace=False)
            G = np.zeros((N_BARS, N_VALS))
            for bi, b in enumerate(bars):
                for vi, v in enumerate(VAL_GRID):
                    d2 = list(d); d2[int(b)] = int(v)
                    o2, _, cv2 = prob.evaluate(d2)
                    G[bi, vi] = design_potential(o2, cv2, MO, MIN_MASK) - phi0
            fb, fv, fi = decompose(G)
            rows.append(dict(seed=seed, stage=gens, f_bar=float(fb),
                             f_val=float(fv), f_int=float(fi),
                             best_bar_gap=float(G.mean(axis=1).max() - G.mean(axis=1).min()),
                             best_val_gap=float(G.mean(axis=0).max() - G.mean(axis=0).min())))
            print('.', end='', flush=True)

print(f"\n\nvariance decomposition of single-bar mutations  "
      f"({len(rows)} designs x {N_BARS} bars x {N_VALS} values)\n")
print(f"{'stage':>18} {'n':>4} {'bar effect':>12} {'value effect':>14} {'interaction':>13}")
print('-' * 66)
for st, lbl in [(0, 'random designs'), (100, 'after 100 gens'), (400, 'after 400 gens')]:
    sel = [r for r in rows if r['stage'] == st]
    if sel:
        print(f"{lbl:>18} {len(sel):>4} "
              f"{np.nanmean([r['f_bar'] for r in sel]):>11.1%} "
              f"{np.nanmean([r['f_val'] for r in sel]):>13.1%} "
              f"{np.nanmean([r['f_int'] for r in sel]):>12.1%}")
print(f"{'ALL':>18} {len(rows):>4} "
      f"{np.nanmean([r['f_bar'] for r in rows]):>11.1%} "
      f"{np.nanmean([r['f_val'] for r in rows]):>13.1%} "
      f"{np.nanmean([r['f_int'] for r in rows]):>12.1%}")

print("\nmean spread in dPhi from choosing the best vs worst option, per head:")
for st, lbl in [(0, 'random designs'), (100, 'after 100 gens'), (400, 'after 400 gens')]:
    sel = [r for r in rows if r['stage'] == st]
    if sel:
        print(f"  {lbl:>16}: bar {np.mean([r['best_bar_gap'] for r in sel]):.5f}"
              f"   value {np.mean([r['best_val_gap'] for r in sel]):.5f}")

print("\nA SMALL bar effect means the pointer is being trained on a distinction that")
print("barely moves the return, and flattening is the correct answer, not a bug.")
json.dump(rows, open(os.path.join(HERE, "data", "r9_variance.json"), "w"))
