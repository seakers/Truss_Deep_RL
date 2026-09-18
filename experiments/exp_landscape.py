"""Round 9, experiment II -- is the physics rule the operator half-learned actually TRUE?

Round 8's trained operator prefers to edit LOW-utilisation bars (corr(|util|, p) = -0.73
across 3 seeds, p=0.0007). That is the mass-reduction half of fully stressed design, and
it sounds right. But nobody has checked whether it is true FOR THIS PROBLEM.

This measures the mutation landscape directly. Take realistic evolved designs, mutate
one bar at a time across the whole utilisation range, and measure the resulting change
in design quality. If editing low-utilisation bars really is better, dPhi should fall as
|utilisation| rises. If there is no relationship, then the rule the network learned is a
red herring, no amount of RL will make it pay, and the right conclusion is that this
information simply is not useful here.

That is the difference between "the learning is broken" and "there is nothing to learn",
which is the question that decides whether any of this is worth pursuing.
"""
import os, sys, json
import numpy as np, pygad
from scipy import stats

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


def evolve(seed, gens):
    """Get realistic designs -- the operator is deployed on evolved populations,
    not on random ones, so the landscape must be measured where it actually acts."""
    pop = []
    def fit(g_, sol, idx):
        ov, isc, cv = prob.evaluate([int(x) for x in sol])
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float); a[~MIN_MASK] = 0.0
            a[-2:] = cv; a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]; return a
        a = np.array(ov + cv, dtype=float); a[MIN_MASK] = -a[MIN_MASK]; return a
    ga = pygad.GA(num_generations=gens, num_parents_mating=32, fitness_func=fit,
                  sol_per_pop=64, num_genes=NV, gene_space=[range(AL)]*NV,
                  parent_selection_type="tournament", K_tournament=3,
                  crossover_type="single_point", mutation_type="random",
                  mutation_num_genes=1, keep_elitism=8, keep_parents=0,
                  save_solutions=False, random_seed=seed)
    ga.run()
    return [[int(x) for x in s] for s in ga.population]


rows = []
rng = np.random.default_rng(0)
for seed in range(3):
    for gens in (0, 100, 400):                 # random, mid-search, converged
        pop = evolve(seed, gens) if gens else [list(rng.integers(0, AL, NV)) for _ in range(64)]
        for d in pop[:20]:
            o, c, cv, util = prob.evaluate(d, return_utilisation=True)
            phi0 = design_potential(o, cv, MO, MIN_MASK)
            for bar in rng.choice(NV, 24, replace=False):    # sample bars, all utilisations
                d2 = list(d); d2[int(bar)] = int(rng.integers(0, AL))
                o2, c2, cv2 = prob.evaluate(d2)
                rows.append(dict(stage=gens, u=float(abs(util[bar])),
                                 dphi=design_potential(o2, cv2, MO, MIN_MASK) - phi0))

U = np.array([r['u'] for r in rows]); D = np.array([r['dphi'] for r in rows])
print(f"mutations measured: {len(rows)}\n")
print(f"{'stage':>22} {'n':>6} {'corr(|util|, dPhi)':>20} {'p':>10}")
print('-' * 62)
for st, lbl in [(0, 'random designs'), (100, 'after 100 gens'), (400, 'after 400 gens')]:
    m = np.array([r['stage'] == st for r in rows])
    if m.sum() > 10:
        r_, p_ = stats.pearsonr(U[m], D[m])
        print(f"{lbl:>22} {int(m.sum()):>6} {r_:>+20.4f} {p_:>10.2e}")
r_, p_ = stats.pearsonr(U, D)
print(f"{'ALL':>22} {len(U):>6} {r_:>+20.4f} {p_:>10.2e}")

print("\nmean dPhi by utilisation quartile (converged designs only):")
m = np.array([r['stage'] == 400 for r in rows])
u, d = U[m], D[m]
qs = np.quantile(u, [0, .25, .5, .75, 1.0])
for i in range(4):
    sel = (u >= qs[i]) & (u <= qs[i+1])
    print(f"  |util| in [{qs[i]:.3f}, {qs[i+1]:.3f}]  n={sel.sum():>5}  "
          f"mean dPhi = {d[sel].mean():+.5f}   P(improves) = {(d[sel] > 0).mean():.1%}")
print("\nA NEGATIVE correlation means editing low-utilisation bars really is better,")
print("and the rule the operator learned is real. Near zero means it is a red herring.")
json.dump(rows, open(os.path.join(HERE, "data", "r9_landscape.json"), "w"))
