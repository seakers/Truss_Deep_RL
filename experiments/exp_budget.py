"""Round 9, experiment VII(b) -- how much budget does the formulation actually save?

Experiment VII compares final hypervolume at a fixed 64,000 evaluations. That understates
the effect if the smaller formulation gets there much sooner: a smoke run of the
16-variable GA reached 0.754 after 6,000 evaluations, close to what the 72-variable GA
takes 64,000 to reach.

This records the hypervolume trajectory rather than the endpoint, so the question becomes
"how many evaluations does each formulation need to reach a given quality?" -- which is
the number that matters to anyone deciding how to spend a compute budget, and a far more
honest summary than one endpoint each.

Three arms, matched seeds, same settings as track C:
  bars72    -- the repo's formulation, the tuned GA
  groups16  -- the classical 16-group formulation
  sym_mut   -- 72 variables, symmetry-aware mutation moves
"""
import argparse, json, os, sys, time
from copy import deepcopy

import numpy as np
import pygad
from pymoo.indicators.hv import HV

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress
from exp_symmetry import GROUP_OF_BAR, BARS_IN_GROUP, N_GROUPS

MV = np.load(os.path.join(HERE, "max_values.npy"))
MIN_MASK = np.array([True, True, False, True, True])
NFE_BUDGET = 64000
POP = 64
CHECKPOINTS = list(range(1000, NFE_BUDGET + 1, 1000))


def run_arm(arm, seed):
    prob = TrussProblem()
    NV, AL = prob.n_elements, len(prob.area_map)
    MO = MV[:prob.num_objectives]
    ref = np.ones(prob.num_objectives)
    hv_ind = HV(ref_point=ref)

    grouped = (arm == 'groups16')
    n_genes = N_GROUPS if grouped else NV

    def expand(sol):
        s = np.asarray(sol, dtype=int)
        return s[GROUP_OF_BAR] if grouped else s

    all_des, all_obj, all_con = [], [], []
    state = {'nfe': 0}

    def fit(ga_, sol, idx):
        d = [int(x) for x in expand(sol)]
        ov, isc, cv = prob.evaluate(d)
        all_des.append(d); all_obj.append(ov); all_con.append(isc)
        state['nfe'] += 1
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float)
            a[~MIN_MASK] = 0.0
            a[-2:] = cv
            a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]
            return a
        a = np.array(ov + cv, dtype=float)
        a[MIN_MASK] = -a[MIN_MASK]
        return a

    rng = np.random.default_rng(seed + 7000)

    def sym_mutate(offspring, ga_):
        for i, d in enumerate(offspring):
            d = np.asarray(d, dtype=int).copy()
            g = int(rng.integers(0, N_GROUPS))
            d[BARS_IN_GROUP[g]] = int(rng.integers(0, AL))
            offspring[i] = d
        return offspring

    def on_gen(ga_):
        if state['nfe'] >= NFE_BUDGET:
            return "stop"

    ga = pygad.GA(num_generations=100000, num_parents_mating=POP // 2, fitness_func=fit,
                  sol_per_pop=POP, num_genes=n_genes, gene_space=[range(AL)] * n_genes,
                  parent_selection_type="tournament", K_tournament=3,
                  crossover_type="single_point",
                  mutation_type=(sym_mutate if arm == 'sym_mut' else "random"),
                  mutation_num_genes=(None if arm == 'sym_mut' else 1),
                  keep_elitism=8, keep_parents=0, save_solutions=False,
                  on_generation=on_gen, random_seed=seed)
    ga.run()

    all_obj = np.array(all_obj); all_des = np.array(all_des); all_con = np.array(all_con)
    ref_values = MO * ref
    ref_values[~MIN_MASK[:prob.num_objectives]] = 0.0
    all_obj[all_con] = ref_values
    norm = all_obj / MO
    hv_obj = deepcopy(norm)
    for k in range(prob.num_objectives):
        if prob.objective_min_max[k].lower() == 'max':
            hv_obj[:, k] = 1 - norm[:, k]
    pfd, pfo, hvs = track_pareto_progress(hv_obj, all_des, all_con, hv_ind)
    hvs = np.asarray(hvs)
    curve = {str(c): float(hvs[min(c, len(hvs)) - 1]) for c in CHECKPOINTS if c <= len(hvs)}
    return dict(hv=float(hvs[-1]), nfe=state['nfe'], curve=curve)


ARMS = ['bars72', 'groups16', 'sym_mut']
SEEDS = [0, 1, 2, 3]
# Extra seeds, run separately and merged. runner25 (using the repo's own
# run_genetic_algorithm) measured groups16 - bars72 = +0.0334, 3/3, p=0.009, while this
# harness measured -0.0007, 2/4, p=0.97 on the same problem and budget. Two harnesses
# disagreeing on a ~0.03 effect with ~0.015 spread is what four seeds buys you, so the
# disagreement is resolved by adding seeds rather than by preferring the significant one.
EXTRA_SEEDS = [4, 5, 6, 7, 8, 9, 10, 11]
JOBS = [(a, s) for a in ARMS for s in SEEDS]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--extra', action='store_true',
                    help='run EXTRA_SEEDS instead, writing to r9bud2_shard_*.jsonl')
    a = ap.parse_args()
    if a.extra:
        JOBS = [(x, sd) for x in ('bars72', 'groups16') for sd in EXTRA_SEEDS]
    out = os.path.join(HERE, "data", f"r9bud{'2' if a.extra else ''}_shard_{a.shard}.jsonl")
    for j, (arm, seed) in enumerate(JOBS):
        if j % a.nshards != a.shard:
            continue
        t0 = time.time()
        rec = run_arm(arm, seed)
        rec.update(job_id=j, name=arm, seed=seed, secs=time.time() - t0)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{j}] {arm} seed={seed} hv={rec['hv']:.4f} {rec['secs']:.0f}s", flush=True)
