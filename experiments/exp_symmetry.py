"""Round 9, experiment VII -- the design variables themselves.

Every method in this repo optimises 72 independent bar areas. That is not how this
benchmark is posed in the structural optimisation literature. The 72-bar space truss is a
standard test problem and it is solved with SIXTEEN design variables, not 72: the
structure has four identical levels, and within each level the four legs are
interchangeable, the eight diagonals are interchangeable, the four top chords are
interchangeable, and the two cross braces are interchangeable. Grouping them is not an
approximation bolted on for convenience -- it is the standard formulation, and this
geometry satisfies it exactly (verified: bar lengths are identical inside every group).

The repo's 72-variable formulation is therefore searching a space 4.5 orders of magnitude
larger than the problem needs (50^72 vs 50^16) for solutions that, by the symmetry of the
loads and supports, mostly want to be symmetric anyway.

Two arms separate the two things grouping does:

  groups16   -- optimise the 16 group areas directly and expand to 72 for evaluation.
                Smaller space, but symmetric designs are the only ones reachable.
  sym_mut    -- keep all 72 variables, but make the MUTATION operator symmetry-aware: it
                picks a group and sets every bar in it to the same new value. The full
                space stays reachable through crossover; only the moves are symmetric.

If groups16 wins, the finding is about the problem formulation and applies to every
method here, learned or not. If sym_mut wins too, symmetry is useful as a move structure
and does not require giving up the larger space. If neither wins, the 72-variable
formulation is justified and that is worth knowing explicitly rather than by default.

Same 64,000 NFE, same GA settings, same reference as round-9 track C, so the numbers sit
directly beside "GA tuned, no operator" = 0.8010.
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

MV = np.load(os.path.join(HERE, "max_values.npy"))
MIN_MASK = np.array([True, True, False, True, True])
NFE_BUDGET = 64000
POP = 64

# The classical 16 groups, in the order evaluation/truss_problem.py._generate_geometry
# emits elements: per level, 4 legs, then 8 diagonals, then 4 top chords, then 2 braces.
GROUP_OF_BAR = []
for _lvl in range(4):
    GROUP_OF_BAR += [_lvl * 4 + 0] * 4 + [_lvl * 4 + 1] * 8 + \
                    [_lvl * 4 + 2] * 4 + [_lvl * 4 + 3] * 2
GROUP_OF_BAR = np.array(GROUP_OF_BAR)
N_GROUPS = 16
BARS_IN_GROUP = [np.flatnonzero(GROUP_OF_BAR == g) for g in range(N_GROUPS)]
assert len(GROUP_OF_BAR) == 72 and sum(len(b) for b in BARS_IN_GROUP) == 72


def run_arm(arm, seed):
    prob = TrussProblem()
    NV, AL = prob.n_elements, len(prob.area_map)
    MO = MV[:prob.num_objectives]
    ref = np.ones(prob.num_objectives)
    hv_ind = HV(ref_point=ref)

    grouped = arm.startswith('g') and arm != 'sym_mut'   # groups16 and every g_* arm
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

    def group_util(design):
        """mean |utilisation| of the bars in each of the 16 groups"""
        _, _, _, u = prob.evaluate([int(x) for x in design], return_utilisation=True)
        state['nfe'] += 1
        u = np.abs(np.asarray(u, dtype=float))
        return np.array([u[b].mean() for b in BARS_IN_GROUP])

    def oracle_group_mutate(offspring, ga_):
        """The round-9 oracle question, asked again in group space.

        In the 72-bar space every physics-guided operator lost to blind mutation, and the
        landscape experiment explained why: the best bar to touch reverses with search
        stage, so a fixed rule is wrong for part of the run. Groups are a different
        object -- physically meaningful, four to eight bars each, and their mean
        utilisation is a far less noisy signal than a single bar's. So the question is
        worth asking once more here rather than assumed to carry over.

        Same honest accounting as before: reading utilisation costs a solve, counted as
        an NFE, and the blind control g_random pays it too.
        """
        for i, d in enumerate(offspring):
            d = np.asarray(d, dtype=int).copy()
            gu = group_util(expand(d))
            if arm == 'g_random':
                g = int(rng.integers(0, N_GROUPS))
                d[g] = int(rng.integers(0, AL))
            elif arm == 'g_low':
                d[int(np.argmin(gu))] = int(rng.integers(0, AL))
            elif arm == 'g_high':
                d[int(np.argmax(gu))] = int(rng.integers(0, AL))
            elif arm == 'g_fsd':
                g = int(rng.integers(0, N_GROUPS))
                tgt = prob.area_map[d[g]] * float(np.clip(gu[g], 0.0, 3.0))
                d[g] = int(np.abs(np.asarray(prob.area_map) - tgt).argmin())
            else:
                raise ValueError(arm)
            offspring[i] = d
        return offspring

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
                  mutation_type=(sym_mutate if arm == 'sym_mut'
                                 else oracle_group_mutate if arm.startswith('g_')
                                 else "random"),
                  mutation_num_genes=(None if arm == 'sym_mut' or arm.startswith('g_')
                                      else 1),
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
    final = pfo[-1] if len(pfo) and pfo[-1] is not None else []
    return dict(hv=float(hvs[-1]) if len(hvs) else 0.0, nfe=state['nfe'],
                front_size=len(final), valid_frac=float(np.mean(~all_con)))


ARMS = ['groups16', 'sym_mut']
ARMS2 = ['g_random', 'g_low', 'g_high', 'g_fsd']
SEEDS = [0, 1, 2, 3, 4, 5]
JOBS = [(a, s) for a in ARMS for s in SEEDS]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--set', default='1')
    a = ap.parse_args()
    if a.set == '2':
        ARMS = ARMS2
        JOBS = [(x, sd) for x in ARMS for sd in SEEDS]
    out = os.path.join(HERE, "data", f"r9s{a.set if a.set != '1' else ''}_shard_{a.shard}.jsonl")
    for j, (arm, seed) in enumerate(JOBS):
        if j % a.nshards != a.shard:
            continue
        t0 = time.time()
        rec = run_arm(arm, seed)
        rec.update(job_id=j, name=arm, seed=seed, secs=time.time() - t0)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{j}] {arm} seed={seed} hv={rec['hv']:.4f} nfe={rec['nfe']} "
              f"front={rec['front_size']} {rec['secs']:.0f}s", flush=True)
