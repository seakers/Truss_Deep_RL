"""Round 9, experiment I -- the ORACLE upper bound on any utilisation-reading operator.

Rounds 5-8 kept asking "why isn't the learned operator learning?" That question only
matters if a GOOD operator would help. This measures that ceiling directly: replace the
transformer with hand-coded operators built from textbook structural engineering, give
them perfect per-bar stress information for free, and run them in the same GA at the same
evaluation budget as everything else.

  low_util   -- shrink the least-stressed active bar. Mass removal: a bar carrying no
                load is free mass. This is the half of the rule the trained pointer
                actually leans toward (round 8: corr(|util|, p) = -0.73).
  high_util  -- thicken the most-stressed bar. Constraint repair.
  fsd_one    -- fully stressed design applied to one random bar: A_new = A * |util|,
                snapped to the grid. The classical sizing rule, one bar at a time.
  fsd_all    -- classical FSD: resize EVERY bar to its fully-stressed area in one move.
                The strongest non-learned operator available and a real baseline from the
                structural optimisation literature.

Each arm pays honestly: reading utilisation needs a solve, and that solve is counted as
an NFE exactly as it is for the stress-aware transformer, so a cheap blind operator gets
more generations for the same budget. Budget and settings match runner23 track C, so the
"GA tuned, no operator" numbers there are directly comparable.

If none of these beat the tuned GA, then utilisation-guided single-bar mutation has no
headroom on this problem, and no amount of better RL would have found any. That is a
result about the PROBLEM, not about the network -- and it is the thing worth knowing.
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


def run_oracle(rule, seed):
    prob = TrussProblem()
    NV, AL = prob.n_elements, len(prob.area_map)
    areas = np.asarray(prob.area_map)
    MO = MV[:prob.num_objectives]
    ref = np.ones(prob.num_objectives)
    hv_ind = HV(ref_point=ref)

    all_des, all_obj, all_con, all_cv = [], [], [], []
    state = {'nfe': 0}

    def fit(ga_, sol, idx):
        d = [int(x) for x in sol]
        ov, isc, cv = prob.evaluate(d)
        all_des.append(d); all_obj.append(ov); all_con.append(isc); all_cv.append(cv)
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

    rng = np.random.default_rng(seed + 9001)

    def snap(target_area):
        """nearest grid index for a desired area"""
        return int(np.abs(areas - target_area).argmin())

    def mutate(offspring, ga_):
        for i, design in enumerate(offspring):
            d = np.asarray(design, dtype=int).copy()

            if rule == 'random':
                # control: identical to the plain GA's operator, but paying the same
                # utilisation solve so the NFE comparison isolates the RULE, not the cost
                _, _, _, util = prob.evaluate([int(x) for x in d], return_utilisation=True)
                state['nfe'] += 1
                b = int(rng.integers(0, NV))
                d[b] = int(rng.integers(0, AL))
                offspring[i] = d
                continue

            _, _, _, util = prob.evaluate([int(x) for x in d], return_utilisation=True)
            state['nfe'] += 1
            u = np.abs(np.asarray(util, dtype=float))
            active = d > 0

            if rule == 'fsd_all':
                # classical fully-stressed design: every bar resized toward its
                # fully-stressed area at once. Bars carrying nothing shrink to the floor.
                tgt = areas[d] * np.clip(u, 0.0, 3.0)
                new = np.array([snap(t) for t in tgt], dtype=int)
                d = np.where(active, new, d)

            elif rule == 'fsd_one':
                cand = np.flatnonzero(active)
                if cand.size:
                    b = int(rng.choice(cand))
                    d[b] = snap(areas[d[b]] * float(np.clip(u[b], 0.0, 3.0)))

            elif rule == 'low_util':
                cand = np.flatnonzero(active)
                if cand.size:
                    b = int(cand[np.argmin(u[cand])])
                    d[b] = int(rng.integers(0, d[b] + 1))    # shrink (0 = remove)
                else:
                    d[int(rng.integers(0, NV))] = int(rng.integers(0, AL))

            elif rule == 'high_util':
                b = int(np.argmax(u))
                d[b] = int(rng.integers(d[b], AL))           # thicken


            elif rule == 'hi_fsd':
                # the cell the landscape data says should win: pick the bar where a
                # mutation has the largest effect (high |util| -- biggest gains AND
                # biggest losses), then remove the loss half by choosing the value from
                # physics instead of at random.
                b = int(np.argmax(u))
                d[b] = snap(areas[d[b]] * float(np.clip(u[b], 0.0, 3.0)))

            elif rule == 'hi_soft':
                # control for a confound in high_util/hi_fsd: those always edit the SAME
                # bar (the argmax), so every offspring in the population gets the same
                # gene touched and diversity collapses. This samples the bar in
                # proportion to |util| -- same physics preference, no determinism -- so
                # the two explanations for their poor scores can be told apart.
                w = u + 1e-9
                b = int(rng.choice(NV, p=w / w.sum()))
                d[b] = int(rng.integers(0, AL))

            elif rule == 'lo_fsd':
                # the same pairing on the other end: least-loaded bar, FSD value.
                # Shrinks dead weight by exactly the right amount.
                cand = np.flatnonzero(active)
                if cand.size:
                    b = int(cand[np.argmin(u[cand])])
                    d[b] = snap(areas[d[b]] * float(np.clip(u[b], 0.0, 3.0)))

            else:
                raise ValueError(rule)

            offspring[i] = d
        return offspring

    def on_gen(ga_):
        if state['nfe'] >= NFE_BUDGET:
            return "stop"

    ga = pygad.GA(num_generations=100000, num_parents_mating=POP // 2, fitness_func=fit,
                  sol_per_pop=POP, num_genes=NV, gene_space=[range(AL)] * NV,
                  parent_selection_type="tournament", K_tournament=3,
                  crossover_type="single_point", mutation_type=mutate,
                  keep_elitism=8, keep_parents=0, save_solutions=False,
                  on_generation=on_gen, random_seed=seed)
    ga.run()

    all_obj = np.array(all_obj); all_des = np.array(all_des)
    all_con = np.array(all_con)
    ref_values = MO * ref
    ref_values[~MIN_MASK[:prob.num_objectives]] = 0.0
    all_obj[all_con] = ref_values
    norm = all_obj / MO
    hv_obj = deepcopy(norm)
    for k in range(prob.num_objectives):
        if prob.objective_min_max[k].lower() == 'max':
            hv_obj[:, k] = 1 - norm[:, k]
    _, _, hvs = track_pareto_progress(hv_obj, all_des, all_con, hv_ind)
    return float(hvs[-1]), state['nfe']


ARMS = ['random', 'low_util', 'high_util', 'fsd_one', 'fsd_all']
ARMS2 = ['hi_fsd', 'lo_fsd', 'hi_soft']
SEEDS = [0, 1, 2, 3, 4, 5]
JOBS = [(a, s) for a in ARMS for s in SEEDS]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--set', default='1')
    a = ap.parse_args()
    if a.set == "2":
        ARMS = ARMS2
        JOBS = [(r, sd) for r in ARMS for sd in SEEDS]
    out = os.path.join(HERE, "data", f"r9o{a.set if a.set!='1' else ''}_shard_{a.shard}.jsonl")
    for j, (rule, seed) in enumerate(JOBS):
        if j % a.nshards != a.shard:
            continue
        t0 = time.time()
        hv, nfe = run_oracle(rule, seed)
        with open(out, 'a') as f:
            f.write(json.dumps(dict(job_id=j, name=rule, seed=seed, hv=hv,
                                    nfe=nfe, secs=time.time() - t0)) + "\n")
        print(f"[{j}] {rule} seed={seed} hv={hv:.4f} nfe={nfe} {time.time()-t0:.0f}s", flush=True)
