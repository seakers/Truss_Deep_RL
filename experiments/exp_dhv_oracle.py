"""Round 10, experiment I -- the DELTA-HYPERVOLUME oracle.

Round 9's oracle study (exp_oracle.py) concluded that utilisation-guided mutation has a
ceiling. That conclusion is withdrawn, and this experiment is the replacement.

The flaw: every operator in exp_oracle.py optimises a SINGLE SCALAR. Fully-stressed design
is a minimum-mass sizing rule with no displacement term and no frequency term, so it
ignores two of the three objectives. The supervised pointer (exp_supervised.py) is labelled
by design_potential, an equal-weight sum of the three normalised objectives. Both are
single directions through objective space, and both were scored by HYPERVOLUME, which pays
for SPREAD across the front. An operator that is excellent at climbing one direction is
expected to underperform on a spread metric, so its losing says nothing about what a
well-aimed operator could do. fsd_all at HV 0.2457 is largely a 64-design population
collapsing onto a fixed point, after which a cumulative-archive HV simply stops growing.

So: build an operator whose target IS the metric. For each offspring, propose K candidate
single-bar edits, evaluate each, and keep the one that adds the most hypervolume to the
running archive. That target is direction-agnostic -- a move that opens a new lightweight
extreme and a move that opens a new stiff extreme both score -- so it cannot collapse the
front the way a fixed rule does.

  random       -- blind control. One uniform (bar, value) edit, charged 1 NFE. Identical to
                  the tuned GA's own operator, so this row should land near 0.80.
  dhv_free     -- K candidates, keep the argmax of true delta-HV. The lookahead is FREE:
                  the rejected candidates are neither charged nor entered into the scoring
                  archive, so the arm gets information, not extra samples.
  dhv_free_soft-- same, but sample among the candidates with positive delta in proportion
                  to it. Controls for the determinism penalty round 9 measured at 0.20-0.25
                  independently on two unrelated operators.
  dhv_charged  -- same as dhv_free but every lookahead evaluation is charged as an NFE and
                  kept in the scoring archive. This is what the oracle actually costs if you
                  compute it directly instead of amortising it into a network.

Why free first. The ceiling question is whether the information is useful at all. Charging
the oracle its own compute conflates "this information is worthless" with "this information
is too expensive to compute directly" -- and amortising an expensive oracle into a cheap
forward pass is the entire premise of a learned operator. dhv_charged answers the second
question; dhv_free answers the first.

Reading the result:
  dhv_free loses  -> the ceiling claim is restored on far stronger evidence than round 9's,
                     because this operator was aimed at the metric and still could not win.
  dhv_free wins   -> the direction is alive and the defect was the target, not the learning.
                     Everything in the repo trains on domination rewards or design_potential
                     while being scored on hypervolume.

Set 2 is the spread diagnostic: re-run round 9's eight physics operators recording final
front cardinality and per-objective range, which round 9 never stored. If those arms lost
through collapsed spread rather than through bad edits, it shows up directly here.

  python exp_dhv_oracle.py --shard 0 --nshards 6 --set 1
  python exp_dhv_oracle.py --shard 0 --nshards 6 --set 2
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
K_CAND = 8                      # candidate edits proposed per offspring
CHECKPOINTS = [8000, 16000, 32000, 64000]


class Archive:
    """Running Pareto front in normalised minimisation space.

    peek() returns what a point WOULD add without inserting it. The dominated shortcut
    matters: a point already dominated by the front adds exactly zero, so it never needs
    the expensive hv_ind call, and late in a run that is most candidates.
    """

    def __init__(self, hv_ind):
        self.hv_ind = hv_ind
        self.pts = None
        self.hv = 0.0
        self.n_hv_calls = 0

    def peek(self, p):
        if self.pts is None:
            self.n_hv_calls += 1
            return float(self.hv_ind(p[None, :]))
        if np.any(np.all(self.pts <= p, axis=1)):        # dominated -> adds nothing
            return 0.0
        keep = ~np.all(p <= self.pts, axis=1)
        self.n_hv_calls += 1
        return float(self.hv_ind(np.vstack([self.pts[keep], p]))) - self.hv

    def insert(self, p):
        if self.pts is None:
            self.pts = p[None, :]
            self.hv = float(self.hv_ind(self.pts))
            return
        if np.any(np.all(self.pts <= p, axis=1)):
            return
        keep = ~np.all(p <= self.pts, axis=1)
        self.pts = np.vstack([self.pts[keep], p])
        self.hv = float(self.hv_ind(self.pts))


def run(rule, seed, budget=NFE_BUDGET, k_cand=K_CAND):
    prob = TrussProblem()
    NV, AL = prob.n_elements, len(prob.area_map)
    areas = np.asarray(prob.area_map)
    MO = MV[:prob.num_objectives]
    NOBJ = prob.num_objectives
    ref = np.ones(NOBJ)
    hv_ind = HV(ref_point=ref)
    is_max = np.array([prob.objective_min_max[k].lower() == 'max' for k in range(NOBJ)])

    all_des, all_obj, all_con, all_cv = [], [], [], []
    state = {'nfe': 0, 'n_mut': 0, 'n_informative': 0, 'n_lookahead': 0}
    arch = Archive(hv_ind)

    def hv_point(obj, isc):
        """Normalised minimisation coordinates, matching the scoring path exactly.

        Infeasible designs are pinned to the reference corner by the scoring code, where
        they contribute nothing, so they are simply not offered to the archive.
        """
        if isc:
            return None
        z = np.asarray(obj, dtype=float)[:NOBJ] / MO
        z = np.where(is_max, 1.0 - z, z)
        return np.clip(z, 0.0, None)

    def log(d, o, isc, cv):
        """Record a design as charged: it counts against NFE and enters the scoring set."""
        all_des.append(d); all_obj.append(o); all_con.append(isc); all_cv.append(cv)
        state['nfe'] += 1
        p = hv_point(o, isc)
        if p is not None:
            arch.insert(p)

    def fit(ga_, sol, idx):
        d = [int(x) for x in sol]
        ov, isc, cv = prob.evaluate(d)
        log(d, ov, isc, cv)
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float)
            a[~MIN_MASK] = 0.0
            a[-2:] = cv
            a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]
            return a
        a = np.array(ov + cv, dtype=float)
        a[MIN_MASK] = -a[MIN_MASK]
        return a

    rng = np.random.default_rng(seed + 10001)

    def snap(target_area):
        return int(np.abs(areas - target_area).argmin())

    charged = rule.endswith('_charged')

    def mutate(offspring, ga_):
        for i, design in enumerate(offspring):
            d = np.asarray(design, dtype=int).copy()

            # ---------------- blind control ----------------
            if rule == 'random':
                b = int(rng.integers(0, NV))
                d[b] = int(rng.integers(0, AL))
                offspring[i] = d
                continue

            # ---------------- delta-HV oracle ----------------
            if rule.startswith('dhv'):
                cands, deltas = [], []
                for _ in range(k_cand):
                    c = d.copy()
                    b = int(rng.integers(0, NV))
                    c[b] = int(rng.integers(0, AL))
                    cd = [int(x) for x in c]
                    o, isc, cv = prob.evaluate(cd)
                    state['n_lookahead'] += 1
                    p = hv_point(o, isc)
                    # peek BEFORE any charged insert. Inserting first would put the point
                    # in the front it is about to be scored against, making every delta
                    # identically zero.
                    deltas.append(0.0 if p is None else arch.peek(p))
                    if charged:
                        log(cd, o, isc, cv)
                    cands.append(c)

                deltas = np.asarray(deltas, dtype=float)
                state['n_mut'] += 1
                pos = deltas > 0.0
                if pos.any():
                    state['n_informative'] += 1
                    if rule.startswith('dhv_free_soft'):
                        w = np.where(pos, deltas, 0.0)
                        j = int(rng.choice(len(cands), p=w / w.sum()))
                    else:
                        j = int(np.argmax(deltas))
                else:
                    # no candidate advances the front: the oracle has no signal, so fall
                    # back to blind choice rather than committing to an arbitrary argmax
                    j = int(rng.integers(0, len(cands)))
                offspring[i] = cands[j]
                continue

            # ---------------- round 9 physics rules, for the spread diagnostic ---------
            _, _, _, util = prob.evaluate([int(x) for x in d], return_utilisation=True)
            state['nfe'] += 1
            u = np.abs(np.asarray(util, dtype=float))
            active = d > 0

            if rule == 'random_solve':
                b = int(rng.integers(0, NV))
                d[b] = int(rng.integers(0, AL))
            elif rule == 'fsd_all':
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
                    d[b] = int(rng.integers(0, d[b] + 1))
                else:
                    d[int(rng.integers(0, NV))] = int(rng.integers(0, AL))
            elif rule == 'high_util':
                b = int(np.argmax(u))
                d[b] = int(rng.integers(d[b], AL))
            elif rule == 'hi_fsd':
                b = int(np.argmax(u))
                d[b] = snap(areas[d[b]] * float(np.clip(u[b], 0.0, 3.0)))
            elif rule == 'hi_soft':
                w = u + 1e-9
                b = int(rng.choice(NV, p=w / w.sum()))
                d[b] = int(rng.integers(0, AL))
            elif rule == 'lo_fsd':
                cand = np.flatnonzero(active)
                if cand.size:
                    b = int(cand[np.argmin(u[cand])])
                    d[b] = snap(areas[d[b]] * float(np.clip(u[b], 0.0, 3.0)))
            else:
                raise ValueError(rule)

            offspring[i] = d
        return offspring

    def on_gen(ga_):
        if state['nfe'] >= budget:
            return "stop"

    ga = pygad.GA(num_generations=100000, num_parents_mating=POP // 2, fitness_func=fit,
                  sol_per_pop=POP, num_genes=NV, gene_space=[range(AL)] * NV,
                  parent_selection_type="tournament", K_tournament=3,
                  crossover_type="single_point", mutation_type=mutate,
                  keep_elitism=8, keep_parents=0, save_solutions=False,
                  on_generation=on_gen, random_seed=seed)
    ga.run()

    # ---- scoring: identical path to exp_oracle.py so the numbers are comparable ----
    all_obj_a = np.array(all_obj); all_des_a = np.array(all_des)
    all_con_a = np.array(all_con)
    ref_values = MO * ref
    ref_values[~MIN_MASK[:NOBJ]] = 0.0
    all_obj_a[all_con_a] = ref_values
    norm = all_obj_a / MO
    hv_obj = deepcopy(norm)
    for k in range(NOBJ):
        if prob.objective_min_max[k].lower() == 'max':
            hv_obj[:, k] = 1 - norm[:, k]
    _, fronts, hvs = track_pareto_progress(hv_obj, all_des_a, all_con_a, hv_ind)

    curve = {str(c): (float(hvs[c - 1]) if len(hvs) >= c else None) for c in CHECKPOINTS}

    # ---- spread diagnostic: what the final front actually covers ----
    front = fronts[-1]
    if front is None or len(front) == 0:
        n_front, rng_obj, spread = 0, [0.0] * NOBJ, 0.0
    else:
        front = np.asarray(front, dtype=float)
        n_front = int(front.shape[0])
        rng_obj = [float(front[:, k].max() - front[:, k].min()) for k in range(NOBJ)]
        spread = float(np.prod([r for r in rng_obj]))

    return dict(
        hv=float(hvs[-1]), nfe=state['nfe'],
        n_front=n_front, range_per_obj=rng_obj, spread_volume=spread,
        curve=curve,
        n_mut=state['n_mut'], n_informative=state['n_informative'],
        informative_frac=(state['n_informative'] / state['n_mut']) if state['n_mut'] else None,
        n_lookahead=state['n_lookahead'], n_hv_calls=arch.n_hv_calls,
        feasible_frac=float((~all_con_a).mean()),
    )


ARMS1 = ['random', 'dhv_free', 'dhv_free_soft', 'dhv_charged']
ARMS2 = ['random_solve', 'low_util', 'high_util', 'fsd_one', 'fsd_all',
         'hi_fsd', 'lo_fsd', 'hi_soft']
SEEDS = [0, 1, 2, 3, 4, 5]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--set', default='1')
    ap.add_argument('--budget', type=int, default=NFE_BUDGET)
    ap.add_argument('--kcand', type=int, default=K_CAND)
    ap.add_argument('--seeds', type=int, default=len(SEEDS))
    a = ap.parse_args()

    arms = ARMS1 if a.set == '1' else ARMS2
    seeds = SEEDS[:a.seeds]
    jobs = [(r, s) for r in arms for s in seeds]

    out = os.path.join(HERE, "data", f"r10d{a.set}_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            line = line.strip()
            if line:
                done.add(json.loads(line)['job_id'])

    for j, (rule, seed) in enumerate(jobs):
        if j % a.nshards != a.shard or j in done:
            continue
        t0 = time.time()
        res = run(rule, seed, budget=a.budget, k_cand=a.kcand)
        rec = dict(job_id=j, name=rule, seed=seed, secs=time.time() - t0,
                   budget=a.budget, k_cand=a.kcand, **res)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{j}] {rule} seed={seed} hv={res['hv']:.4f} nfe={res['nfe']} "
              f"front={res['n_front']} spread={res['spread_volume']:.4g} "
              f"info={res['informative_frac']} {time.time()-t0:.0f}s", flush=True)
