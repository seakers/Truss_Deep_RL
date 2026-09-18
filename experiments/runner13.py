"""Round 5, track H -- where is the remaining headroom in the GA?

Track F established that mutation size dominates every method change in rounds 1-4.
The GA has three more settings that were never tuned and are set to values that look
wrong for a multi-objective problem: keep_elitism=0 and keep_parents=0 (nothing
protects a good design from being replaced), parent_selection_type="sss" (steady-state,
which pygad itself warns about when no parents are kept, and which is not a
multi-objective selector), and a 64x1000 population/generation split. This sweeps them
on top of the new 1-bar mutation default to see what is worth chasing next.

Original track F docstring follows.

Round 5, track F -- is the hybrid's advantage the learned operator, or just a
smaller mutation rate?

genetic_algorithm.py and intelligent_mutation_ga.py are identical apart from one
argument: mutation_type="random" with mutation_percent_genes=10 (8 of 72 bars per
offspring) versus the repair actor. Probing the actor shows it edits 1.69 bars when
randomly initialised and ~1.05 once trained, choosing bars and areas near-uniformly.
So the operator may be nothing but a mutation-rate setting. This sweeps the rate with
the plain GA to find out. Everything else -- population, generations, fitness function,
selection, crossover, elitism -- is copied verbatim from genetic_algorithm.py.
"""
import argparse, json, os, sys, time
import numpy as np, pygad
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy
from pymoo.indicators.hv import HV
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress

HERE = os.path.dirname(os.path.abspath(__file__))
MV = np.load(os.path.join(HERE, "max_values.npy"))
POP, NGEN = 64, 1000                      # 64,000 NFE, the budget used in round 4


def run_ga(seed, mutation, n_edits=None, geometric=False, elitism=0,
           selection='sss', pop=POP, ngen=NGEN, crossover='single_point'):
    prob = TrussProblem()
    ng = prob.n_elements
    AL = len(prob.area_map)
    gene_space = [range(AL) for _ in range(ng)]
    min_mask = np.array([True, True, False, True, True])
    nobj = prob.num_objectives
    mo = MV[:nobj]
    hv_ind = HV(ref_point=np.ones(nobj))
    all_des, all_obj, all_con, all_cv = [], [], [], []
    NFE = 0

    # verbatim from genetic_algorithm.py / intelligent_mutation_ga.py (identical there)
    def fitness_func(ga, solution, idx):
        ds = [int(x) for x in solution]
        ov, isc, cv = prob.evaluate(ds)
        nonlocal NFE
        all_des.append(ds); all_obj.append(ov); all_con.append(isc); all_cv.append(cv)
        NFE += 1
        if isc:
            a = -np.array(np.append(mo, cv), dtype=float)
            a[~min_mask] = 0.0
            a[-2:] = cv
            a[-2:][min_mask[-2:]] = -a[-2:][min_mask[-2:]]
            return a
        a = np.array(ov + cv, dtype=float)
        a[min_mask] = -a[min_mask]
        return a

    rng = np.random.default_rng(seed)

    def uniform_k_edits(offspring, ga):
        """The random-init actor's behaviour, written out: pick a bar uniformly, pick
        an area uniformly, repeat k times. geometric=True draws k the way the actor's
        untrained stop head does (p=0.5), otherwise k is fixed."""
        for i in range(offspring.shape[0]):
            k = (1 + rng.geometric(0.5) - 1) if geometric else n_edits
            k = max(1, int(k))
            for _ in range(k):
                offspring[i, rng.integers(ng)] = rng.integers(AL)
        return offspring

    kw = dict(num_generations=ngen, num_parents_mating=pop // 2,
              fitness_func=fitness_func, sol_per_pop=pop, num_genes=ng,
              gene_space=gene_space, parent_selection_type=selection,
              crossover_type=crossover, keep_elitism=elitism, keep_parents=0,
              save_solutions=False, random_seed=seed)
    if mutation == "random":
        kw.update(mutation_type="random", mutation_num_genes=n_edits)
    else:
        kw.update(mutation_type=uniform_k_edits)

    pygad.GA(**kw).run()

    all_des = np.array(all_des); all_obj = np.array(all_obj)
    all_con = np.array(all_con)
    ref = mo * np.ones(nobj)
    ref[~min_mask[:nobj]] = 0.0
    all_obj[all_con] = ref
    norm = all_obj / mo
    hv_obj = deepcopy(norm)
    for i in range(nobj):
        if prob.objective_min_max[i].lower() == 'max':
            hv_obj[:, i] = 1 - norm[:, i]
    _, _, hvs = track_pareto_progress(hv_obj, all_des, all_con, hv_ind)
    return hvs[-1], NFE, float(np.mean(all_con == False))


_B = dict(mutation="random", n_edits=1)          # the new default, on top of which we sweep
ARMS = [
    ("1 bar, sss, no elitism (new default)", dict(_B)),
    ("+ elitism 8",                          dict(_B, elitism=8)),
    ("+ nsga2 selection",                    dict(_B, selection="nsga2")),
    ("+ nsga2 + elitism 8",                  dict(_B, selection="nsga2", elitism=8)),
    ("+ pop 128 x 500 gen",                  dict(_B, pop=128, ngen=500)),
    ("+ pop 32 x 2000 gen",                  dict(_B, pop=32, ngen=2000)),
    ("+ two-point crossover",                dict(_B, crossover="two_points")),
    ("+ uniform crossover",                  dict(_B, crossover="uniform")),
]
SEEDS = [0, 1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r5h_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip():
                r = json.loads(l)
                if r.get("hv") is not None:
                    done.add(r["job_id"])
    jobs = [(i, nm, kw, s) for i, (nm, kw, s) in
            enumerate((nm, kw, s) for nm, kw in ARMS for s in SEEDS)]
    for jid, nm, kw, s in jobs:
        if jid % a.nshards != a.shard or jid in done:
            continue
        t0 = time.perf_counter()
        try:
            hv, nfe, vf = run_ga(s, **kw)
            rec = dict(job_id=jid, name=nm, seed=s, hv=hv, nfe=nfe, valid_frac=vf,
                       secs=time.perf_counter() - t0)
        except Exception as e:
            rec = dict(job_id=jid, name=nm, seed=s, hv=None, error=repr(e))
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[H{a.shard}] {jid:3d} {nm:<34} seed={s} "
              f"hv={rec.get('hv') and round(rec['hv'],4)} {rec.get('secs',0):.0f}s", flush=True)
    print(f"[H{a.shard}] done")


if __name__ == "__main__":
    main()
