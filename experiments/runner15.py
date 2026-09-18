"""Round 6, step 1a -- find real bounds for max_values.npy.

The current reference is the per-objective best of a 16,000-sample random search.
A GA reaches frequency 1.585 against its 0.879, so the maximisation axis is 80% low and
every design past it scores a negative normalised coordinate. This pools every design
evaluated across a deliberately diverse set of methods and reports the extremes.
Diversity matters more than seed count: the goal is a bound, not a mean.
"""
import argparse, json, os, sys, time
import numpy as np, pygad, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
WORK = os.path.join(HERE, "r4work")
os.chdir(WORK)
from evaluation.truss_problem import TrussProblem
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga

MV = np.load(os.path.join(HERE, "max_values.npy"))
MO = MV[:3]
MIN_MASK = np.array([True, True, False, True, True])
POP, NGEN = 64, 1000


def ga(seed, n_edits, elitism=0):
    prob = TrussProblem(); ng = prob.n_elements; AL = len(prob.area_map)
    objs, cons, cvs = [], [], []
    def fit(g_, sol, idx):
        ov, isc, cv = prob.evaluate([int(x) for x in sol])
        objs.append(ov); cons.append(isc); cvs.append(cv)
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float); a[~MIN_MASK] = 0.0
            a[-2:] = cv; a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]; return a
        a = np.array(ov + cv, dtype=float); a[MIN_MASK] = -a[MIN_MASK]; return a
    pygad.GA(num_generations=NGEN, num_parents_mating=POP // 2, fitness_func=fit,
             sol_per_pop=POP, num_genes=ng, gene_space=[range(AL)] * ng,
             parent_selection_type="sss", crossover_type="single_point",
             mutation_type="random", mutation_num_genes=n_edits, keep_elitism=elitism,
             keep_parents=0, save_solutions=False, random_seed=seed).run()
    return np.array(objs), np.array(cons), np.array(cvs)


def rnd(seed, n=64000):
    prob = TrussProblem(); rng = np.random.default_rng(seed)
    AL = len(prob.area_map)
    objs, cons, cvs = [], [], []
    for _ in range(n):
        ov, isc, cv = prob.evaluate(list(rng.integers(0, AL, prob.n_elements)))
        objs.append(ov); cons.append(isc); cvs.append(cv)
    return np.array(objs), np.array(cons), np.array(cvs)


def imga(seed):
    p = dict(num_epochs=NGEN, mini_batch_size=POP, gamma=0.999, learning_rate=3e-4,
             clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
             model_folder=None, random_init_actor=False,
             date_str=f"DOM_DES_t{seed % 5}")
    p['lambda'] = 0.9
    torch.manual_seed(seed); np.random.seed(seed)
    r = run_intelligent_mutation_ga(MV, p)
    # r = all_des, all_obj, front_des, front_obj, hvs, NFE; all_obj has infeasibles
    # already overwritten with the reference, so recompute feasibility from scratch
    prob = TrussProblem()
    objs, cons, cvs = [], [], []
    for d in r[0]:
        ov, isc, cv = prob.evaluate([int(x) for x in d])
        objs.append(ov); cons.append(isc); cvs.append(cv)
    return np.array(objs), np.array(cons), np.array(cvs)


ARMS = [
    ("random search",        lambda s: rnd(s)),
    ("GA 8 bars",            lambda s: ga(s, 8)),
    ("GA 1 bar",             lambda s: ga(s, 1)),
    ("GA 1 bar + elitism 8", lambda s: ga(s, 1, elitism=8)),
    ("GA 1 bar + elitism 32",lambda s: ga(s, 1, elitism=32)),
    ("IMGA trained",         lambda s: imga(s)),
]
SEEDS = [0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r6a_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip() and json.loads(l).get("n_feas") is not None:
                done.add(json.loads(l)["job_id"])
    jid = -1
    for nm, fn in ARMS:
        for s in SEEDS:
            jid += 1
            if jid % a.nshards != a.shard or jid in done:
                continue
            t0 = time.perf_counter()
            try:
                objs, cons, cvs = fn(s)
                f = objs[~cons]
                rec = dict(job_id=jid, name=nm, seed=s, n_feas=int(len(f)), n_all=int(len(objs)),
                           mass_min=float(f[:, 0].min()), mass_max=float(f[:, 0].max()),
                           disp_min=float(f[:, 1].min()), disp_max=float(f[:, 1].max()),
                           freq_min=float(f[:, 2].min()), freq_max=float(f[:, 2].max()),
                           cv_max=[float(cvs[:, i].max()) for i in range(cvs.shape[1])],
                           secs=time.perf_counter() - t0)
            except Exception as e:
                rec = dict(job_id=jid, name=nm, seed=s, n_feas=None, error=repr(e))
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[A{a.shard}] {jid:3d} {nm:<22} s={s} feas={rec.get('n_feas')} "
                  f"freq_max={rec.get('freq_max') and round(rec['freq_max'],4)} "
                  f"{rec.get('secs',0):.0f}s", flush=True)
    print(f"[A{a.shard}] done")


if __name__ == "__main__":
    main()
