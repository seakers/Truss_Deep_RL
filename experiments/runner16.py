"""Round 6, step 2 -- tune the GA. All numbers on the REPAIRED reference (max_values.npy
with the frequency axis rebuilt); not comparable to rounds 1-5.

Stage A: keep_elitism x mutation_num_genes. These interact -- round 5's 1-bar mutation
was compensating for the fact that nothing protected a good design, so with elitism on
the best mutation size may move back up. Seeds 0-3 (screening).
Stage B: everything else, on Stage A's winner. Seeds 0-3 (screening).
Stage C: Stage A/B winner vs the round-5 default on FRESH seeds 4-11 (confirmation).

Stage selected with --stage; B and C read the winner from the previous stage's records.
"""
import argparse, json, os, sys, time
import numpy as np, pygad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))
from pymoo.indicators.hv import HV
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress

MV = np.load(os.path.join(HERE, "max_values.npy"))
MO = MV[:3]
MIN_MASK = np.array([True, True, False, True, True])
HVI = HV(ref_point=np.ones(3))
BUDGET = 64000


def run(seed, n_edits=1, elitism=0, selection="sss", crossover="single_point",
        pop=64, keep_parents=0):
    ngen = BUDGET // pop
    prob = TrussProblem(); ng = prob.n_elements; AL = len(prob.area_map)
    des, objs, cons = [], [], []

    def fit(g_, sol, idx):
        ov, isc, cv = prob.evaluate([int(x) for x in sol])
        des.append([int(x) for x in sol]); objs.append(ov); cons.append(isc)
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float); a[~MIN_MASK] = 0.0
            a[-2:] = cv; a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]; return a
        a = np.array(ov + cv, dtype=float); a[MIN_MASK] = -a[MIN_MASK]; return a

    kw = dict(num_generations=ngen, num_parents_mating=pop // 2, fitness_func=fit,
              sol_per_pop=pop, num_genes=ng, gene_space=[range(AL)] * ng,
              parent_selection_type=selection, crossover_type=crossover,
              mutation_type="random", mutation_num_genes=n_edits,
              keep_elitism=elitism, keep_parents=keep_parents,
              save_solutions=False, random_seed=seed)
    if selection == "tournament":
        kw["K_tournament"] = 3
    pygad.GA(**kw).run()

    objs = np.array(objs); cons = np.array(cons); des = np.array(des)
    ref = MO * np.ones(3); ref[~MIN_MASK[:3]] = 0.0
    o2 = objs.copy(); o2[cons] = ref
    norm = o2 / MO
    hv_obj = norm.copy(); hv_obj[:, 2] = 1 - norm[:, 2]
    _, _, hvs = track_pareto_progress(hv_obj, des, cons, HVI)
    feas = hv_obj[~cons]
    return dict(hv=hvs[-1], escape=float((feas < 0).any(1).mean()) if len(feas) else 0.0,
                valid_frac=float((~cons).mean()), nfe=int(len(des)))


def stage_a():
    J = []
    for el in [0, 2, 4, 8, 16, 32]:
        for ne in [1, 2, 4]:
            J.append((f"elitism {el:>2}, {ne} bar(s)", dict(elitism=el, n_edits=ne), [0, 1, 2, 3]))
    return J


def best_of(tag, keyfn=None):
    rows = []
    for f in os.listdir(HERE):
        if f.startswith(f"r6{tag}_shard_") and f.endswith(".jsonl"):
            for l in open(os.path.join(HERE, "data", f)):
                if l.strip():
                    r = json.loads(l)
                    if r.get("hv") is not None:
                        rows.append(r)
    agg = {}
    for r in rows:
        agg.setdefault(r["name"], []).append(r["hv"])
    if not agg:
        raise SystemExit(f"stage {tag} has no records yet -- run it first")
    nm = max(agg, key=lambda k: np.mean(agg[k]))
    cfg = json.loads([r for r in rows if r["name"] == nm][0]["cfg"])
    return nm, cfg


def stage_b():
    _, base = best_of("b0")
    J = []
    for sel in ["sss", "nsga2", "tournament", "rank"]:
        J.append((f"select {sel}", dict(base, selection=sel), [0, 1, 2, 3]))
    for cx in ["two_points", "uniform"]:
        J.append((f"crossover {cx}", dict(base, crossover=cx), [0, 1, 2, 3]))
    for p in [32, 128]:
        J.append((f"pop {p}", dict(base, pop=p), [0, 1, 2, 3]))
    J.append(("keep_parents 8", dict(base, keep_parents=8), [0, 1, 2, 3]))
    return J


def stage_c():
    _, win = best_of("b1")
    S = [4, 5, 6, 7, 8, 9, 10, 11]
    return [("tuned winner", win, S),
            ("round-5 default (1 bar, no elitism)", dict(n_edits=1, elitism=0), S),
            ("original default (8 bars, no elitism)", dict(n_edits=8, elitism=0), S)]


STAGES = {"a": ("b0", stage_a), "b": ("b1", stage_b), "c": ("b2", stage_c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    a = ap.parse_args()
    tag, build = STAGES[a.stage]
    jobs = []
    for nm, cfg, seeds in build():
        for s in seeds:
            jobs.append((nm, cfg, s))
    out = os.path.join(HERE, "data", f"r6{tag}_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip() and json.loads(l).get("hv") is not None:
                done.add(json.loads(l)["job_id"])
    for jid, (nm, cfg, s) in enumerate(jobs):
        if jid % a.nshards != a.shard or jid in done:
            continue
        t0 = time.perf_counter()
        try:
            rec = dict(job_id=jid, name=nm, seed=s, cfg=json.dumps(cfg),
                       **run(s, **cfg), secs=time.perf_counter() - t0)
        except Exception as e:
            rec = dict(job_id=jid, name=nm, seed=s, cfg=json.dumps(cfg), hv=None, error=repr(e))
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{a.stage.upper()}{a.shard}] {jid:3d} {nm:<38} s={s} "
              f"hv={rec.get('hv') and round(rec['hv'],4)} {rec.get('secs',0):.0f}s", flush=True)
    print(f"[{a.stage.upper()}{a.shard}] done")


if __name__ == "__main__":
    main()
