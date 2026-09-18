"""Round 5, track I -- re-measure the GA sweep with a metric that is not broken.

max_values.npy is the per-objective best of a 16,000-sample random search, and every
hypervolume in rounds 1-5 divides by it. That is fine only while no method beats it.
The old 8-bar mutation never did (1 design in 61,857). A 1-bar mutation does, on 48% of
its feasible designs -- frequency reaches 1.289 against a reference of 0.879 -- which
makes the normalised coordinate negative and lets pymoo integrate volume outside the
unit cube, so the reported score exceeds 1 and is not comparable to anything.

This records three numbers per run:
  hv_raw     -- what the repo reports today (inflated once designs escape the box)
  hv_clip    -- coordinates clipped at 0, so the score stays in [0,1] and stays
                comparable to every historical number; a design better than the
                reference simply stops earning more credit on that axis
  escape     -- fraction of feasible designs outside the box, i.e. how much hv_raw
                is to be distrusted
plus the per-objective best, so max_values can be rebuilt from real data.
"""
import argparse, json, os, sys, time
import numpy as np, pygad
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pymoo.indicators.hv import HV
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress

HERE = os.path.dirname(os.path.abspath(__file__))
MV = np.load(os.path.join(HERE, "max_values.npy"))
MO = MV[:3]
MIN_MASK = np.array([True, True, False, True, True])
HVI = HV(ref_point=np.ones(3))


def run(seed, n_edits, elitism=0, selection="sss", pop=64, ngen=1000):
    prob = TrussProblem(); ng = prob.n_elements; AL = len(prob.area_map)
    des, objs, cons = [], [], []

    def fit(ga, sol, idx):
        ds = [int(x) for x in sol]
        ov, isc, cv = prob.evaluate(ds)
        des.append(ds); objs.append(ov); cons.append(isc)
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float); a[~MIN_MASK] = 0.0
            a[-2:] = cv; a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]
            return a
        a = np.array(ov + cv, dtype=float); a[MIN_MASK] = -a[MIN_MASK]
        return a

    pygad.GA(num_generations=ngen, num_parents_mating=pop // 2, fitness_func=fit,
             sol_per_pop=pop, num_genes=ng, gene_space=[range(AL)] * ng,
             parent_selection_type=selection, crossover_type="single_point",
             mutation_type="random", mutation_num_genes=n_edits,
             keep_elitism=elitism, keep_parents=0, save_solutions=False,
             random_seed=seed).run()

    objs = np.array(objs); cons = np.array(cons); des = np.array(des)
    ref = MO * np.ones(3); ref[~MIN_MASK[:3]] = 0.0
    o2 = objs.copy(); o2[cons] = ref
    norm = o2 / MO
    hv_obj = norm.copy(); hv_obj[:, 2] = 1 - norm[:, 2]
    _, _, raw = track_pareto_progress(hv_obj, des, cons, HVI)
    _, _, clip = track_pareto_progress(np.clip(hv_obj, 0.0, None), des, cons, HVI)
    feas = hv_obj[~cons]
    fo = objs[~cons]
    return dict(hv_raw=raw[-1], hv_clip=clip[-1],
                escape=float((feas < 0).any(1).mean()) if len(feas) else 0.0,
                best_mass=float(fo[:, 0].min()), best_disp=float(fo[:, 1].min()),
                best_freq=float(fo[:, 2].max()), n_feas=int(len(feas)), nfe=len(des))


ARMS = [
    ("8 bars, no elitism (old default)", dict(n_edits=8)),
    ("4 bars, no elitism",               dict(n_edits=4)),
    ("2 bars, no elitism",               dict(n_edits=2)),
    ("1 bar, no elitism (new default)",  dict(n_edits=1)),
    ("1 bar + elitism 8",                dict(n_edits=1, elitism=8)),
    ("1 bar + nsga2 + elitism 8",        dict(n_edits=1, elitism=8, selection="nsga2")),
    ("1 bar + elitism 8, pop 128",       dict(n_edits=1, elitism=8, pop=128, ngen=500)),
]
SEEDS = [0, 1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r5i_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip():
                r = json.loads(l)
                if r.get("hv_raw") is not None:
                    done.add(r["job_id"])
    jid = -1
    for nm, kw in ARMS:
        for s in SEEDS:
            jid += 1
            if jid % a.nshards != a.shard or jid in done:
                continue
            t0 = time.perf_counter()
            try:
                rec = dict(job_id=jid, name=nm, seed=s, **run(s, **kw),
                           secs=time.perf_counter() - t0)
            except Exception as e:
                rec = dict(job_id=jid, name=nm, seed=s, hv_raw=None, error=repr(e))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[I{a.shard}] {jid:3d} {nm:<34} s={s} raw={rec.get('hv_raw') and round(rec['hv_raw'],4)} "
                  f"clip={rec.get('hv_clip') and round(rec['hv_clip'],4)} {rec.get('secs',0):.0f}s", flush=True)
    print(f"[I{a.shard}] done")


if __name__ == "__main__":
    main()
