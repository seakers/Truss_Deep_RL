"""Round 6 addendum -- re-confirm the GA tuning on the LAB'S ORIGINAL reference.

Stages A-C were measured against a rebuilt max_values.npy. That rebuild has since been
reverted (it is not ours to change; see RESULTS_ROUND6.md). This re-runs the Stage C
confirmation against the original reference, with clipping, so the headline number is on
the lab's own scale and comparable to rounds 1-5. Clipping is conservative: it caps a
design that beats the reference rather than letting it earn unbounded credit, so it
UNDERSTATES the tuned GA. Held-out seeds 4-11, never used for tuning.
"""
import argparse, json, os, sys, time
import numpy as np, pygad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))
from pymoo.indicators.hv import HV
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress

MV = np.load(os.path.join(HERE, "max_values.npy"))     # whatever is current = lab original
MO = MV[:3]
MIN_MASK = np.array([True, True, False, True, True])
HVI = HV(ref_point=np.ones(3))


def run(seed, n_edits, elitism, selection):
    prob = TrussProblem(); ng = prob.n_elements; AL = len(prob.area_map)
    des, objs, cons = [], [], []

    def fit(g_, sol, idx):
        ov, isc, cv = prob.evaluate([int(x) for x in sol])
        des.append([int(x) for x in sol]); objs.append(ov); cons.append(isc)
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float); a[~MIN_MASK] = 0.0
            a[-2:] = cv; a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]; return a
        a = np.array(ov + cv, dtype=float); a[MIN_MASK] = -a[MIN_MASK]; return a

    kw = dict(num_generations=1000, num_parents_mating=32, fitness_func=fit,
              sol_per_pop=64, num_genes=ng, gene_space=[range(AL)] * ng,
              parent_selection_type=selection, crossover_type="single_point",
              mutation_type="random", mutation_num_genes=n_edits,
              keep_elitism=elitism, keep_parents=0, save_solutions=False,
              random_seed=seed)
    if selection == "tournament":
        kw["K_tournament"] = 3
    pygad.GA(**kw).run()

    objs = np.array(objs); cons = np.array(cons); des = np.array(des)
    ref = MO * np.ones(3); ref[~MIN_MASK[:3]] = 0.0
    o2 = objs.copy(); o2[cons] = ref
    norm = o2 / MO
    hv_obj = norm.copy(); hv_obj[:, 2] = 1 - norm[:, 2]
    feas = hv_obj[~cons]
    escape = float((feas < 0).any(1).mean()) if len(feas) else 0.0
    _, _, raw = track_pareto_progress(hv_obj.copy(), des, cons, HVI)   # clipped inside
    return dict(hv_clip=raw[-1], escape=escape, valid_frac=float((~cons).mean()))


ARMS = [
    ("original default (8 bars, no elitism)", dict(n_edits=8, elitism=0, selection="sss")),
    ("round-5 default (1 bar, no elitism)",   dict(n_edits=1, elitism=0, selection="sss")),
    ("tuned (1 bar, elitism 8, tournament)",  dict(n_edits=1, elitism=8, selection="tournament")),
    ("tuned but sss selection",               dict(n_edits=1, elitism=8, selection="sss")),
]
SEEDS = [4, 5, 6, 7, 8, 9, 10, 11]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r6d_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip() and json.loads(l).get("hv_clip") is not None:
                done.add(json.loads(l)["job_id"])
    jid = -1
    for nm, cfg in ARMS:
        for s in SEEDS:
            jid += 1
            if jid % a.nshards != a.shard or jid in done:
                continue
            t0 = time.perf_counter()
            try:
                rec = dict(job_id=jid, name=nm, seed=s, **run(s, **cfg),
                           secs=time.perf_counter() - t0)
            except Exception as e:
                rec = dict(job_id=jid, name=nm, seed=s, hv_clip=None, error=repr(e))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[D{a.shard}] {jid:3d} {nm:<40} s={s} "
                  f"hv={rec.get('hv_clip') and round(rec['hv_clip'],4)} {rec.get('secs',0):.0f}s", flush=True)
    print(f"[D{a.shard}] done")


if __name__ == "__main__":
    main()
