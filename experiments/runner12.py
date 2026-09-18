"""Round 5, track G -- the paired confirmation track F could not give.

Track F swept mutation size in a standalone harness and found hypervolume monotone in
it, with every round-4 hybrid number landing where its measured edit count predicts.
That was unpaired against round 4's IMGA runs. This runs all four arms through the
REPO's own functions, on the same eight seeds, so the comparison is paired and is
measuring shipped code rather than a reimplementation:

    plain GA, 1 bar mutated      (the new default)
    plain GA, 8 bars mutated     (the old default)
    IMGA, untrained actor
    IMGA, trained actor          (the round-4 DOM_DES checkpoints t0-t4)

If the learned operator is doing anything beyond setting a mutation rate, arm 1 should
lose to arm 4. If it is not, arm 1 should match or beat it.
"""
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
WORK = os.path.join(HERE, "r4work")          # holds results/DOM_DES_t*/actor_repair_model.pth
os.chdir(WORK)                               # IMGA loads results/<date_str>/... relatively

from optimization.genetic_algorithm import run_genetic_algorithm
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga

MV = np.load(os.path.join(HERE, "max_values.npy"))
BASE = dict(num_epochs=1000, mini_batch_size=64, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None)
BASE['lambda'] = 0.9

ARMS = [
    ("GA 1 bar (new default)",  dict(fn='ga',   mutation_num_genes=1)),
    ("GA 8 bars (old default)", dict(fn='ga',   mutation_num_genes=8)),
    ("IMGA untrained actor",    dict(fn='imga', random_init_actor=True)),
    ("IMGA trained actor",      dict(fn='imga', random_init_actor=False)),
]
SEEDS = list(range(8))


def run(arm, seed):
    cfg = dict(ARMS[arm][1])
    fn = cfg.pop('fn')
    p = dict(BASE, **cfg)
    torch.manual_seed(seed); np.random.seed(seed)
    if fn == 'ga':
        r = run_genetic_algorithm(MV, p)
    else:
        # cycle the five round-4 checkpoints across seeds, as runner10 did
        p['date_str'] = f"DOM_DES_t{seed % 5}"
        r = run_intelligent_mutation_ga(MV, p)
    return r[4][-1], r[5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r5g_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip():
                r = json.loads(l)
                if r.get("hv") is not None:
                    done.add(r["job_id"])
    jid = -1
    for ai in range(len(ARMS)):
        for s in SEEDS:
            jid += 1
            if jid % a.nshards != a.shard or jid in done:
                continue
            t0 = time.perf_counter()
            try:
                hv, nfe = run(ai, s)
                rec = dict(job_id=jid, name=ARMS[ai][0], seed=s, hv=hv, nfe=nfe,
                           secs=time.perf_counter() - t0)
            except Exception as e:
                rec = dict(job_id=jid, name=ARMS[ai][0], seed=s, hv=None, error=repr(e))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[G{a.shard}] {jid:3d} {ARMS[ai][0]:<24} seed={s} "
                  f"hv={rec.get('hv') and round(rec['hv'],4)} {rec.get('secs',0):.0f}s", flush=True)
    print(f"[G{a.shard}] done")


if __name__ == "__main__":
    main()
