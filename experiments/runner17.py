"""Round 6, step 3 -- re-baseline every method on the repaired reference.

Every arm runs through the repo's own function at 64,000 NFE on seeds 0-7, against the
rebuilt max_values.npy. Numbers are NOT comparable to rounds 1-5, which used a reference
80% low on the frequency axis.

Each method is run in its CURRENT repo configuration (so the tuned GA gets its new
defaults and IMGA keeps its historical elitism 0 / sss), plus the original GA settings
as the historical baseline. IMGA inherits the same missing-elitism flaw the GA had;
that is noted in the writeup rather than silently fixed here, because changing it would
make the arm something other than the method rounds 1-5 measured.

--device gpu selects the synthesis PPO only; run it with fewer shards (one card).
"""
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
WORK = os.path.join(HERE, "r4work")
os.chdir(WORK)                                  # IMGA loads results/<date_str>/ relatively

from optimization.random_search import run_random_search
from optimization.genetic_algorithm import run_genetic_algorithm
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga
from optimization.ppo_rand_weights_state import run_ppo_rand_weights_state

MV = np.load(os.path.join(HERE, "max_values.npy"))
BASE = dict(num_epochs=1000, mini_batch_size=64, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None, date_str="rebaseline")
BASE['lambda'] = 0.9
os.makedirs(os.path.join(WORK, "results", "rebaseline"), exist_ok=True)

CPU_ARMS = [
    ("random search",                 dict(fn='rs')),
    ("GA original (8 bars, no elit)", dict(fn='ga', mutation_num_genes=8, keep_elitism=0,
                                           parent_selection_type='sss')),
    ("GA tuned (round-6 default)",    dict(fn='ga')),
    ("IMGA untrained actor",          dict(fn='imga', random_init_actor=True)),
    ("IMGA trained actor",            dict(fn='imga', random_init_actor=False)),
]
GPU_ARMS = [
    ("synthesis PPO (hv, reset 3)",   dict(fn='ppo', reward_mode='hv',
                                           hv_reset_period=3, hv_scale=200.0)),
]
SEEDS = list(range(8))


def run(cfg, seed):
    cfg = dict(cfg); fn = cfg.pop('fn')
    p = dict(BASE, **cfg)
    torch.manual_seed(seed); np.random.seed(seed)
    if fn == 'rs':
        r = run_random_search(p)
    elif fn == 'ga':
        r = run_genetic_algorithm(MV, p)
    elif fn == 'imga':
        p['date_str'] = f"DOM_DES_t{seed % 5}"
        r = run_intelligent_mutation_ga(MV, p)
    else:
        r = run_ppo_rand_weights_state(MV, p)
    return r[4][-1], r[5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    a = ap.parse_args()
    arms = CPU_ARMS if a.device == "cpu" else GPU_ARMS
    out = os.path.join(HERE, "data", f"r6c_{a.device}_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip() and json.loads(l).get("hv") is not None:
                done.add(json.loads(l)["job_id"])
    jid = -1
    for nm, cfg in arms:
        for s in SEEDS:
            jid += 1
            if jid % a.nshards != a.shard or jid in done:
                continue
            t0 = time.perf_counter()
            try:
                hv, nfe = run(cfg, s)
                rec = dict(job_id=jid, name=nm, seed=s, hv=hv, nfe=nfe,
                           secs=time.perf_counter() - t0)
            except Exception as e:
                rec = dict(job_id=jid, name=nm, seed=s, hv=None, error=repr(e))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{a.device.upper()}{a.shard}] {jid:3d} {nm:<32} s={s} "
                  f"hv={rec.get('hv') and round(rec['hv'],4)} {rec.get('secs',0):.0f}s", flush=True)
    print(f"[{a.device.upper()}{a.shard}] done")


if __name__ == "__main__":
    main()
