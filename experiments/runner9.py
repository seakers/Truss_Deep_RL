"""Round 4, track D: locate the archive-reset optimum.

Track A found the reset period is not monotone -- resetting the HVI archive every 5
epochs (0.6023) beats every epoch (0.5526), every 25 (0.5335) and never (0.4742). The
interior optimum was not predicted by anything in rounds 1-3, and it contradicts round 3's
stated mechanism: reset=5 wins while its critic's explained variance sits at -6e6, i.e.
diverged, so "the reset makes the reward learnable by the critic" cannot be the reason.

This brackets the optimum (2, 3, 10) and takes the current best to n=16.

Usage: runner9.py --shard i --nshards N
"""
import sys, os, json, time, argparse, warnings, glob, numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import variant

MV = np.load(os.path.join(HERE, "max_values.npy"))
EPOCHS, BATCH = 1000, 64          # 64,000 NFE

FIX = {'fix_pe': True, 'fix_position': True, 'dropout': 0.0, 'amp': False}


def cfg(period):
    return {**FIX, 'reward_mode': 'hv', 'hv_reset_period': period}


# bracket first, then confirm the incumbent -- if the budget runs out, the shape of the
# curve is worth more than a wider confidence interval on a point already at 8/8 seeds
CONFIGS = [
    ("reset every 3",  cfg(3),  range(8)),
    ("reset every 2",  cfg(2),  range(8)),
    ("reset every 10", cfg(10), range(8)),
    ("reset every 5 (seeds 8-15)", cfg(5), range(8, 16)),
]


def build_jobs():
    return [dict(name=n, cfg=c, seed=int(s)) for n, c, ss in CONFIGS for s in ss]


def run_one(j):
    t0 = time.perf_counter()
    r = variant.run_ppo(j['cfg'], MV, j['seed'], epochs=EPOCHS, batch=BATCH)
    ent = np.array(r['entropy']); ev = np.array(r['ev'])
    half = ent[:10].mean() / 2
    below = np.where(ent < half)[0]
    return dict(hv=r['hv'], valid_frac=r['valid_frac'],
                entropy_final=float(ent[-10:].mean()),
                entropy_start=float(ent[:10].mean()),
                collapse_epoch=int(below[0]) if len(below) else -1,
                ev=float(np.mean(ev[-50:])), ev_median=float(np.median(ev[-50:])),
                ev_curve=[float(x) for x in ev[::10]],
                kl_mean=float(np.mean(r['kl'])),
                wall=time.perf_counter() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()

    jobs = build_jobs()
    outp = os.path.join(HERE, "data", f"r4d_shard_{a.shard}.jsonl")
    done = set()
    for fp in glob.glob(os.path.join(HERE, "data", "r4d_shard_*.jsonl")):
        for line in open(fp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if 'error' not in r:
                done.add(r['job_id'])

    with open(outp, "a") as f:
        for i, j in enumerate(jobs):
            if i % a.nshards != a.shard or i in done:
                continue
            try:
                res = run_one(j)
                rec = dict(job_id=i, name=j['name'], seed=j['seed'], **res)
                msg = "hv=%.4f ev=%.3g vf=%.3f %.0fs" % (
                    res['hv'], res['ev_median'], res['valid_frac'], res['wall'])
            except Exception as e:
                rec = dict(job_id=i, name=j['name'], seed=j['seed'], error=repr(e))
                msg = "ERROR " + repr(e)[:120]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print("[D%d] %3d %-28s seed=%2d %s" % (a.shard, i, j['name'], j['seed'], msg),
                  flush=True)
