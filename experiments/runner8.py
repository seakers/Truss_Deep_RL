"""Round 4, track C: hyperparameters re-tested on top of the learnable reward.

Round 1 tested critic width, entropy bonus, actor width and gamma/lambda against the
*scalarised* reward, whose rank correlation with hypervolume is -0.30. Every one of them
was a null. That test is not informative about the batch-local HVI reward, where the
critic's explained variance is -0.065 instead of -3.5e6: with a reward the value function
can actually fit, the components that feed off the value function have something to do.

Two of these are motivated by the reward's shape rather than by round 1:

  target_kl      the ratio is valid now that the PE/position/dropout defects are fixed, so
                 the KL that truncated the actor to one update per epoch was measuring
                 policy change rather than numerical noise. Loosening it is only meaningful
                 after the fixes.
  gamma=lam=1    the HVI reward is paid once, on the terminal step of a 72-action episode.
                 At gamma=0.999, lam=0.9 the GAE weight on the first action is ~1e-4, so
                 nearly the whole design is trained on almost no signal.

All arms are the round-3 winner plus one change, on the same 8 seeds every other arm uses.
Usage: runner8.py --shard i --nshards N
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
WIN = {**FIX, 'reward_mode': 'hv', 'hv_reset_period': 1}

CONFIGS = [
    ("W + gamma=lam=1",     {**WIN, 'gamma': 1.0, 'lam': 1.0}),
    ("W + target_kl 0.03",  {**WIN, 'target_kl': 0.03}),
    ("W + critic d=64",     {**WIN, 'd_model_critic': 64}),
    ("W + entropy 0.01",    {**WIN, 'entropy_coef': 0.01}),
    ("W + actor d=64",      {**WIN, 'd_model_actor': 64}),
]
SEEDS = 8


def build_jobs():
    return [dict(name=n, cfg=c, seed=s) for n, c in CONFIGS for s in range(SEEDS)]


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
    outp = os.path.join(HERE, "data", f"r4c_shard_{a.shard}.jsonl")
    done = set()
    for fp in glob.glob(os.path.join(HERE, "data", "r4c_shard_*.jsonl")):
        for line in open(fp):
            try:
                done.add(json.loads(line)['job_id'])
            except Exception:
                pass

    with open(outp, "a") as f:
        for i, j in enumerate(jobs):
            if i % a.nshards != a.shard or i in done:
                continue
            try:
                res = run_one(j)
                rec = dict(job_id=i, name=j['name'], seed=j['seed'], **res)
                msg = "hv=%.4f ev=%.3g ent=%.2f %.0fs" % (
                    res['hv'], res['ev_median'], res['entropy_final'], res['wall'])
            except Exception as e:
                rec = dict(job_id=i, name=j['name'], seed=j['seed'], error=repr(e))
                msg = "ERROR " + repr(e)[:120]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print("[C%d] %3d %-22s seed=%d %s" % (a.shard, i, j['name'], j['seed'], msg),
                  flush=True)
