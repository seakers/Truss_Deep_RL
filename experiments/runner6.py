"""Round 4, track A: what is the batch-local HVI archive actually buying?

Round 3 found that resetting the HVI archive every epoch turns an unlearnable reward
into a learnable one (HV 0.5596 vs 0.4303 baseline, p=0.0045, n=8). This sweep asks
which property of that reset is responsible, and confirms the winner at n=16.

  reset period      1 / 5 / 25 / never   -- how stale may the archive be?
  hv_binary         front membership instead of HVI magnitude
  batch size        32 / 64 / 128 at fixed 64,000 NFE -- is it the number of competitors?

Usage: runner6.py --shard i --nshards N
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
NFE = 64000

FIX = {'fix_pe': True, 'fix_position': True, 'dropout': 0.0, 'amp': False}
HV1 = {**FIX, 'reward_mode': 'hv', 'hv_reset_period': 1}

# (name, cfg, seeds, epochs, batch). Seeds 8-15 extend the two round-3 arms that the
# headline comparison rests on to n=16; everything new is measured on seeds 0-7, which
# are the seeds every previous arm already has, so the tests stay paired.
CONFIGS = [
    ("batch archive (seeds 8-15)", HV1,                          range(8, 16), 1000, 64),
    ("baseline (seeds 8-15)",      {},                           range(8, 16), 1000, 64),
    ("reset every 5",              {**HV1, 'hv_reset_period': 5},  range(8),   1000, 64),
    ("reset every 25",             {**HV1, 'hv_reset_period': 25}, range(8),   1000, 64),
    ("batch front membership",     {**HV1, 'hv_binary': True},     range(8),   1000, 64),
    ("batch archive, batch=32",    HV1,                            range(8),   2000, 32),
    ("batch archive, batch=128",   HV1,                            range(8),   500, 128),
]


def build_jobs():
    J = []
    for name, cfg, seeds, ep, ba in CONFIGS:
        for s in seeds:
            J.append(dict(name=name, cfg=cfg, seed=int(s), epochs=ep, batch=ba))
    return J


def run_one(j):
    t0 = time.perf_counter()
    r = variant.run_ppo(j['cfg'], MV, j['seed'], epochs=j['epochs'], batch=j['batch'])
    ent = np.array(r['entropy']); ev = np.array(r['ev'])
    half = ent[:10].mean() / 2
    below = np.where(ent < half)[0]
    return dict(hv=r['hv'],
                valid_frac=r['valid_frac'],
                entropy_final=float(ent[-10:].mean()),
                entropy_start=float(ent[:10].mean()),
                collapse_epoch=int(below[0]) if len(below) else -1,
                ev=float(np.mean(ev[-50:])),
                ev_median=float(np.median(ev[-50:])),
                # every 10th epoch, so the seed-7 style critic blow-ups in round 3 can be
                # located in time instead of only noticed in the final average
                ev_curve=[float(x) for x in ev[::10]],
                kl_mean=float(np.mean(r['kl'])),
                wall=time.perf_counter() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()

    jobs = build_jobs()
    outp = os.path.join(HERE, "data", f"r4a_shard_{a.shard}.jsonl")
    done = set()
    for fp in glob.glob(os.path.join(HERE, "data", "r4a_shard_*.jsonl")):
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
                msg = f"hv={res['hv']:.4f} ev={res['ev_median']:.3g} {res['wall']:.0f}s"
            except Exception as e:
                rec = dict(job_id=i, name=j['name'], seed=j['seed'], error=repr(e))
                msg = "ERROR " + repr(e)[:120]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[A{a.shard}] {i:3d} {j['name']:<28} seed={j['seed']:2d} {msg}", flush=True)
