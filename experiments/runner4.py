"""Round 3: do the bug fixes and the two reward changes move hypervolume?

n=8 seeds, 1000 epochs x batch 64 = 64,000 NFE per run, matching main.py's real budget
and round 2's synthesis arms. Usage: runner4.py --shard i --nshards N
"""
import sys, os, json, time, argparse, warnings, numpy as np

warnings.filterwarnings("ignore")
# one thread per worker: these are tiny models, and 4 workers x N intra-op threads
# thrashes the cores instead of using them
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import variant

MV = np.load(os.path.join(HERE, "max_values.npy"))
SEEDS = 8
EPOCHS, BATCH = 1000, 64          # 64,000 evaluations

# the three defects from RESULTS.md 5a/5b, plus fp32 so the ratio is exactly 1.0
FIX = {'fix_pe': True, 'fix_position': True, 'dropout': 0.0, 'amp': False}

CONFIGS = [
    ("baseline (repo as-was)",   {}),
    ("bugs fixed",               dict(FIX)),
    ("fixed + unpinned weights", {**FIX, 'pin_constraint_weights': False}),
    ("fixed + hv reward",        {**FIX, 'reward_mode': 'hv'}),
    ("fixed + hv + unpinned",    {**FIX, 'reward_mode': 'hv', 'pin_constraint_weights': False}),
    ("hv reward, bugs left in",  {'reward_mode': 'hv'}),
    # appended after probe8: batch-local archive keeps rho=+0.893 with 23% signal density
    # instead of 5%, and selects the same optimal policy
    ("fixed + hv, batch archive", {**FIX, 'reward_mode': 'hv', 'hv_archive': 'batch'}),
]


# Dropped after measurement: unpinning the constraint weights leaves the reward's rank
# correlation with hypervolume at -0.306 (vs -0.301 pinned), so these two arms were a
# predictable null. Job ids are left untouched so completed work still resumes.
SKIP = {"fixed + unpinned weights", "fixed + hv + unpinned"}


def build_jobs():
    J = []
    for name, cfg in CONFIGS:
        for s in range(SEEDS):
            J.append(dict(kind='ppo', name=name, cfg=cfg, seed=s))
    for s in range(SEEDS):
        J.append(dict(kind='rs', name='random search', cfg={}, seed=s))
    return J


def run_one(j):
    t0 = time.perf_counter()
    if j['kind'] == 'ppo':
        r = variant.run_ppo(j['cfg'], MV, j['seed'], epochs=EPOCHS, batch=BATCH)
        ent = np.array(r['entropy'])
        half = ent[:10].mean() / 2
        below = np.where(ent < half)[0]
        return dict(hv=r['hv'],
                    valid_frac=r['valid_frac'],
                    entropy_final=float(ent[-10:].mean()),
                    entropy_start=float(ent[:10].mean()),
                    collapse_epoch=int(below[0]) if len(below) else -1,
                    ev=float(np.mean(r['ev'][-50:])),
                    wall=time.perf_counter() - t0)
    from optimization.random_search import run_random_search
    np.random.seed(5000 + j['seed'])
    out = run_random_search({'num_epochs': EPOCHS, 'mini_batch_size': BATCH})
    return dict(hv=float(out[4][-1]), wall=time.perf_counter() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()

    jobs = build_jobs()
    outp = os.path.join(HERE, "data", f"r3_shard_{a.shard}.jsonl")
    # completed ids are read from EVERY shard file, so --nshards can change between
    # launches without re-running finished work
    import glob
    done = set()
    for fp in glob.glob(os.path.join(HERE, "data", "r3_shard_*.jsonl")):
        for line in open(fp):
            try:
                done.add(json.loads(line)['job_id'])
            except Exception:
                pass

    with open(outp, "a") as f:
        for i, j in enumerate(jobs):
            if i % a.nshards != a.shard or i in done or j['name'] in SKIP:
                continue
            try:
                res = run_one(j)
                rec = dict(job_id=i, kind=j['kind'], name=j['name'], seed=j['seed'], **res)
                msg = f"hv={res['hv']:.4f} {res['wall']:.0f}s"
            except Exception as e:
                rec = dict(job_id=i, kind=j['kind'], name=j['name'], seed=j['seed'], error=repr(e))
                msg = "ERROR " + repr(e)[:80]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[shard {a.shard}] {i:3d} {j['name']:<26} seed={j['seed']} {msg}", flush=True)
