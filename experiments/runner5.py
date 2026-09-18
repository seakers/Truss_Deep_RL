"""Round 4: does goal-conditioning beat the batch-archive HV reward?

Same budget and same 8 seeds as runner4 (1000 epochs x batch 64 = 64,000 NFE), so the
r3 arms are directly comparable without re-running them. Usage: runner5.py --shard i --nshards N
"""
import sys, os, json, time, glob, argparse, warnings, numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from optimization.ppo_goal_conditioned import run_ppo_goal_conditioned

MV = np.load(os.path.join(HERE, "max_values.npy"))
SEEDS = 8
EPOCHS, BATCH = 1000, 64

BASE = dict(num_epochs=EPOCHS, mini_batch_size=BATCH, gamma=0.999, **{'lambda': 0.9},
            learning_rate=3e-4, clip_ratio=0.05, update_iterations=5, target_kl=0.003,
            chunk_size=256, model_folder=None,
            # the bug fixes from round 3, which that sweep showed are what let PPO beat
            # random search at all (p=0.022)
            dropout=0.0, use_amp=False)

CONFIGS = [
    # the reward as designed: dense goal-reaching + sparse front bonus
    ("goal-conditioned",              dict(goal_coef=1.0, front_bonus=1.0, hv_scale=0.0)),
    # + the batch-archive HV term that won round 3, to see if they compose
    ("goal-conditioned + hv term",    dict(goal_coef=1.0, front_bonus=1.0, hv_scale=200.0)),
    # ablation: is the goal signal doing anything, or is it all the front bonus?
    ("front bonus only (no goals)",   dict(goal_coef=0.0, front_bonus=1.0, hv_scale=0.0)),
]


def run_one(j):
    t0 = time.perf_counter()
    seed = j['seed']
    np.random.seed(1000 + seed)
    torch.manual_seed(1000 + seed)
    date_str = f"r4_{j['tag']}_s{seed}"
    os.makedirs(f"results/{date_str}", exist_ok=True)
    p = {**BASE, **j['cfg'], 'date_str': date_str, 'save_path': f"results/{date_str}/"}
    out = run_ppo_goal_conditioned(MV, p, run_idx=seed)
    all_obj, all_constraints = out[1], None
    return dict(hv=float(out[6]),
                valid_frac=float(np.mean(out[1][:, 0] != MV[0])),
                nfe=int(out[5]),
                wall=time.perf_counter() - t0)


def build_jobs():
    J = []
    for i, (name, cfg) in enumerate(CONFIGS):
        for s in range(SEEDS):
            J.append(dict(name=name, tag=f"c{i}", cfg=cfg, seed=s))
    return J


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()

    jobs = build_jobs()
    done = set()
    for fp in glob.glob(os.path.join(HERE, "data", "r4_shard_*.jsonl")):
        for line in open(fp):
            try: done.add(json.loads(line)['job_id'])
            except Exception: pass

    outp = os.path.join(HERE, "data", f"r4_shard_{a.shard}.jsonl")
    with open(outp, "a") as f:
        for i, j in enumerate(jobs):
            if i % a.nshards != a.shard or i in done:
                continue
            try:
                res = run_one(j)
                rec = dict(job_id=i, name=j['name'], seed=j['seed'], **res)
                msg = f"hv={res['hv']:.4f} {res['wall']:.0f}s"
            except Exception as e:
                import traceback; traceback.print_exc()
                rec = dict(job_id=i, name=j['name'], seed=j['seed'], error=repr(e))
                msg = "ERROR " + repr(e)[:100]
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[shard {a.shard}] {i:3d} {j['name']:<28} seed={j['seed']} {msg}", flush=True)
