"""Round 4, track B: does the reinforcement learning contribute anything to the hybrid?

Round 2 concluded it does not, on the strength of a single trained actor reused across
8 GA seeds. Two things are re-done here:

1. **The actor is treated as the random variable it is.** 5 independently trained actors
   per configuration x 2 GA seeds each, instead of 1 actor x 8 GA seeds, so the test is
   over training runs rather than over GA noise.

2. **A rollout/update observation mismatch is fixed.** `design_repair.py` rolls the policy
   out on a 72-token design (`sample_action(current_design)`) but runs `ppo_update` on the
   77-token `design + objectives + constraint_vals` state. The importance ratio is
   therefore computed at states the policy never acted in, and
   `intelligent_mutation_ga.py` later queries it with 72 tokens again. `obs_mode='design'`
   makes rollout, update and deployment agree.

Phase 3 measures the hybrid arms at the synthesis budget (64,000 NFE), which no round has
done: every hybrid number so far is at 16,000 NFE and so is not comparable with the
synthesis PPO results the recommendations rest on. It runs last because it is the most
expensive per job and the least likely to change a conclusion.

Usage: runner7.py --shard i --nshards N
"""
import sys, os, json, time, argparse, warnings, glob, numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import torch
torch.set_num_threads(1)

# Track A already holds 8 CUDA contexts on the one GPU; adding these workers exhausted
# its memory ("CUDA error: out of memory" on every job). These models are 16 wide and
# roll out one design at a time, so the GPU was buying nothing anyway -- a 1,280-NFE
# benchmark ran 71s on CPU against 65s on GPU. CUDA_VISIBLE_DEVICES is not honoured
# here, so the availability check the repo calls at runtime is stubbed instead.
if os.environ.get("R4B_DEVICE", "cpu") == "cpu":
    torch.cuda.is_available = lambda: False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
WORK = os.path.join(HERE, "r4work")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)                # intelligent_mutation_ga loads results/<tag>/... relatively

import repair_variant
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga
from optimization.genetic_algorithm import run_genetic_algorithm

MV = np.load(os.path.join(HERE, "max_values.npy"))

TRAIN_SEEDS = 5          # independently trained actors per configuration
GA_PER_ACTOR = 2         # GA seeds per actor -> 10 runs per arm
GA64_SEEDS = 6
GA_EPOCHS, GA_BATCH = 500, 32        # 16,000 NFE, matching round 2's hybrid arms

BASE_GA = dict(num_epochs=GA_EPOCHS, mini_batch_size=GA_BATCH, gamma=0.999,
               learning_rate=3e-4, clip_ratio=0.05, update_iterations=5,
               target_kl=0.003, chunk_size=256, model_folder=None)
BASE_GA['lambda'] = 0.9

# (tag, repair cfg). 'full' == the repo's mismatched update state, 'design' == fixed.
# HVG_FULL (global hv reward on the mismatched state, i.e. round 2's exact "long-hv"
# config) was dropped for budget; DOM_FULL keeps one replication of round 2's setup.
TRAIN_CFGS = [
    ("DOM_FULL", {'reward': 'domination', 'obs_mode': 'full'}),
    ("DOM_DES",  {'reward': 'domination', 'obs_mode': 'design'}),
    ("HVG_DES",  {'reward': 'hv',         'obs_mode': 'design'}),
    ("HVB_DES",  {'reward': 'hv',         'obs_mode': 'design', 'hv_reset_period': 1}),
]


def actor_path(tag, t):
    return os.path.join(WORK, "results", f"{tag}_t{t}", "actor_repair_model.pth")


def build_p1():
    """Repair-actor training, 32,000 NFE each."""
    return [dict(kind='train', tag=tag, cfg=cfg, tseed=t)
            for tag, cfg in TRAIN_CFGS for t in range(TRAIN_SEEDS)]


def build_p2():
    """IMGA at 16,000 NFE with each operator. Run k uses GA seed 700+k for every arm,
    so arms are paired on GA randomness; k // GA_PER_ACTOR selects the actor."""
    J = [dict(kind='imga', name='random init', tag=None, tseed=None, k=k)
         for k in range(TRAIN_SEEDS * GA_PER_ACTOR)]
    for tag, _ in TRAIN_CFGS:
        for k in range(TRAIN_SEEDS * GA_PER_ACTOR):
            J.append(dict(kind='imga', name=tag, tag=tag, tseed=k // GA_PER_ACTOR, k=k))
    return J


def build_p3():
    """The hybrid arms at the synthesis budget, so the two families are comparable."""
    J = [dict(kind='ga64', name='IMGA untrained @64k', rnd=True, seed=s)
         for s in range(GA64_SEEDS)]
    J += [dict(kind='ga64', name='plain GA @64k', rnd=None, seed=s)
          for s in range(GA64_SEEDS)]
    return J


def run_p1(j):
    t0 = time.perf_counter()
    r = repair_variant.run_repair(
        {**j['cfg'], 'save_actor': actor_path(j['tag'], j['tseed'])},
        MV, seed=100 + j['tseed'], epochs=1000, batch=32)              # 32,000 NFE
    return dict(hv=r['hv'], NFE=r['NFE'], valid_frac=r['valid_frac'],
                reward_first=float(np.mean(r['reward'][:20])),
                reward_last=float(np.mean(r['reward'][-20:])),
                wall=time.perf_counter() - t0)


def run_p2(j):
    t0 = time.perf_counter()
    np.random.seed(700 + j['k']); torch.manual_seed(700 + j['k'])
    p = dict(BASE_GA)
    if j['tag'] is None:
        p['random_init_actor'] = True; p['date_str'] = 'NONE'
    else:
        p['random_init_actor'] = False; p['date_str'] = f"{j['tag']}_t{j['tseed']}"
    out = run_intelligent_mutation_ga(MV, p)
    return dict(hv=float(out[4][-1]), NFE=out[5], wall=time.perf_counter() - t0)


def run_p3(j):
    t0 = time.perf_counter()
    np.random.seed(700 + j['seed']); torch.manual_seed(700 + j['seed'])
    p = dict(BASE_GA); p['num_epochs'] = 2000                          # 64,000 NFE
    if j['rnd'] is None:
        out = run_genetic_algorithm(MV, p)
    else:
        p['random_init_actor'] = True; p['date_str'] = 'NONE'
        out = run_intelligent_mutation_ga(MV, p)
    return dict(hv=float(out[4][-1]), NFE=out[5], wall=time.perf_counter() - t0)


def wait_for_actors(timeout=9000):
    """Barrier: phase 2 needs every actor, not just the ones this shard trained."""
    need = [actor_path(tag, t) for tag, _ in TRAIN_CFGS for t in range(TRAIN_SEEDS)]
    t0 = time.time()
    while time.time() - t0 < timeout:
        missing = [p for p in need if not os.path.exists(p)]
        if not missing:
            return True
        print(f"  ... barrier: waiting on {len(missing)} actors", flush=True)
        time.sleep(30)
    print("  ... barrier TIMED OUT; running what is available", flush=True)
    return False


def load_done():
    d = set()
    for fp in glob.glob(os.path.join(HERE, "data", "r4b_shard_*.jsonl")):
        for line in open(fp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            # a failed job is not done: an OOM or a transient fault must be retried on
            # the next launch, not silently skipped forever
            if 'error' not in r:
                d.add(r['job_id'])
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()

    outp = os.path.join(HERE, "data", f"r4b_shard_{a.shard}.jsonl")
    f = open(outp, "a")

    def do(jobs, fn, off, label, skip):
        for i, j in enumerate(jobs):
            gid = off + i
            if gid % a.nshards != a.shard or gid in skip:
                continue
            nm = j.get('name') or j.get('tag')
            common = dict(job_id=gid, kind=j['kind'], name=nm, tag=j.get('tag'),
                          tseed=j.get('tseed'), k=j.get('k'), seed=j.get('seed'))
            try:
                res = fn(j)
                rec = dict(common, **res)
                msg = "hv=%.4f %.0fs" % (res['hv'], res['wall'])
            except Exception as e:
                rec = dict(common, error=repr(e))
                msg = "ERROR " + repr(e)[:120]
            f.write(json.dumps(rec) + "\n"); f.flush()
            print("[B%d] %s %4d %-22s t=%s k=%s s=%s %s"
                  % (a.shard, label, gid, nm, j.get('tseed'), j.get('k'),
                     j.get('seed'), msg), flush=True)

    do(build_p1(), run_p1, 0, "p1", load_done())
    wait_for_actors()
    do(build_p2(), run_p2, 1000, "p2", load_done())
    do(build_p3(), run_p3, 2000, "p3", load_done())
    f.close()
    print("[B%d] done" % a.shard, flush=True)
