"""Round 4, track E: the hybrid at the real budget, and more actors behind the claim.

Track B established two things at 16,000 NFE: a trained repair operator beats a random one
(+0.0588, p=0.021, 5/5 actors) once `obs_mode='design'` makes the rollout and update states
agree, and the hybrid family beats every synthesis-PPO configuration at 64,000 NFE
(0.7301 untrained vs 0.6023 for the best PPO arm).

What is still missing is the intersection of the two: does the *learned* operator keep its
edge at 64,000 NFE, or is it an advantage that only shows up when the budget is small?
That is the number the project's headline claim actually needs.

Phase 1 needs no new training -- it reuses track B's five DOM_DES actors -- so it runs
first. Phase 2 doubles the actor count behind the p=0.029.

Runs on CPU: the repair actor rolls out one design at a time, which measured 2.5x faster
on CPU than on the GPU (and the GPU is busy with track D).

Usage: runner10.py --shard i --nshards N
"""
import sys, os, json, time, argparse, warnings, glob, numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import torch
torch.set_num_threads(1)
torch.cuda.is_available = lambda: False        # see runner7.py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
WORK = os.path.join(HERE, "r4work")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

import repair_variant
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga

MV = np.load(os.path.join(HERE, "max_values.npy"))

OLD_ACTORS = 5          # DOM_DES t0..t4, already trained by runner7
NEW_ACTORS = 5          # DOM_DES t5..t9, trained here -> 10 actors total
BASE = dict(mini_batch_size=32, gamma=0.999, learning_rate=3e-4, clip_ratio=0.05,
            update_iterations=5, target_kl=0.003, chunk_size=256, model_folder=None)
BASE['lambda'] = 0.9
DOM_DES = {'reward': 'domination', 'obs_mode': 'design'}


def actor_path(t):
    return os.path.join(WORK, "results", f"DOM_DES_t{t}", "actor_repair_model.pth")


def build_p1():
    """64,000-NFE hybrid at both operators, paired on GA seed. Track B's phase 3 already
    measured random init on k=0..5, so only k=6..11 is new there."""
    J = [dict(kind='imga64', name='DOM_DES @64k', tseed=k % OLD_ACTORS, k=k)
         for k in range(12)]
    J += [dict(kind='imga64', name='random init @64k', tseed=None, k=k)
          for k in range(6, 12)]
    return J


def build_p2():
    return [dict(kind='train', tseed=t) for t in range(OLD_ACTORS, OLD_ACTORS + NEW_ACTORS)]


def build_p3():
    """16,000-NFE runs for the five new actors, on GA seeds k=10..19 so they do not
    collide with track B's k=0..9."""
    J = []
    for k in range(10, 20):
        J.append(dict(kind='imga', name='DOM_DES', tseed=OLD_ACTORS + (k - 10) // 2, k=k))
        J.append(dict(kind='imga', name='random init', tseed=None, k=k))
    return J


def run_imga(j, epochs):
    t0 = time.perf_counter()
    np.random.seed(700 + j['k']); torch.manual_seed(700 + j['k'])
    p = dict(BASE); p['num_epochs'] = epochs
    if j['tseed'] is None:
        p['random_init_actor'] = True; p['date_str'] = 'NONE'
    else:
        p['random_init_actor'] = False; p['date_str'] = f"DOM_DES_t{j['tseed']}"
    out = run_intelligent_mutation_ga(MV, p)
    return dict(hv=float(out[4][-1]), NFE=out[5], wall=time.perf_counter() - t0)


def run_train(j):
    t0 = time.perf_counter()
    r = repair_variant.run_repair({**DOM_DES, 'save_actor': actor_path(j['tseed'])},
                                  MV, seed=100 + j['tseed'], epochs=1000, batch=32)
    return dict(hv=r['hv'], NFE=r['NFE'], valid_frac=r['valid_frac'],
                reward_first=float(np.mean(r['reward'][:20])),
                reward_last=float(np.mean(r['reward'][-20:])),
                wall=time.perf_counter() - t0)


def load_done():
    d = set()
    for fp in glob.glob(os.path.join(HERE, "data", "r4e_shard_*.jsonl")):
        for line in open(fp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if 'error' not in r:
                d.add(r['job_id'])
    return d


def wait_for(paths, timeout=6000):
    t0 = time.time()
    while time.time() - t0 < timeout:
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            return True
        print(f"  ... barrier: {len(missing)} actors", flush=True)
        time.sleep(30)
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    a = ap.parse_args()

    f = open(os.path.join(HERE, "data", f"r4e_shard_{a.shard}.jsonl"), "a")

    def do(jobs, fn, off, label, skip):
        for i, j in enumerate(jobs):
            gid = off + i
            if gid % a.nshards != a.shard or gid in skip:
                continue
            common = dict(job_id=gid, kind=j['kind'], name=j.get('name'),
                          tseed=j.get('tseed'), k=j.get('k'))
            try:
                rec = dict(common, **fn(j))
                msg = "hv=%.4f %.0fs" % (rec['hv'], rec['wall'])
            except Exception as e:
                rec = dict(common, error=repr(e))
                msg = "ERROR " + repr(e)[:120]
            f.write(json.dumps(rec) + "\n"); f.flush()
            print("[E%d] %s %4d %-18s t=%s k=%s %s"
                  % (a.shard, label, gid, j.get('name', 'train'), j.get('tseed'),
                     j.get('k'), msg), flush=True)

    do(build_p1(), lambda j: run_imga(j, 2000), 0, "p1", load_done())
    do(build_p2(), run_train, 1000, "p2", load_done())
    wait_for([actor_path(t) for t in range(OLD_ACTORS, OLD_ACTORS + NEW_ACTORS)])
    do(build_p3(), lambda j: run_imga(j, 500), 2000, "p3", load_done())
    f.close()
    print("[E%d] done" % a.shard, flush=True)
