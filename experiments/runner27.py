"""Round 9, experiment XI -- can a graded infeasibility penalty get the best of both?

Experiment X measured a clean trade-off with no middle ground tested:

  reward_mode='hv'          hv 0.6071   feasible 10.3%
  reward_mode='scalarized'  hv 0.4197   feasible 75.4%

The hv reward scores better but is blind among infeasible designs (they all get exactly
0.0). The scalarized reward can tell them apart and scores far worse, because it keeps the
policy safely inside the feasible region while the best trade-offs sit on the boundary.

Neither is obviously the right design. The hv reward's blindness is real -- ~90% of
episodes return the same constant -- but removing it by switching reward modes throws away
the hypervolume objective at the same time. `infeasible_penalty` (added to
ppo_rand_weights_state.py, default 0.0 so existing behaviour is unchanged) separates those:
keep the hypervolume reward for feasible designs, and give infeasible ones a small graded
penalty proportional to total constraint violation instead of a flat zero.

  step_reward = -infeasible_penalty * sum(constraint_vals)      if infeasible
                 archive.offer(...)  * hv_scale                 if feasible

Two penalty sizes against the current default of zero. Small enough that boundary-probing
is still worth it, large enough that a near-miss beats a disaster. If neither beats
penalty=0, the flat zero is vindicated as a design choice rather than merely inherited, and
that is worth knowing explicitly.

The penalties are chosen relative to the reward scale actually in play, not swept: the
constraint values are already squashed into [0,1) by truss_problem.solve, and hv_scale is
200, so 0.5 and 2.0 make a typical violation cost roughly the same as, and a few times
more than, a small hypervolume contribution.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
WORK = os.path.join(HERE, "r9ywork")
os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
os.chdir(WORK)

from optimization.ppo_rand_weights_state import run_ppo_rand_weights_state

MV = np.load(os.path.join(HERE, "max_values.npy"))
REF_ROW = MV[:3].copy()
REF_ROW[2] = 0.0

BASE = dict(num_epochs=1000, mini_batch_size=64, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None, reward_mode='hv', hv_reset_period=3)
BASE['lambda'] = 0.9

ARMS = [
    ("penalty 0.0 (current default)", 0.0),
    ("penalty 0.5",                   0.5),
    ("penalty 2.0",                   2.0),
]
SEEDS = [0, 1, 2]
JOBS = [(ai, s) for ai in range(len(ARMS)) for s in SEEDS]


def valid_fraction(all_obj):
    a = np.asarray(all_obj, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0:
        return float('nan')
    hit = np.all(np.isclose(a, REF_ROW[None, :], rtol=0, atol=1e-9), axis=1)
    return float(np.mean(~hit))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r9y_shard_{a.shard}.jsonl")
    for j, (ai, seed) in enumerate(JOBS):
        if j % a.nshards != a.shard:
            continue
        label, pen = ARMS[ai]
        tag = f"y_{ai}_s{seed}"
        os.makedirs(os.path.join(WORK, "results", tag), exist_ok=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        p = dict(BASE); p['infeasible_penalty'] = pen; p['date_str'] = tag
        t0 = time.time()
        des, obj, pfd, pfo, hvs, nfe, _ = run_ppo_rand_weights_state(MV, p, run_idx=seed)
        rec = dict(job_id=j, name=label, penalty=pen, seed=seed,
                   hv=float(hvs[-1]) if len(hvs) else 0.0, nfe=int(nfe),
                   front_size=int(len(pfo[-1])) if len(pfo) and pfo[-1] is not None else 0,
                   valid_frac=valid_fraction(obj), secs=time.time() - t0)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{j}] {label} seed={seed} hv={rec['hv']:.4f} "
              f"valid={rec['valid_frac']:.3f} {rec['secs']:.0f}s", flush=True)
