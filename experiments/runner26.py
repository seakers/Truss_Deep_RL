"""Round 9, experiment X -- does the hv reward starve the synthesis PPO of signal?

Experiment V measured that the synthesis PPO generates only **7.9%** feasible designs,
against 65.5% for random search and 96-99% for the tuned GA. That is where its 64,000
evaluations go.

The reason is visible in one line. Under `reward_mode='hv'`
(optimization/ppo_rand_weights_state.py:246):

    step_reward = 0.0 if constraints else archive.offer(to_hv(obj_values))[1] * hv_scale

**Every infeasible design receives exactly 0.0.** Not a small reward, not a graded penalty
-- zero, identically, for all of them. With 92% of generated designs infeasible, roughly
92% of episodes return the same constant and carry no information about how close the
design was to feasibility, or which of two infeasible designs was better.

The alternative reward mode already in the repository does not have this property. Under
`reward_mode='scalarized'` an infeasible design still scores through the two constraint
dimensions, which are pinned to ~58% of the weight mass, so a design that violates its
limits by a little scores better than one that violates them by a lot. There is a gradient
toward feasibility.

That gradient was removed when the hv reward was adopted. This measures what removing it
cost, on two axes:

  hv          -- does the scalarized reward actually score better?
  valid_frac  -- does it fix the feasibility collapse, which is the mechanism claimed?

The second matters even if the first comes out flat. If scalarized lifts feasibility a lot
without lifting hypervolume, the diagnosis is right and the bottleneck is elsewhere; if it
lifts neither, the diagnosis is wrong and the 7.9% is a symptom rather than a cause.

Same 64,000 evaluations, the lab's own max_values.npy, and matched seeds.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
WORK = os.path.join(HERE, "r9xwork")
os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
os.chdir(WORK)

from optimization.ppo_rand_weights_state import run_ppo_rand_weights_state

MV = np.load(os.path.join(HERE, "max_values.npy"))
REF_ROW = MV[:3].copy()
REF_ROW[2] = 0.0

BASE = dict(num_epochs=1000, mini_batch_size=64, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None)
BASE['lambda'] = 0.9

ARMS = [
    ("hv reward (current default)", dict(reward_mode='hv', hv_reset_period=3)),
    ("scalarized reward",           dict(reward_mode='scalarized')),
]
SEEDS = [0, 1, 2]
JOBS = [(ai, s) for ai in range(len(ARMS)) for s in SEEDS]


def valid_fraction(all_obj):
    """Infeasible designs have their objectives overwritten with this exact row before
    hypervolume, so feasibility is recoverable from the returned objectives for free."""
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
    out = os.path.join(HERE, "data", f"r9x_shard_{a.shard}.jsonl")
    for j, (ai, seed) in enumerate(JOBS):
        if j % a.nshards != a.shard:
            continue
        label, extra = ARMS[ai]
        tag = f"x_{ai}_s{seed}"
        os.makedirs(os.path.join(WORK, "results", tag), exist_ok=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        p = dict(BASE); p.update(extra); p['date_str'] = tag
        t0 = time.time()
        des, obj, pfd, pfo, hvs, nfe, _ = run_ppo_rand_weights_state(MV, p, run_idx=seed)
        rec = dict(job_id=j, name=label, seed=seed,
                   hv=float(hvs[-1]) if len(hvs) else 0.0, nfe=int(nfe),
                   front_size=int(len(pfo[-1])) if len(pfo) and pfo[-1] is not None else 0,
                   valid_frac=valid_fraction(obj),
                   unique_frac=float(len(set(map(tuple, np.asarray(des).tolist()))))
                               / max(len(des), 1),
                   secs=time.time() - t0)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{j}] {label} seed={seed} hv={rec['hv']:.4f} "
              f"valid={rec['valid_frac']:.3f} {rec['secs']:.0f}s", flush=True)
