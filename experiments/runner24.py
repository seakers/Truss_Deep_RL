"""Round 9, experiment V -- does the synthesis PPO's multi-objective conditioning work?

The synthesis PPO scores 0.30 where random search scores 0.40 (round 6 re-baseline). A
method that loses to random sampling is not slightly mistuned; something structural is
wrong, and nobody has checked the most load-bearing assumption in the design.

The method is scalarised multi-objective RL: each episode draws a random weight vector
over the five objective/constraint dimensions, the reward is the weighted sum, and the
policy is supposed to produce a DIFFERENT design depending on the weights it is given --
that is the whole reason the resulting set of designs is expected to spread along a
Pareto front rather than pile up in one place. The repo ships two variants:

  state       -- the weight vector is prepended to the observation, so the policy can see
                 which trade-off it is being asked for. This is the one that is supposed
                 to work, and the only one re-baselined in round 6.
  uninformed  -- the reward still uses the random weights, but the policy never sees
                 them. It is being asked to hit a target it cannot observe.

If the two score the same, the conditioning is decorative: the policy is not using the
weights, every episode is effectively the same non-stationary scalar objective, and the
spread along the front that the method depends on is coming from sampling noise rather
than from control. That would explain losing to random search far better than any
hyperparameter.

Beyond hypervolume this records three diagnostics that separate the plausible mechanisms:

  valid_frac    -- fraction of generated designs that satisfy the constraints. Infeasible
                   designs are all mapped to the same reference point, so a policy that
                   spends its budget there is optimising on a flat plateau with no
                   gradient toward feasibility. The tuned GA sits at 96-99%; if PPO sits
                   far lower, the plateau is the story.
  unique_frac   -- fraction of distinct designs generated. Mode collapse to one design
                   would produce a one-point front and a small hypervolume no matter how
                   good that design is.
  front_size    -- number of non-dominated points found. Directly separates "found a good
                   region but only one point in it" from "found bad designs".

Same 64,000 NFE and the lab's original max_values.npy as every other round-9 arm.
"""
import argparse, json, os, sys, time
from copy import deepcopy

import numpy as np
import torch
from pymoo.indicators.hv import HV


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
WORK = os.path.join(HERE, "r9vwork")
os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
os.chdir(WORK)

from optimization.ppo_rand_weights_state import run_ppo_rand_weights_state
from optimization.ppo_rand_weights_uninformed import run_ppo_rand_weights_uninformed
from optimization.random_search import run_random_search
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress

MV = np.load(os.path.join(HERE, "max_values.npy"))

# 1000 epochs x 64 = 64,000 NFE, matching the round-6 re-baseline and round-9 track C.
BASE = dict(num_epochs=1000, mini_batch_size=64, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None)
BASE['lambda'] = 0.9

ARMS = [
    ("PPO state (sees weights)",     'state',  {}),
    ("PPO uninformed (blind)",       'unin',   {}),
    ("random search (both metrics)", 'rs',     {}),
]
SEEDS = [0, 1, 2, 3, 4, 5]
JOBS = [(ai, s) for ai in range(len(ARMS)) for s in SEEDS]

# Every runner overwrites the objectives of a constraint-violating design with this exact
# vector before computing hypervolume (max value on each minimised axis, 0 on the
# maximised one). So infeasibility is recoverable from the returned objectives for free,
# with no extra solves: a row equal to it is a design that violated a constraint.
REF_ROW = MV[:3].copy()
REF_ROW[2] = 0.0


def valid_fraction(all_obj):
    a = np.asarray(all_obj, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0:
        return float('nan')
    hit = np.all(np.isclose(a, REF_ROW[None, :], rtol=0, atol=1e-9), axis=1)
    return float(np.mean(~hit))


def random_search_both_metrics(seed):
    """Random search scored on BOTH references from a single sample.

    optimization/random_search.py:48 builds its own reference from the run's own results
    (`max_values = max(all_obj) * 1.1`) instead of taking the shared max_values.npy that
    every other method is normalised by. Its hypervolume is therefore computed on a
    different scale from the numbers it gets compared against -- it is graded on a curve
    fitted to its own sample. The round-6 re-baseline reported "random search 0.4008 beats
    synthesis PPO 0.3000" on exactly that mismatch.

    This reproduces the shipped sampling exactly, then scores the same designs twice: once
    the shipped way, once against the shared reference. Being the same sample, the
    difference is purely the metric, with no sampling noise between the two numbers.
    """
    prob = TrussProblem()
    n_obj = prob.num_objectives
    min_mask = np.array([o.lower() == 'min' for o in prob.objective_min_max])
    hv_ind = HV(ref_point=np.ones(n_obj))
    n = BASE['num_epochs'] * BASE['mini_batch_size']

    rng = np.random.RandomState(seed)
    des, obj, con, cv = [], [], [], []
    for _ in range(n):
        d = rng.randint(0, len(prob.area_map), prob.n_elements).tolist()
        o, ic, c = prob.evaluate(d)
        des.append(d); obj.append(o); con.append(ic); cv.append(c)
    des = np.array(des); obj = np.array(obj)
    con = np.array(con); cv = np.array(cv)

    own = np.max(np.concatenate((obj, cv), axis=1), axis=0) * 1.1 + 1e-6

    def score(max_values):
        mo = max_values[:n_obj]
        ref_values = mo * np.ones(n_obj)
        ref_values[~min_mask[:n_obj]] = 0.0
        o2 = obj.copy()
        o2[con] = ref_values
        norm = o2 / mo
        hv_obj = deepcopy(norm)
        for i in range(n_obj):
            if prob.objective_min_max[i].lower() == 'max':
                hv_obj[:, i] = 1 - norm[:, i]
        pfd, pfo, hvs = track_pareto_progress(hv_obj, des, con, hv_ind)
        # pareto_front_obj is one entry PER EVALUATION (the front as it stood at that
        # point), so the front itself is the last entry, not the length of the list.
        final = pfo[-1] if len(pfo) and pfo[-1] is not None else []
        return (float(hvs[-1]) if len(hvs) else 0.0), len(final)

    hv_own, fs_own = score(own)
    hv_shared, fs_shared = score(MV)
    return dict(hv=hv_shared, hv_own_reference=hv_own, front_size=fs_shared,
                front_size_own=fs_own, nfe=n,
                valid_frac=float(np.mean(~con)),
                unique_frac=float(len(set(map(tuple, des.tolist())))) / max(len(des), 1),
                own_reference=own.tolist(), shared_reference=MV.tolist())


def run(kind, seed, tag):
    np.random.seed(seed)
    torch.manual_seed(seed)
    p = dict(BASE)
    p['date_str'] = tag
    os.makedirs(os.path.join(WORK, "results", tag), exist_ok=True)
    if kind == 'state':
        # hv reward with the round-5 reset period, i.e. the configuration round 6
        # re-baselined -- so this arm reproduces a known number before the comparison
        # rests on it
        p.update(reward_mode='hv', hv_reset_period=3)
        return run_ppo_rand_weights_state(MV, p, run_idx=seed)[:6]
    if kind == 'unin':
        return run_ppo_rand_weights_uninformed(MV, p)[:6]
    return run_random_search(MV, p)[:6]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r9v_shard_{a.shard}.jsonl")
    for j, (ai, seed) in enumerate(JOBS):
        if j % a.nshards != a.shard:
            continue
        label, kind, extra = ARMS[ai]
        tag = f"v_{kind}_s{seed}"
        t0 = time.time()
        if kind == 'rs':
            rec = dict(job_id=j, name=label, kind=kind, seed=seed,
                       secs=0.0, **random_search_both_metrics(seed))
            rec['secs'] = time.time() - t0
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{j}] {label} seed={seed} hv_shared={rec['hv']:.4f} "
                  f"hv_own={rec['hv_own_reference']:.4f} {rec['secs']:.0f}s", flush=True)
            continue
        des, obj, pfd, pfo, hvs, nfe = run(kind, seed, tag)
        des_arr = np.asarray(des)
        rec = dict(job_id=j, name=label, kind=kind, seed=seed,
                   hv=float(hvs[-1]) if len(hvs) else 0.0, nfe=int(nfe),
                   front_size=int(len(pfo[-1])) if len(pfo) and pfo[-1] is not None else 0,
                   valid_frac=valid_fraction(obj),
                   unique_frac=float(len(set(map(tuple, des_arr.tolist())))) / max(len(des_arr), 1),
                   secs=time.time() - t0)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{j}] {label} seed={seed} hv={rec['hv']:.4f} front={rec['front_size']} "
              f"valid={rec['valid_frac']:.3f} uniq={rec['unique_frac']:.3f} "
              f"{rec['secs']:.0f}s", flush=True)
