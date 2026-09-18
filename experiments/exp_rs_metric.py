"""Quantify the random-search metric bug on its own, at full budget.

optimization/random_search.py:48 normalises hypervolume by maxima computed from the run's
OWN sample instead of the shared max_values.npy that every other method uses. This scores
one identical sample of 64,000 random designs both ways, so the difference is purely the
metric with no sampling noise between the two numbers.

Split out of runner24.py because it is pure numpy and finishes in minutes, while that
runner's PPO arms take hours and would have kept this queued behind them.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(HERE))

from runner24 import random_search_both_metrics

if __name__ == '__main__':
    out = os.path.join(HERE, "data", "r9rs.jsonl")
    open(out, 'w').close()
    for seed in range(6):
        t0 = time.time()
        rec = random_search_both_metrics(seed)
        rec.update(seed=seed, secs=time.time() - t0)
        rec.pop('own_reference', None)
        rec.pop('shared_reference', None)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"seed={seed} shared={rec['hv']:.4f} own={rec['hv_own_reference']:.4f} "
              f"{rec['secs']:.0f}s", flush=True)
