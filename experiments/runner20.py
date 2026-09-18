"""Round 8 -- round 7 rerun with the input-scaling bug fixed.

Original round 7 docstring follows.

Round 7 -- do the two fixes make the operator learn anything?

RESULTS_ROUND5.md measured that the trained mutation operator's pointer head ends at
KL 0.0000 from uniform: it never learned WHICH bar to edit. Two causes were identified:

  #1 it cannot see the physics. The solver computes per-bar stress and threw all 72
     values away. `use_stress=True` feeds utilisation (|stress|/limit) as a second input
     channel, so bars are finally distinguishable.
  #2 its reward punishes episode length. Every edit has negative expected reward, so
     long episodes accumulate worse totals and the policy learned to stop after one
     edit. `reward_shape='potential'` pays Phi(child) - Phi(parent), which telescopes to
     Phi(end) - Phi(start): still dense, but the total no longer depends on step count.

2x2 design so each fix can be attributed separately. The headline measurement is the
pointer's KL from uniform after training -- a direct read of whether the network learned
to discriminate between bars, and independent of the hypervolume metric entirely.
"""
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
WORK = os.path.join(HERE, "r8work")
os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
os.chdir(WORK)

from optimization.design_repair import run_design_repair_ppo

MV = np.load(os.path.join(HERE, "max_values.npy"))
BASE = dict(num_epochs=1000, mini_batch_size=32, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None, max_edits=50)
BASE['lambda'] = 0.9

# Round 7's B/D arms fed the utilisation channel RAW, where the design index (0-49,
# mean |x| 25) outweighed it (mean |x| 0.023) by ~1000x. The physics was numerically
# invisible and the pointer stayed at KL 0.0000, which said nothing about physics and
# everything about input scaling. Actor._normalise_input now maps the design channel to
# [0,1] and standardises utilisation across the bars of each design; measured pointer
# logit spread from the stress channel went 0.0012 -> 0.0312.
#
# Both arms below use the normalised design channel and potential shaping, so E vs F
# isolates the stress channel and nothing else. Round 7's C_shaping (raw design +
# shaping) stays as the control for the normalisation itself.
ARMS = [
    ("E_normdesign", dict(normalise_design=True, reward_shape='potential')),
    ("F_stress",     dict(use_stress=True,       reward_shape='potential')),
]
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r8_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip() and json.loads(l).get("hv") is not None:
                done.add(json.loads(l)["job_id"])
    jid = -1
    for nm, cfg in ARMS:
        for s in SEEDS:
            jid += 1
            if jid % a.nshards != a.shard or jid in done:
                continue
            tag = f"{nm}_t{s}"
            os.makedirs(os.path.join(WORK, "results", tag), exist_ok=True)
            t0 = time.perf_counter()
            try:
                torch.manual_seed(100 + s); np.random.seed(100 + s)
                r = run_design_repair_ppo(MV, dict(BASE, date_str=tag, **cfg))
                rec = dict(job_id=jid, name=nm, seed=s, tag=tag, hv=r[4][-1], nfe=r[5],
                           secs=time.perf_counter() - t0)
            except Exception as e:
                rec = dict(job_id=jid, name=nm, seed=s, tag=tag, hv=None, error=repr(e))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[R8-{a.shard}] {jid:2d} {nm:<18} s={s} "
                  f"hv={rec.get('hv') and round(rec['hv'],4)} {rec.get('secs',0):.0f}s", flush=True)
    print(f"[R8-{a.shard}] done")


if __name__ == "__main__":
    main()
