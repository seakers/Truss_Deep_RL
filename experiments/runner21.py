"""Round 9, track A -- deployment: is the physics-tilted operator worth its cost?

Two questions in one table.

1. Does the learned operator beat a random one? Rounds 5-6 said the apparent gain was a
   mutation-rate artifact. Round 8 gave the pointer real physics-aligned signal
   (corr(|util|,p) -0.17 -> -0.73, p=0.0007), so the question is live again -- but the
   stress operator must pay for the extra solves it needs, which are counted as NFE.
2. Does ANY of it survive a GA that actually works? Every hybrid number in rounds 1-6
   came from a GA with keep_elitism=0 and sss selection, which round 6 measured as
   costing ~0.28 hypervolume. Each operator is therefore run under BOTH the historical
   settings and the tuned ones.

All at 64,000 NFE against the lab's original max_values.npy.
"""
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
WORK = os.path.join(HERE, "r9work")
os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
# actors live in r7work/r8work; symlink-free approach: copy the folders we need once
import shutil
for src in ("r7work", "r8work"):
    d = os.path.join(HERE, src, "results")
    if os.path.isdir(d):
        for tag in os.listdir(d):
            dst = os.path.join(WORK, "results", tag)
            if not os.path.isdir(dst) and os.path.isfile(os.path.join(d, tag, "actor_repair_model.pth")):
                shutil.copytree(os.path.join(d, tag), dst)
os.chdir(WORK)

from optimization.genetic_algorithm import run_genetic_algorithm
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga

MV = np.load(os.path.join(HERE, "max_values.npy"))
BASE = dict(num_epochs=1000, mini_batch_size=64, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None, max_edits=50)
BASE['lambda'] = 0.9
TUNED = dict(keep_elitism=8, parent_selection_type='tournament', K_tournament=3)
OLD   = dict(keep_elitism=0, parent_selection_type='sss')

# (label, kind, actor-prefix or None, extra actor params, GA settings)
ARMS = [
    ("GA tuned, no operator",        'ga',   None,             {},                                     TUNED),
    ("IMGA untrained, OLD ga",       'imga', None,             dict(random_init_actor=True),           OLD),
    ("IMGA untrained, TUNED ga",     'imga', None,             dict(random_init_actor=True),           TUNED),
    ("IMGA A_baseline, TUNED ga",    'imga', "A_baseline",     {},                                     TUNED),
    ("IMGA E_normdesign, TUNED ga",  'imga', "E_normdesign",   dict(normalise_design=True),            TUNED),
    ("IMGA F_stress, OLD ga",        'imga', "F_stress",       dict(use_stress=True),                  OLD),
    ("IMGA F_stress, TUNED ga",      'imga', "F_stress",       dict(use_stress=True),                  TUNED),
]
SEEDS = [0, 1, 2, 3, 4, 5]


def run(arm, seed):
    label, kind, prefix, extra, ga = ARMS[arm]
    p = dict(BASE, **extra, **ga)
    torch.manual_seed(seed); np.random.seed(seed)
    if kind == 'ga':
        p['mutation_num_genes'] = 1
        r = run_genetic_algorithm(MV, p)
    else:
        p['date_str'] = f"{prefix}_t{seed % 3}" if prefix else "unused"
        if prefix is None:
            p['random_init_actor'] = True
        r = run_intelligent_mutation_ga(MV, p)
    return r[4][-1], r[5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r9a_shard_{a.shard}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip() and json.loads(l).get("hv") is not None:
                done.add(json.loads(l)["job_id"])
    jid = -1
    for ai in range(len(ARMS)):
        for s in SEEDS:
            jid += 1
            if jid % a.nshards != a.shard or jid in done:
                continue
            t0 = time.perf_counter()
            try:
                hv, nfe = run(ai, s)
                rec = dict(job_id=jid, name=ARMS[ai][0], seed=s, hv=hv, nfe=nfe,
                           secs=time.perf_counter() - t0)
            except Exception as e:
                rec = dict(job_id=jid, name=ARMS[ai][0], seed=s, hv=None, error=repr(e))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[9A-{a.shard}] {jid:3d} {ARMS[ai][0]:<30} s={s} "
                  f"hv={rec.get('hv') and round(rec['hv'],4)} nfe={rec.get('nfe')} "
                  f"{rec.get('secs',0):.0f}s", flush=True)
    print(f"[9A-{a.shard}] done")


if __name__ == "__main__":
    main()
