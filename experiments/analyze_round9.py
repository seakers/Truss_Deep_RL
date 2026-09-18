"""Regenerate every table in RESULTS_ROUND7-9.md from the raw shard files.

One command, no hand-copied numbers. Run it again after any experiment finishes and the
writeup can be re-checked against it. Arms with no data yet are reported as missing
rather than silently omitted, so a half-finished experiment cannot quietly become a
finished-looking table.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def load(patterns, key='name'):
    d = {}
    for pat in patterns:
        for f in glob.glob(os.path.join(HERE, "data", pat)):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                d.setdefault(r[key], {})[r['seed']] = r
    return d


def table(d, title, field='hv', order=None, extra=()):
    print(f"\n{title}")
    keys = order or sorted(d, key=lambda k: -np.mean([r[field] for r in d[k].values()]))
    head = f"{'arm':>30} {'n':>3} {'mean':>9} {'sd':>8}"
    for e in extra:
        head += f" {e:>13}"
    print(head)
    print('-' * len(head))
    for k in keys:
        if k not in d:
            print(f"{k:>30}  -- no data --")
            continue
        v = np.array([r[field] for r in d[k].values()])
        row = f"{k:>30} {len(v):>3} {v.mean():>9.4f} {v.std(ddof=1) if len(v) > 1 else 0:>8.4f}"
        for e in extra:
            vals = [r[e] for r in d[k].values() if e in r]
            row += f" {np.mean(vals):>13.4f}" if vals else f" {'--':>13}"
        print(row)


def paired(d, a, b, field='hv', label=''):
    if a not in d or b not in d:
        print(f"  {b} vs {a}: -- missing data --")
        return
    ks = sorted(set(d[a]) & set(d[b]))
    if len(ks) < 2:
        print(f"  {b} vs {a}: -- only {len(ks)} shared seeds --")
        return
    x = np.array([d[a][k][field] for k in ks])
    y = np.array([d[b][k][field] for k in ks])
    t, p = stats.ttest_rel(y, x)
    print(f"  {b:>26} - {a:<26} {y.mean()-x.mean():+.4f}  "
          f"({int((y > x).sum())}/{len(ks)})  p={p:.4g}   {label}")


print("=" * 96)
print("ROUND 7-8 -- training the mutation operator (32,000 NFE, 3 seeds)")
print("=" * 96)
r78 = load(['r7_shard_*.jsonl', 'r8_shard_*.jsonl'])
table(r78, "hypervolume by arm",
      order=['A_baseline', 'B_stress', 'C_shaping', 'D_stress_shaping',
             'E_normdesign', 'F_stress'])
print("\npaired contrasts:")
for a, b, why in [('A_baseline', 'B_stress', '(stress channel, raw scaling)'),
                  ('A_baseline', 'C_shaping', '(potential shaping)'),
                  ('A_baseline', 'D_stress_shaping', '(both)'),
                  ('E_normdesign', 'F_stress', '(stress channel, fixed scaling)')]:
    paired(r78, a, b, label=why)

print()
print("=" * 96)
print("ROUND 9 EXPERIMENT I -- oracle operators (64,000 NFE, 6 seeds)")
print("=" * 96)
orc = load(['r9o_shard_*.jsonl', 'r9o2_shard_*.jsonl'])
trackc = load(['r9c_shard_*.jsonl'])
if 'GA tuned, no operator' in trackc:
    orc['GA tuned (no util solve)'] = trackc['GA tuned, no operator']
table(orc, "hypervolume by operator",
      extra=('nfe',))
print("\nevery arm paired against the blind control that pays the same solve cost:")
for k in ['low_util', 'hi_soft', 'high_util', 'lo_fsd', 'fsd_one', 'hi_fsd', 'fsd_all']:
    paired(orc, 'random', k)
print("\nseparating greedy determinism from the rule itself:")
paired(orc, 'high_util', 'hi_soft', label='(same preference, sampled not argmax)')
paired(orc, 'random', 'hi_soft', label='(preference still costs, after determinism)')

print()
print("=" * 96)
print("ROUND 9 EXPERIMENT VII -- design-variable formulation (64,000 NFE, 6 seeds)")
print("=" * 96)
sym = load(['r9s_shard_*.jsonl'])
if 'GA tuned, no operator' in trackc:
    sym['GA 72 free bars (tuned)'] = trackc['GA tuned, no operator']
table(sym, "hypervolume by formulation", extra=('front_size', 'valid_frac'))
print()
for k in ['groups16', 'sym_mut']:
    paired(sym, 'GA 72 free bars (tuned)', k)

print()
print("=" * 96)
print("ROUND 9 EXPERIMENT V -- synthesis PPO weight conditioning (64,000 NFE, 6 seeds)")
print("=" * 96)
v = load(['r9v_shard_*.jsonl'])
table(v, "hypervolume and diagnostics", extra=('front_size', 'valid_frac', 'unique_frac'))
print()
# CONFOUNDED: ppo_rand_weights_uninformed.py has no hv reward branch, so the two shipped
# variants differ in reward mode as well as weight visibility. Experiment X's
# 'scalarized reward' arm is the missing control; see the writeup for the decomposition.
paired(v, 'PPO uninformed (blind)', 'PPO state (sees weights)',
       label='(CONFOUNDED with reward mode -- see experiment X)')
rs = v.get('random search (both metrics)')
if rs:
    own = np.array([r['hv_own_reference'] for r in rs.values()])
    shd = np.array([r['hv'] for r in rs.values()])
    t, p = stats.ttest_rel(own, shd)
    print(f"\n  random search scored on its OWN reference vs the SHARED one, same sample:")
    print(f"    own    = {own.mean():.4f}    (what random_search.py reports)")
    print(f"    shared = {shd.mean():.4f}    (what every other method is scored on)")
    print(f"    inflation = {own.mean()-shd.mean():+.4f}  ({int((own > shd).sum())}/{len(own)})  p={p:.4g}")

print()
print("=" * 96)
print("ROUND 9 TRACK C -- deployment of the trained operator (64,000 NFE, 6 seeds)")
print("=" * 96)
table(trackc, "hypervolume by arm", extra=('nfe',))
print()
for k in trackc:
    if k != 'GA tuned, no operator':
        paired(trackc, 'GA tuned, no operator', k)


print()
print("=" * 96)
print("ROUND 9 EXPERIMENT IX -- deploying the supervised pointer (64,000 NFE, 6 seeds)")
print("=" * 96)
dep = load(['r9d_shard_*.jsonl'])
table(dep, "hypervolume by operator", extra=('nfe', 'valid_frac'))
print()
for k in ['supervised', 'sup_greedy']:
    paired(dep, 'random', k)
paired(dep, 'supervised', 'sup_greedy', label='(cost of taking the argmax)')

print()
print("=" * 96)
print("ROUND 9 EXPERIMENT VIII -- grouped vs 72-bar, GA and PPO (64,000 NFE, 3 seeds)")
print("=" * 96)
g = load(['r9g_shard_*.jsonl'])
table(g, "hypervolume by method and formulation", extra=('front_size', 'secs'))
print()
paired(g, 'GA tuned, 72 bars', 'GA tuned, 16 groups', label='(grouping, on the GA)')
paired(g, 'PPO synthesis, 72 bars', 'PPO synthesis, 16 groups', label='(grouping, on the PPO)')

print()
print("=" * 96)
print("ROUND 9 EXPERIMENT VII(b) -- budget curves, 12 seeds")
print("=" * 96)
bud = load(['r9bud_shard_*.jsonl', 'r9bud2_shard_*.jsonl'])
table(bud, "final hypervolume", order=['groups16', 'bars72', 'sym_mut'])
print()
paired(bud, 'bars72', 'groups16', label='(16 groups vs 72 free bars)')
print("\nmean hypervolume vs evaluations:")
hdr = f"{'evals':>8}" + "".join(f"{k:>12}" for k in ['bars72', 'groups16'])
print(hdr); print('-' * len(hdr))
for n in (2000, 4000, 8000, 16000, 32000, 64000):
    row = f"{n:>8}"
    for k in ('bars72', 'groups16'):
        vals = [r['curve'][str(n)] for r in bud.get(k, {}).values() if str(n) in r.get('curve', {})]
        row += f"{np.mean(vals):>12.4f}" if vals else f"{'--':>12}"
    print(row)

print()
print("=" * 96)
print("ROUND 9 EXPERIMENTS X / XI -- synthesis PPO reward (64,000 NFE, 3 seeds)")
print("=" * 96)
rx = load(['r9x_shard_*.jsonl'])
table(rx, "X: reward mode", extra=('valid_frac', 'front_size'))
print()
paired(rx, 'hv reward (current default)', 'scalarized reward', label='(hypervolume)')
paired(rx, 'hv reward (current default)', 'scalarized reward', field='valid_frac',
       label='(feasible fraction)')
ry = load(['r9y_shard_*.jsonl'])
table(ry, "XI: graded infeasibility penalty (hv reward kept)", extra=('valid_frac',))
print()
for k in ['penalty 0.5', 'penalty 2.0']:
    paired(ry, 'penalty 0.0 (current default)', k, label='(hypervolume)')
    paired(ry, 'penalty 0.0 (current default)', k, field='valid_frac',
           label='(feasible fraction)')

print()
print("=" * 96)
print("ROUND 9 -- random search scored on both references (same sample, 6 seeds)")
print("=" * 96)
rsf = os.path.join(HERE, 'data', 'r9rs.jsonl')
if os.path.exists(rsf):
    rs = [json.loads(l) for l in open(rsf) if l.strip()]
    own = np.array([r['hv_own_reference'] for r in rs])
    shd = np.array([r['hv'] for r in rs])
    t, p = stats.ttest_rel(own, shd)
    print(f"  own reference (as shipped) : {own.mean():.4f} +- {own.std(ddof=1):.4f}")
    print(f"  shared reference           : {shd.mean():.4f} +- {shd.std(ddof=1):.4f}")
    print(f"  inflation                  : {(own-shd).mean():+.4f}  "
          f"({int((own > shd).sum())}/{len(rs)})  p={p:.4g}")
    print(f"  feasible fraction          : {np.mean([r['valid_frac'] for r in rs]):.3f}")
else:
    print("  -- no data --")
