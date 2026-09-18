"""Round-3 analysis. Seeds are shared across arms, so the paired test is the correct one;
the unpaired Welch column is shown only because rounds 1-2 used it."""
import os, json, glob, numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
recs = []
for f in sorted(glob.glob(os.path.join(HERE, "data", "r3_shard_*.jsonl"))):
    for line in open(f):
        if line.strip():
            recs.append(json.loads(line))

err = [r for r in recs if 'error' in r]
recs = [r for r in recs if 'error' not in r]
if err:
    print(f"!! {len(err)} failed jobs, e.g. {err[0]['name']}: {err[0]['error'][:120]}\n")

ORDER = ["baseline (repo as-was)", "bugs fixed", "fixed + unpinned weights",
         "fixed + hv reward", "fixed + hv + unpinned", "hv reward, bugs left in",
         "random search"]
by = {}
for r in recs:
    by.setdefault(r['name'], {})[r['seed']] = r


def col(name, key):
    d = by.get(name, {})
    return np.array([d[s][key] for s in sorted(d) if key in d[s]])


print("=" * 104)
print("ROUND 3 - synthesis PPO @ 64,000 evaluations (1000 epochs x batch 64)")
print("=" * 104)
print(f"{'arm':28} {'n':>3} {'mean HV':>9} {'sd':>8} {'min':>8} {'max':>8} "
      f"{'valid%':>8} {'entropy':>8} {'critic EV':>10}")
for name in ORDER:
    hv = col(name, 'hv')
    if not len(hv):
        continue
    vf, en, ev = col(name, 'valid_frac'), col(name, 'entropy_final'), col(name, 'ev')
    f = lambda a: f"{a.mean():>8.3f}" if len(a) else f"{'-':>8}"
    print(f"{name:28} {len(hv):>3} {hv.mean():>9.4f} {hv.std(ddof=1):>8.4f} "
          f"{hv.min():>8.4f} {hv.max():>8.4f} "
          f"{(vf.mean() * 100 if len(vf) else float('nan')):>7.1f}% {f(en)} "
          f"{(f'{ev.mean():.2f}' if len(ev) else '-'):>10}")

base = col("baseline (repo as-was)", 'hv')
print()
print("=" * 104)
print("vs baseline (repo as-was).  Seeds are shared, so PAIRED is the right test.")
print("=" * 104)
print(f"{'comparison':28} {'delta':>9} {'%':>8} {'r(seed)':>9} {'p paired':>10} {'p Welch':>9}")
for name in ORDER[1:]:
    x = col(name, 'hv')
    if len(x) != len(base) or not len(x):
        continue
    d = x - base
    tp = stats.ttest_rel(x, base).pvalue
    tw = stats.ttest_ind(x, base, equal_var=False).pvalue
    r = np.corrcoef(x, base)[0, 1]
    print(f"{name:28} {d.mean():>+9.4f} {100 * d.mean() / base.mean():>+7.1f}% "
          f"{r:>9.3f} {tp:>10.4f} {tw:>9.4f}")

rs = col("random search", 'hv')
if len(rs):
    print()
    print("vs random search (the bar any of this has to clear):")
    for name in ORDER[:-1]:
        x = col(name, 'hv')
        if not len(x):
            continue
        p = stats.ttest_ind(x, rs, equal_var=False).pvalue
        print(f"  {name:28} delta={x.mean() - rs.mean():>+8.4f}  p={p:.4f}"
              f"{'  SIGNIFICANT' if p < 0.05 else ''}")

print()
print("=" * 104)
print("Is hypervolume still just tracking the feasible fraction?")
print("=" * 104)
for name in ORDER[:-1]:
    hv, vf = col(name, 'hv'), col(name, 'valid_frac')
    if len(hv) < 3 or not len(vf):
        continue
    print(f"  {name:28} r(HV, valid_frac) = {np.corrcoef(hv, vf)[0, 1]:>+6.3f}"
          f"   valid% range {vf.min() * 100:.0f}-{vf.max() * 100:.0f}")

print()
print("per-seed hypervolume (seeds 0-7 in order):")
for name in ORDER:
    hv = col(name, 'hv')
    if len(hv):
        print(f"  {name:28} " + " ".join(f"{v:.4f}" for v in hv))

tot = len(recs)
print(f"\n{tot} runs recorded"
      + (f", {sum(r.get('wall', 0) for r in recs) / 3600:.1f} core-hours" if tot else ""))
