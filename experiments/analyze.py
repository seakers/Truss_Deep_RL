import os, json, glob, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from scipy import stats

rows=[json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, 'data', 'shard_*.jsonl'))) for l in open(f)]
errs=[r for r in rows if 'error' in r]
if errs:
    print(f"!! {len(errs)} failed jobs"); [print("   ",e['name'],e['seed'],e['error'][:120]) for e in errs[:5]]
d=collections.defaultdict(list)
for r in rows:
    if 'hv' in r: d[r['name']].append(r)

def blk(title, names, extra=None):
    print(f"\n{title}")
    print(f"  {'config':<30}{'n':>3}{'mean HV':>10}{'sd':>8}{'min':>8}{'max':>8}" + ("".join(f"{e[0]:>14}" for e in (extra or ""))))
    base=None
    for nm in names:
        if nm not in d: continue
        v=[r['hv'] for r in d[nm]]
        line=f"  {nm:<30}{len(v):>3}{np.mean(v):>10.4f}{np.std(v,ddof=1) if len(v)>1 else 0:>8.4f}{min(v):>8.4f}{max(v):>8.4f}"
        for _,key,fmt in (extra or []):
            vals=[r[key] for r in d[nm] if key in r]
            line+=format(np.mean(vals), fmt).rjust(14) if vals else "-".rjust(14)
        print(line)
    return

PPO=["baseline","entropy 0.01","entropy 0.01 decay","all fixes","all fixes + entropy decay","random search"]
blk("SYNTHESIS PPO @ 64,000 evaluations", PPO,
    [("entropy",'entropy_final','.3f'),("collapse ep",'collapse_epoch','.0f'),("critic EV",'ev','+.2f')])
print("\n  Welch t-test vs baseline:")
if 'baseline' in d:
    b=[r['hv'] for r in d['baseline']]
    for nm in PPO[1:]:
        if nm not in d or len(d[nm])<2: continue
        v=[r['hv'] for r in d[nm]]
        t,p=stats.ttest_ind(v,b,equal_var=False)
        print(f"    {nm:<30} delta={np.mean(v)-np.mean(b):+.4f}  p={p:.4f}  "
              f"{'SIGNIFICANT' if p<0.05 else 'ns'}")

for tag,pref in [("REPAIR PPO @ 4,800 NFE","repair "),("REPAIR PPO @ 12,800 NFE","repair-big ")]:
    names=[f"{pref}{m}" for m in ['domination','front','hv']]
    if not any(n in d for n in names): continue
    blk(tag, names)
    if names[0] in d and len(d[names[0]])>1:
        b=[r['hv'] for r in d[names[0]]]
        print("\n  Welch t-test vs domination (current):")
        for nm in names[1:]:
            if nm not in d or len(d[nm])<2: continue
            v=[r['hv'] for r in d[nm]]
            t,p=stats.ttest_ind(v,b,equal_var=False)
            print(f"    {nm:<30} delta={np.mean(v)-np.mean(b):+.4f} ({(np.mean(v)/np.mean(b)-1)*100:+.1f}%)  p={p:.4f}  "
                  f"{'SIGNIFICANT' if p<0.05 else 'ns'}")

# entropy collapse vs outcome
print("\nENTROPY COLLAPSE vs FINAL HV (synthesis, all configs pooled)")
pts=[(r['collapse_epoch'],r['hv'],r['name']) for n in PPO for r in d.get(n,[]) if 'collapse_epoch' in r]
coll=[h for c,h,_ in pts if c>=0]; nocoll=[h for c,h,_ in pts if c<0]
if coll and nocoll:
    t,p=stats.ttest_ind(coll,nocoll,equal_var=False)
    print(f"  collapsed    n={len(coll):>3} mean HV={np.mean(coll):.4f} sd={np.std(coll,ddof=1):.4f}")
    print(f"  no collapse  n={len(nocoll):>3} mean HV={np.mean(nocoll):.4f} sd={np.std(nocoll,ddof=1):.4f}   p={p:.4f}")
print(f"\ntotal jobs recorded: {len(rows)}")
