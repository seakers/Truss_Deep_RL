import os, json, glob, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from scipy import stats
rows=[json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, 'data', 'ctl_shard_*.jsonl'))) for l in open(f)]
abl=[json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, 'data', 'abl_shard_*.jsonl'))) for l in open(f)]
errs=[r for r in rows if 'error' in r]
if errs: print(f"!! {len(errs)} failed"); [print("  ",e.get('tag'),e['error'][:110]) for e in errs[:3]]
d=collections.defaultdict(list)
for r in rows:
    if r.get('kind')=='imga' and 'hv' in r: d[r['tag']].append(r['hv'])
for r in abl:
    if 'hv' in r: d[r['name']].append(r['hv'])
print("IMGA MUTATION OPERATOR: does better training help?  (n=8, 16,000 NFE)")
print(f"  {'repair actor used':<34}{'n':>3}{'mean HV':>10}{'sd':>9}")
ORDER=[("random init (untrained)","IMGA untrained"),
       ("trained 9,600 NFE, domination","IMGA trained"),
       ("trained 32,000 NFE, domination","long-domination"),
       ("trained 32,000 NFE, hv reward","long-hv"),
       ("plain GA (no learned operator)","plain GA")]
for label,key in ORDER:
    if key not in d: continue
    v=d[key]
    print(f"  {label:<34}{len(v):>3}{np.mean(v):>10.4f}{(np.std(v,ddof=1) if len(v)>1 else 0):>9.4f}")
print("\n  vs random-init operator:")
if "IMGA untrained" in d:
    b=d["IMGA untrained"]
    for label,key in ORDER[1:4]:
        if key not in d or len(d[key])<2: continue
        t,p=stats.ttest_ind(d[key],b,equal_var=False)
        print(f"    {label:<34} delta={np.mean(d[key])-np.mean(b):+.4f} "
              f"({(np.mean(d[key])/np.mean(b)-1)*100:+.1f}%) p={p:.4f} {'SIG' if p<0.05 else 'ns'}")
