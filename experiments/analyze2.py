import os, json, glob, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from scipy import stats
rows=[json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, 'data', 'abl_shard_*.jsonl'))) for l in open(f)]
errs=[r for r in rows if 'error' in r]
if errs: print(f"!! {len(errs)} failed:"); [print("  ",e['name'],e['seed'],e['error'][:100]) for e in errs[:4]]
d=collections.defaultdict(list)
for r in rows:
    if 'hv' in r: d[r['name']].append(r)
ORDER=["plain GA","WarmStart untrained","WarmStart trained","IMGA untrained","IMGA trained"]
print(f"HYBRID ABLATION @ 16,000 NFE   (identical NFE across arms)")
print(f"  {'arm':<24}{'n':>3}{'mean HV':>10}{'sd':>9}{'min':>9}{'max':>9}{'NFE':>8}")
for k in ORDER:
    if k not in d: continue
    v=[r['hv'] for r in d[k]]; nfe=np.mean([r['NFE'] for r in d[k]])
    print(f"  {k:<24}{len(v):>3}{np.mean(v):>10.4f}{(np.std(v,ddof=1) if len(v)>1 else 0):>9.4f}{min(v):>9.4f}{max(v):>9.4f}{nfe:>8.0f}")
def cmp(a,b,label):
    if a not in d or b not in d or len(d[a])<2 or len(d[b])<2: return
    va=[r['hv'] for r in d[a]]; vb=[r['hv'] for r in d[b]]
    t,p=stats.ttest_ind(va,vb,equal_var=False)
    print(f"  {label:<46} delta={np.mean(va)-np.mean(vb):+.4f} ({(np.mean(va)/np.mean(vb)-1)*100:+.1f}%) "
          f"p={p:.4f} {'SIGNIFICANT' if p<0.05 else 'ns'}")
print("\nDecomposition:")
cmp("IMGA untrained","plain GA",          "operator structure alone (IMGA untrained vs plain GA)")
cmp("IMGA trained","IMGA untrained",      "what the LEARNING adds (IMGA trained vs untrained)")
cmp("WarmStart untrained","plain GA",     "warm-start structure alone (untrained vs plain GA)")
cmp("WarmStart trained","WarmStart untrained","what the LEARNING adds (WarmStart trained vs untrained)")
cmp("IMGA trained","plain GA",            "total hybrid effect (IMGA trained vs plain GA)")
print(f"\ntotal jobs: {len(rows)}")
