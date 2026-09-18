import os, sys, json, warnings, numpy as np
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import variant
from scipy import stats
mv=np.load(os.path.join(HERE, "max_values.npy")); EP,B,SEEDS=1000,64,3     # the real 64,000-eval budget
FIX={'fix_pe':True,'fix_position':True,'dropout':0.0,'amp':False}
out={}
for name,cfg in [("baseline (repo as-is)",{}),
                 ("all fixes + 5 update iters",{**FIX,'target_kl':1e9})]:
    hv=[];ent=[]
    for s in range(SEEDS):
        r=variant.run_ppo(cfg,mv,s,epochs=EP,batch=B)
        hv.append(r['hv']); ent.append(np.mean(r['entropy'][-10:]))
        print(f"   {name} seed {s}: hv={r['hv']:.4f} ent={ent[-1]:.3f} {r['wall']:.0f}s",flush=True)
    out[name]=dict(hv=hv,mean=float(np.mean(hv)),sd=float(np.std(hv,ddof=1)),ent=float(np.mean(ent)))
    print(f"{name:<30} hv={np.mean(hv):.4f}+-{np.std(hv,ddof=1):.4f} ent={np.mean(ent):.3f}",flush=True)
    json.dump(out,open(os.path.join(HERE, "data", "fullbudget.json"), "w"),indent=1)

from optimization.random_search import run_random_search
rs=[]
for s in range(3):
    np.random.seed(2000+s)
    rs.append(run_random_search({'num_epochs':EP,'mini_batch_size':B})[4][-1])
out['random search']=dict(hv=rs,mean=float(np.mean(rs)),sd=float(np.std(rs,ddof=1)))
print(f"{'random search':<30} hv={np.mean(rs):.4f}+-{np.std(rs,ddof=1):.4f}")
json.dump(out,open(os.path.join(HERE, "data", "fullbudget.json"), "w"),indent=1)
for k in out:
    if k=='baseline (repo as-is)': continue
    t,p=stats.ttest_ind(out[k]['hv'],out['baseline (repo as-is)']['hv'],equal_var=False)
    print(f"   {k:<30} vs baseline: delta={out[k]['mean']-out['baseline (repo as-is)']['mean']:+.4f} p={p:.4f}")
