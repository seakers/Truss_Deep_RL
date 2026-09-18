import os, sys, json, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import variant
from scipy import stats
mv=np.load(os.path.join(HERE, "max_values.npy")); EP,B,SEEDS=300,32,8
FIX={'fix_pe':True,'fix_position':True,'dropout':0.0,'amp':False}
def wresp(r):
    prob,ntot,nact=r['problem'],r['weights_dim'],r['problem'].n_elements
    def gen(w,n=32):
        w=np.tile(np.array(w,float)/np.sum(w),(n,1)); obs=[w[i].tolist() for i in range(n)]
        with torch.no_grad():
            for _ in range(nact):
                _,a,_=r['actor'].sample_action(obs)
                for i,x in enumerate(a.tolist()): obs[i].append(x)
        return np.array([prob.evaluate([int(v) for v in o[ntot:]])[0] for o in obs])
    l,s=gen([1,0,0,1,1]),gen([0,0,1,1,1])
    return abs(s[:,0].mean()-l[:,0].mean())/l[:,0].mean(), abs(s[:,2].mean()-l[:,2].mean())/l[:,2].mean()
CONFIGS=[("baseline (repo as-is)",{}),
         ("PE fix only",{'fix_pe':True}),
         ("all ratio fixes",FIX),
         ("all fixes + 5 update iters",{**FIX,'target_kl':1e9})]
out={}
for name,cfg in CONFIGS:
    hv=[];ent=[];ev=[];wr=[]
    for s in range(SEEDS):
        r=variant.run_ppo(cfg,mv,s,epochs=EP,batch=B)
        hv.append(r['hv']);ent.append(np.mean(r['entropy'][-10:]))
        ev.append(np.mean(r['ev'][-50:]));wr.append(wresp(r))
    wr=np.array(wr)
    out[name]=dict(hv=hv,hv_mean=float(np.mean(hv)),hv_sd=float(np.std(hv,ddof=1)),
                   entropy=float(np.mean(ent)),ev=float(np.mean(ev)),
                   mass_resp=float(wr[:,0].mean()),freq_resp=float(wr[:,1].mean()))
    print(f"{name:<30} hv={np.mean(hv):.4f}+-{np.std(hv,ddof=1):.4f} ent={np.mean(ent):.3f} "
          f"ev={np.mean(ev):+.2f} massResp={wr[:,0].mean()*100:5.1f}% freqResp={wr[:,1].mean()*100:5.1f}%",flush=True)
    json.dump(out,open(os.path.join(HERE, "data", "final.json"), "w"),indent=1)
b=out["baseline (repo as-is)"]['hv']
print(f"\nRandom search reference (n=5): 0.3795 +- 0.0110")
print("\nWelch t-test vs baseline:")
for k in out:
    if k=="baseline (repo as-is)": continue
    t,p=stats.ttest_ind(out[k]['hv'],b,equal_var=False)
    print(f"   {k:<30} delta={np.mean(out[k]['hv'])-np.mean(b):+.4f} p={p:.4f} "
          f"{'SIGNIFICANT' if p<0.05 else 'ns'}")
