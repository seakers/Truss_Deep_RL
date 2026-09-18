import os, sys, json, warnings, numpy as np
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import variant
from scipy import stats
mv=np.load(os.path.join(HERE, "max_values.npy")); EP,B,SEEDS=300,32,8

CONFIGS=[("baseline",{}),
         ("fp32",{'amp':False}),
         ("fp32 + critic128x2",{'amp':False,'d_model_critic':128,'nlayers_critic':2}),
         ("split encoders",{'token_encoding':'split'})]
out={}
for name,cfg in CONFIGS:
    hv=[]
    for s in range(SEEDS):
        hv.append(variant.run_ppo(cfg,mv,s,epochs=EP,batch=B)['hv'])
    out[name]=hv
    print(f"{name:<22} n={SEEDS} mean={np.mean(hv):.4f} sd={np.std(hv,ddof=1):.4f} "
          f"min={min(hv):.4f} max={max(hv):.4f}",flush=True)
    json.dump(out,open(os.path.join(HERE, "data", "confirm.json"), "w"),indent=1)

# random-search reference at the identical budget
from optimization.random_search import run_random_search
rs=[]
for s in range(5):
    np.random.seed(1000+s)
    *_ ,= [0]
    r=run_random_search({'num_epochs':EP,'mini_batch_size':B})
    rs.append(r[4][-1])
out['random search']=rs
print(f"{'random search':<22} n=5 mean={np.mean(rs):.4f} sd={np.std(rs,ddof=1):.4f}",flush=True)

print("\nWelch t-test vs baseline:")
b=out['baseline']
for name in ['fp32','fp32 + critic128x2','split encoders','random search']:
    t,p=stats.ttest_ind(out[name],b,equal_var=False)
    d=(np.mean(out[name])-np.mean(b))
    print(f"   {name:<22} delta={d:+.4f}  t={t:+.2f}  p={p:.4f}  {'SIGNIFICANT' if p<0.05 else 'not significant'}")
json.dump(out,open(os.path.join(HERE, "data", "confirm.json"), "w"),indent=1)
