import os, sys, json, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import variant
mv=np.load(os.path.join(HERE, "max_values.npy")); EP,B,SEEDS=300,32,3

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

CONFIGS=[
  ("baseline",                       {}),
  ("fp32",                           {'amp':False}),
  ("fp32 + critic64",                {'amp':False,'d_model_critic':64}),
  ("fp32 + critic128x2",             {'amp':False,'d_model_critic':128,'nlayers_critic':2}),
  ("fp32 + critic64 + g1l1",         {'amp':False,'d_model_critic':64,'gamma':1.0,'lam':1.0}),
  ("target_kl inf (5 actor iters)",  {'target_kl':1e9}),
  ("fp32 + critic64 + kl inf",       {'amp':False,'d_model_critic':64,'target_kl':1e9}),
  ("split encoders (recheck)",       {'token_encoding':'split'}),
  ("fp32 + critic64 + split",        {'amp':False,'d_model_critic':64,'token_encoding':'split'}),
  ("actor64 + entropy 0.01",         {'d_model_actor':64,'entropy_coef':0.01}),
  ("actor64 + entropy 0.05",         {'d_model_actor':64,'entropy_coef':0.05}),
  ("fp32+critic64+actor64+ent0.05",  {'amp':False,'d_model_critic':64,'d_model_actor':64,'entropy_coef':0.05}),
]
out={}
for name,cfg in CONFIGS:
    hv=[];ent=[];ev=[];wr=[]
    for s in range(SEEDS):
        r=variant.run_ppo(cfg,mv,s,epochs=EP,batch=B)
        hv.append(r['hv']);ent.append(np.mean(r['entropy'][-10:]))
        ev.append(np.mean(r['ev'][-50:]));wr.append(wresp(r))
    wr=np.array(wr)
    out[name]=dict(hv_mean=float(np.mean(hv)),hv_std=float(np.std(hv)),hv=hv,
                   entropy=float(np.mean(ent)),ev=float(np.mean(ev)),
                   mass_resp=float(wr[:,0].mean()),freq_resp=float(wr[:,1].mean()))
    print(f"{name:<40} hv={np.mean(hv):.4f}+-{np.std(hv):.4f} ent={np.mean(ent):.3f} "
          f"ev={np.mean(ev):+.2f} massResp={wr[:,0].mean()*100:5.1f}% freqResp={wr[:,1].mean()*100:5.1f}%",flush=True)
    json.dump(out,open(os.path.join(HERE, "data", "followup.json"), "w"),indent=1)
