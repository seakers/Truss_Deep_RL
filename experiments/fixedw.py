"""Is the random-weight scalarisation itself destroying the learning signal?
Train with a FIXED preference vector and see whether the policy improves that
single scalar objective. If it learns here but not with random weights, the
scalarisation (and normalising advantages across mixed weight vectors) is the cause."""
import os, sys, json, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import variant
from variant import Actor, Critic, discounted_cumulative_sums, DEFAULTS
from evaluation.truss_problem import TrussProblem

def run_fixed(cfg, max_values, seed, epochs, batch, fixed_w):
    cfg={**DEFAULTS, **cfg}
    torch.manual_seed(seed); np.random.seed(seed)
    dev=torch.device("cuda"); prob=TrussProblem()
    nobj=prob.num_objectives; omm=prob.objective_min_max
    nact=prob.n_elements; arange=len(prob.area_map); ntot=len(max_values)
    min_mask=np.array([o.lower()=='min' for o in omm])
    actor=Actor(dev,cfg,arange,ntot).to(dev); critic=Critic(dev,cfg,ntot,nact+ntot,arange,ntot).to(dev)
    fw=np.array(fixed_w,float); fw=fw/fw.sum()
    scal=[]; ent=[]
    for ep in range(epochs):
        rewards=[[] for _ in range(batch)]; actions=[[] for _ in range(batch)]
        logps=[[] for _ in range(batch)]; designs=[[] for _ in range(batch)]
        weights=np.tile(fw,(batch,1)); obs=[weights[i].tolist() for i in range(batch)]
        es=[]
        with torch.no_grad():
            for _ in range(nact):
                lp,a,e=actor.sample_action(obs); es.append(e)
                for i,act in enumerate(a.tolist()):
                    actions[i].append(act); logps[i].append(lp[i].item())
                    obs[i].append(act); rewards[i].append(0.); designs[i].append(int(act))
        ent.append(float(np.mean(es)))
        eps=[]
        for i,d in enumerate(designs):
            ov,con,cv=prob.evaluate(d); arr=np.array(ov+cv,float)
            if con:
                rv=np.where(min_mask,-np.ones(ntot),0.0); rv[-2:]=np.array(cv)
                rv[-2:][min_mask[-2:]]=-rv[-2:][min_mask[-2:]]
            else:
                n=arr/np.array(max_values,float); rv=np.where(min_mask,-n,n)
            r=float(np.dot(fw,rv)); rewards[i][-1]+=r; eps.append(r)
        scal.append(float(np.mean(eps)))
        cvals=[]
        with torch.no_grad():
            for k in range(nact):
                co=[obs[i][:k+1+ntot] for i in range(batch)]
                cv_=np.array(critic.sample_critic(co,ntot).float().tolist())
                cvals.append(np.sum(np.multiply(weights,cv_),axis=1))
        values=[[cvals[k][i] for k in range(nact)] for i in range(batch)]
        for i in range(batch): values[i].append(values[i][-1])
        advs=[];rets=[]
        for i in range(batch):
            r=np.array(rewards[i]); v=np.array(values[i])
            deltas=r+cfg['gamma']*v[1:]-v[:-1]
            advs.append(discounted_cumulative_sums(deltas,cfg['gamma']*cfg['lam']))
            rets.append(np.array(discounted_cumulative_sums(r,cfg['gamma']*cfg['lam']),np.float32))
        af=np.concatenate(advs); advs=[(a-af.mean())/(af.std()+1e-8) for a in advs]
        O=[];A=[];L=[];AD=[];R=[];W=[]
        for i in range(batch):
            for s in range(nact):
                frag=obs[i][:s+1+ntot]
                O.append(frag+[0]*(nact+ntot-len(frag))); A.append(actions[i][s])
                L.append(logps[i][s]); AD.append(advs[i][s]); R.append(rets[i][s]); W.append(weights[i])
        t=lambda x,d=torch.float32: torch.tensor(x,dtype=d,device=dev)
        O=t(O);A=t(A);L=t(L);AD=t(AD);R=t(R);W=t(np.array(W))
        for _ in range(cfg['update_iterations']):
            _,kl=actor.ppo_update(O,A,L,AD)
            if kl>cfg['target_kl']: break
        for _ in range(cfg['update_iterations']): critic.ppo_update(O,R,W)
    return scal, ent

mv=np.load(os.path.join(HERE, "max_values.npy"))
out={}
for label,w in [("mass-only [1,0,0,1,1]",[1,0,0,1,1]), ("freq-only [0,0,1,1,1]",[0,0,1,1,1])]:
    first=[];last=[];ents=[]
    for s in range(3):
        sc,en=run_fixed({}, mv, s, 300, 32, w)
        sc=np.array(sc); first.append(sc[:20].mean()); last.append(sc[-20:].mean()); ents.append(en[-1])
    out[label]=dict(first=float(np.mean(first)),last=float(np.mean(last)),ent=float(np.mean(ents)))
    print(f"{label:<26} scalarised reward: first20={np.mean(first):+.4f} -> last20={np.mean(last):+.4f} "
          f"(improvement {np.mean(last)-np.mean(first):+.4f})  final entropy={np.mean(ents):.3f}",flush=True)
json.dump(out,open(os.path.join(HERE, "data", "fixedw.json"), "w"),indent=1)
