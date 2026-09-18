import os, sys, warnings, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from variant import Actor, DEFAULTS
from evaluation.truss_problem import TrussProblem
dev=torch.device("cuda"); prob=TrussProblem(); nact=prob.n_elements; arange=len(prob.area_map); ntot=5; B=16

def measure(cfg, label):
    cfg={**DEFAULTS, **cfg}
    torch.manual_seed(0); np.random.seed(0)
    a=Actor(dev,cfg,arange,ntot).to(dev)
    if cfg['eval_rollout']: a.eval()
    w=np.random.RandomState(0).rand(B,ntot); w[:,-2:]=1.0; w=w/w.sum(axis=1,keepdims=True)
    obs=[w[i].tolist() for i in range(B)]; acts=[[] for _ in range(B)]; lps=[[] for _ in range(B)]
    with torch.no_grad():
        for _ in range(nact):
            lp,act,_=a.sample_action(obs)
            for i,x in enumerate(act.tolist()):
                acts[i].append(x); lps[i].append(lp[i].item()); obs[i].append(x)
    a.train()
    O=[];A=[];L=[];P=[]
    for i in range(B):
        for s in range(nact):
            frag=obs[i][:s+1+ntot]
            O.append(frag+[0]*(nact+ntot-len(frag))); A.append(acts[i][s]); L.append(lps[i][s]); P.append(len(frag)-1)
    O=torch.tensor(O,dtype=torch.float32,device=dev); A=torch.tensor(A,device=dev)
    L=torch.tensor(L,dtype=torch.float32,device=dev); P=torch.tensor(P,device=dev)
    if cfg['eval_rollout']: a.eval()
    with torch.no_grad():
        out=a.forward(O)
        p = out[torch.arange(out.size(0),device=dev),P,:] if cfg['fix_position'] else out[:,-1,:]
        new=torch.sum(F.one_hot(A.long(),arange)*torch.log(p+1e-10),dim=-1)
    r=torch.exp(new-L); c=cfg['clip_ratio']
    print(f"{label:<42} ratio mean={r.mean():.4f} sd={r.std():.5f} within1%={(abs(r-1)<0.01).float().mean()*100:5.1f}%  "
          f"outside clip band={(((r<1-c)|(r>1+c)).float().mean()*100):5.1f}%")

measure({}, "repo as-is")
measure({'fix_pe':True}, "+ positional encoding fixed")
measure({'fix_position':True}, "+ gather at true position")
measure({'fix_position':True,'dropout':0.0}, "+ no dropout")
measure({'fix_pe':True,'fix_position':True}, "+ PE + true position")
measure({'fix_pe':True,'fix_position':True,'dropout':0.0}, "+ PE + position + no dropout")
measure({'fix_pe':True,'fix_position':True,'dropout':0.0,'amp':False}, "+ PE + position + no dropout + fp32")
