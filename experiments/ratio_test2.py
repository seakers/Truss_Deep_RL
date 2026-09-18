import os, sys, warnings, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimization.transformer_rand_weights_state import Actor
from evaluation.truss_problem import TrussProblem
dev=torch.device("cuda"); torch.manual_seed(0); np.random.seed(0)
prob=TrussProblem(); nact=prob.n_elements; arange=len(prob.area_map); ntot=5; B=16
actor=Actor(device=dev,params={'learning_rate':3e-4,'clip_ratio':0.05},action_length=arange).to(dev)

def rollout(train_mode):
    actor.train(train_mode)
    w=np.random.RandomState(0).rand(B,ntot); w[:,-2:]=1.0; w=w/w.sum(axis=1,keepdims=True)
    obs=[w[i].tolist() for i in range(B)]; acts=[[] for _ in range(B)]; lps=[[] for _ in range(B)]
    torch.manual_seed(1)
    with torch.no_grad():
        for _ in range(nact):
            lp,a=actor.sample_action(obs)
            for i,x in enumerate(a.tolist()):
                acts[i].append(x); lps[i].append(lp.tolist()[i]); obs[i].append(x)
    O=[];A=[];L=[];P=[]
    for i in range(B):
        for s in range(nact):
            frag=obs[i][:s+1+ntot]
            O.append(frag+[0]*(nact+ntot-len(frag))); A.append(acts[i][s])
            L.append(lps[i][s]); P.append(len(frag)-1)
    return (torch.tensor(O,dtype=torch.float32,device=dev), torch.tensor(A,device=dev),
            torch.tensor(L,dtype=torch.float32,device=dev), torch.tensor(P,device=dev))

def recompute(O,A,P,pos_correct):
    actor.eval()
    with torch.no_grad():
        out=actor.forward(O)
        p = out[torch.arange(out.size(0),device=dev),P,:] if pos_correct else out[:,-1,:]
        return torch.sum(F.one_hot(A.long(),arange)*torch.log(p+1e-10),dim=-1)

def report(tag,r):
    print(f"   {tag:<46} mean={r.mean():.4f} sd={r.std():.4f} "
          f"median={r.median():.4f} within1%={(abs(r-1)<0.01).float().mean()*100:5.1f}%")

print("A) Rollout with dropout OFF (isolates the indexing question alone)")
O,A,L,P = rollout(train_mode=False)
report("gather at true action position", torch.exp(recompute(O,A,P,True)-L))
report("repo: always take position -1 of padded seq", torch.exp(recompute(O,A,P,False)-L))

print("\nB) Rollout with dropout ON (what the repo actually does)")
O,A,L,P = rollout(train_mode=True)
report("gather at true action position", torch.exp(recompute(O,A,P,True)-L))
report("repo: always take position -1 of padded seq", torch.exp(recompute(O,A,P,False)-L))

print("\nC) How far is position -1 from the action position, on average?")
print(f"   mean padding length appended: {(nact+ntot-1-P.float()).mean():.1f} tokens "
      f"(range {int((nact+ntot-1-P.float()).min())}-{int((nact+ntot-1-P.float()).max())})")
