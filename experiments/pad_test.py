import os, sys, warnings, torch, numpy as np
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from variant import Actor, DEFAULTS
dev=torch.device("cuda")
cfg={**DEFAULTS,'dropout':0.0,'amp':False,'fix_pe':True}
torch.manual_seed(0)
a=Actor(dev,cfg,50,5).to(dev); a.eval()

prefix=torch.rand(1,20,device=dev)                      # a length-20 prefix
padded=torch.cat([prefix, torch.zeros(1,57,device=dev)],1)   # padded to 77
with torch.no_grad():
    o_short=a.forward(prefix)[0,19,:]
    o_long =a.forward(padded)[0,19,:]
print(f"same prefix, unpadded vs padded, output at position 19:")
print(f"   max abs diff = {float((o_short-o_long).abs().max()):.3e}   -> causal mask {'OK' if float((o_short-o_long).abs().max())<1e-5 else 'NOT respected'}")

# batch-size dependence?
b1=torch.cat([prefix, torch.rand(7,20,device=dev)],0)
b2=torch.cat([torch.rand(31,20,device=dev), prefix],0)
with torch.no_grad():
    p1=a.forward(b1)[0,-1,:]; p2=a.forward(b2)[31,-1,:]
print(f"same prefix at batch row 0 (of 8) vs row 31 (of 32): max diff = {float((p1-p2).abs().max()):.3e}")

# LayerNorm / attention over padding: check a masked position's influence
with torch.no_grad():
    padA=torch.cat([prefix, torch.zeros(1,57,device=dev)],1)
    padB=torch.cat([prefix, torch.ones(1,57,device=dev)*9.0],1)
    oa=a.forward(padA)[0,19,:]; ob=a.forward(padB)[0,19,:]
print(f"changing the PADDING values only: max diff at position 19 = {float((oa-ob).abs().max()):.3e}")
