import os, sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimization.transformer_rand_weights_state import PositionalEncoding, Actor
dev=torch.device("cuda")

pe=PositionalEncoding(16).to(dev); pe.eval()
print("pe buffer shape:", tuple(pe.pe.shape), " (max_len, 1, d_model)")
x=torch.zeros(8, 40, 16, device=dev)          # (batch=8, seq=40, d_model=16)
out=pe(x)
print("input (batch=8, seq=40, d=16) ->", tuple(out.shape))
print("slice taken: self.pe[:x.size(0)] = self.pe[:8] -> shape", tuple(pe.pe[:8].shape),
      "  i.e. indexed by BATCH, broadcast across SEQ")
print(f"\nare all 40 sequence positions given the SAME encoding? "
      f"{torch.allclose(out[0,0], out[0,39])}   <- positional information is absent")
print(f"do two different batch rows get DIFFERENT encodings for identical input? "
      f"{not torch.allclose(out[0,0], out[3,0])}   <- the 'position' encoded is the batch index")

# consequence: identical sequence, different batch slot -> different policy output
torch.manual_seed(0)
a=Actor(device=dev,params={'learning_rate':3e-4,'clip_ratio':0.05},action_length=50).to(dev); a.eval()
seq=torch.rand(1,30,device=dev)
b1=torch.cat([seq, torch.rand(7,30,device=dev)],0)      # our sequence at row 0
b2=torch.cat([torch.rand(4,30,device=dev), seq, torch.rand(3,30,device=dev)],0)  # same seq at row 4
with torch.no_grad():
    p1=a.forward(b1)[0,-1,:]
    p2=a.forward(b2)[4,-1,:]
print(f"\nSAME sequence at batch row 0 vs row 4 -> max prob difference {float((p1-p2).abs().max()):.6f}"
      f"  (should be 0.0)")
print(f"   total-variation distance between the two action distributions: {float((p1-p2).abs().sum()/2):.4f}")

# the repair model transposes first -- check it is unaffected
import inspect
from optimization.transformer_design_repair import Actor as RA
src=inspect.getsource(RA.forward)
print("\nrepair model applies PE as:",
      [l.strip() for l in src.split("\n") if "positional_encoding" in l])
