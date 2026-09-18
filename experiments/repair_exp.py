import os, sys, json, warnings, numpy as np
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import repair_variant as R
mv=np.load(os.path.join(HERE, "max_values.npy")); EP,B,SEEDS=150,32,3
out={}
for mode in ['domination','front','hv']:
    hv=[];curves=[];nfe=[]
    for s in range(SEEDS):
        r=R.run_repair({'reward':mode},mv,s,epochs=EP,batch=B)
        hv.append(r['hv']);curves.append(r['hv_curve']);nfe.append(r['NFE'])
    n=min(len(c) for c in curves)
    at=[c[n-1] for c in curves]     # compare at a common NFE
    out[mode]=dict(hv_mean=float(np.mean(hv)),hv_std=float(np.std(hv)),hv=hv,
                   common_nfe=n,hv_at_common=[float(x) for x in at],
                   hv_at_common_mean=float(np.mean(at)),nfe=nfe)
    print(f"{mode:<12} hv={np.mean(hv):.4f}+-{np.std(hv):.4f}  "
          f"@commonNFE={n}: {np.mean(at):.4f}  NFE={nfe}",flush=True)
    json.dump(out,open(os.path.join(HERE, "data", "repair.json"), "w"),indent=1)
