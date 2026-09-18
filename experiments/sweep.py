import os, sys, json, time, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import variant

mv = np.load(os.path.join(HERE, "max_values.npy")); EP, B, SEEDS = 300, 32, 3

CONFIGS = [
    ("baseline",                     {}),
    # --- change #1: what the network reads ---
    ("scaled indices",               {'token_encoding':'scaled'}),
    ("split encoders",               {'token_encoding':'split'}),
    ("embedding",                    {'token_encoding':'embedding'}),
    # --- change #2: entropy bonus ---
    ("entropy bonus 0.01",           {'entropy_coef':0.01}),
    # --- change #3: credit assignment / critic ---
    ("gamma=1 lambda=1",             {'gamma':1.0,'lam':1.0}),
    ("critic d=64",                  {'d_model_critic':64}),
    ("critic d=128 x2",              {'d_model_critic':128,'nlayers_critic':2}),
    # --- actor capacity ---
    ("actor d=64",                   {'d_model_actor':64}),
    ("actor d=128 x2",               {'d_model_actor':128,'nlayers_actor':2}),
    # --- change #5: precision ---
    ("fp32 (no amp)",                {'amp':False}),
    # --- new hypothesis from the diagnostics: updates too timid ---
    ("looser clip 0.2/kl 0.03",      {'clip_ratio':0.2,'target_kl':0.03}),
]

def weight_response(r):
    prob, ntot, nact = r['problem'], r['weights_dim'], r['problem'].n_elements
    def gen(w, n=32):
        w = np.tile(np.array(w,float)/np.sum(w), (n,1))
        obs=[w[i].tolist() for i in range(n)]
        with torch.no_grad():
            for _ in range(nact):
                _,a,_ = r['actor'].sample_action(obs)
                for i,x in enumerate(a.tolist()): obs[i].append(x)
        d=[[int(v) for v in o[ntot:]] for o in obs]
        return np.array([prob.evaluate(x)[0] for x in d])
    light = gen([1,0,0,1,1]); stiff = gen([0,0,1,1,1])
    # relative separation between the two requested trade-offs
    return float(abs(stiff[:,0].mean()-light[:,0].mean())/max(light[:,0].mean(),1e-9)), \
           float(abs(stiff[:,2].mean()-light[:,2].mean())/max(light[:,2].mean(),1e-9))

out={}
for name,cfg in CONFIGS:
    hv=[]; ent=[]; ev=[]; wr=[]; wall=[]; ap=cp=0
    for s in range(SEEDS):
        r = variant.run_ppo(cfg, mv, s, epochs=EP, batch=B)
        hv.append(r['hv']); ent.append(float(np.mean(r['entropy'][-10:])))
        ev.append(float(np.mean(r['ev'][-50:]))); wall.append(r['wall'])
        wr.append(weight_response(r)); ap, cp = r['actor_params'], r['critic_params']
    wr=np.array(wr)
    out[name]=dict(hv_mean=float(np.mean(hv)), hv_std=float(np.std(hv)), hv=hv,
                   entropy=float(np.mean(ent)), ev=float(np.mean(ev)),
                   mass_resp=float(wr[:,0].mean()), freq_resp=float(wr[:,1].mean()),
                   wall=float(np.mean(wall)), actor_params=ap, critic_params=cp)
    print(f"{name:<26} hv={np.mean(hv):.4f}+-{np.std(hv):.4f}  ent={np.mean(ent):.3f}  "
          f"ev={np.mean(ev):+.2f}  massResp={wr[:,0].mean()*100:5.1f}%  "
          f"freqResp={wr[:,1].mean()*100:5.1f}%  {np.mean(wall):.0f}s", flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "sweep.json"), "w"), indent=1)
