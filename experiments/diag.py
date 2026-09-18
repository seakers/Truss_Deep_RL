import os, sys, json, warnings, numpy as np, torch
warnings.filterwarnings("ignore"); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import variant
mv = np.load(os.path.join(HERE, "max_values.npy"))
EP, B = 300, 32
res = []
for s in range(3):
    r = variant.run_ppo({}, mv, s, epochs=EP, batch=B)
    res.append(r)
    print(f"seed {s}: hv={r['hv']:.4f} valid={r['valid_frac']:.2f} wall={r['wall']:.0f}s "
          f"skipped={r['skipped']}/{r['steps']} ({r['skipped']/r['steps']*100:.1f}%)")

ent = np.array([r['entropy'] for r in res]); ev = np.array([r['ev'] for r in res])
print(f"\nActor params {res[0]['actor_params']}, critic params {res[0]['critic_params']}")
print(f"\nENTROPY  (max possible = ln 50 = {np.log(50):.3f})")
for lo,hi in [(0,10),(10,50),(50,100),(100,200),(200,300)]:
    print(f"   epochs {lo:>3}-{hi:<3}  {ent[:,lo:hi].mean():.3f}")
print(f"   final 10 epochs   {ent[:,-10:].mean():.3f}   -> {ent[:,-10:].mean()/np.log(50)*100:.0f}% of maximum")

print(f"\nCRITIC EXPLAINED VARIANCE  (1.0 = perfect, 0.0 = no better than predicting the mean, <0 = worse)")
for lo,hi in [(0,10),(10,50),(50,100),(100,200),(200,300)]:
    print(f"   epochs {lo:>3}-{hi:<3}  {ev[:,lo:hi].mean():+.3f}")
print(f"   final 50 epochs   {ev[:,-50:].mean():+.3f}")

# --- does the policy actually respond to the preference weights it is given? ---
print(f"\nWEIGHT CONDITIONING: designs generated under opposite preferences")
prob = res[0]['problem']; ntot = res[0]['weights_dim']; nact = prob.n_elements
def gen(actor, wvec, n=32):
    w = np.tile(np.array(wvec,float)/np.sum(wvec), (n,1))
    obs = [w[i].tolist() for i in range(n)]
    with torch.no_grad():
        for _ in range(nact):
            _, a, _ = actor.sample_action(obs)
            for i,x in enumerate(a.tolist()): obs[i].append(x)
    des = [[int(v) for v in o[ntot:]] for o in obs]
    return np.array([prob.evaluate(d)[0] for d in des])
rows=[]
for r in res:
    mass_only = gen(r['actor'], [1,0,0,1,1])     # care only about mass
    freq_only = gen(r['actor'], [0,0,1,1,1])     # care only about frequency
    rows.append((mass_only[:,0].mean(), freq_only[:,0].mean(),
                 mass_only[:,2].mean(), freq_only[:,2].mean()))
rows=np.array(rows)
print(f"   mean MASS      asked for light={rows[:,0].mean():8.1f}   asked for stiff={rows[:,1].mean():8.1f}   diff={rows[:,1].mean()-rows[:,0].mean():+8.1f}")
print(f"   mean FREQUENCY asked for light={rows[:,2].mean():8.4f}   asked for stiff={rows[:,3].mean():8.4f}   diff={rows[:,3].mean()-rows[:,2].mean():+8.4f}")

json.dump({'entropy':ent.tolist(),'ev':ev.tolist(),
           'hv':[r['hv'] for r in res],'skipped':[r['skipped'] for r in res],
           'steps':[r['steps'] for r in res]}, open(os.path.join(HERE, "data", "diag.json"), "w"))
