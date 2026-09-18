"""Control: does a BETTER-trained repair actor help as a mutation operator?
Trains long-budget actors (domination reward and hv reward), then runs IMGA with each.
Usage: runner3.py --shard i --nshards N   (cwd must be the abl/ workdir)"""
import sys, os, json, time, argparse, warnings, numpy as np, torch, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
EXP=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP); sys.path.insert(0, os.path.dirname(EXP))
import repair_variant
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga
MV=np.load(os.path.join(EXP,"max_values.npy")); SEEDS=8
BASE=dict(num_epochs=500, mini_batch_size=32, gamma=0.999, learning_rate=3e-4, clip_ratio=0.05,
          update_iterations=5, target_kl=0.003, chunk_size=256, model_folder=None)
BASE['lambda']=0.9
VARIANTS=[("long-domination","ABL_LONG","domination"), ("long-hv","ABL_HV","hv")]

def build_jobs():
    J=[dict(kind='train', tag=t, ds=ds, reward=r) for t,ds,r in VARIANTS]
    for t,ds,_ in VARIANTS:
        for s in range(SEEDS):
            J.append(dict(kind='imga', tag=t, ds=ds, seed=s))
    return J

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--shard",type=int); ap.add_argument("--nshards",type=int); ap.add_argument("--mode",default="all")
    a=ap.parse_args(); jobs=build_jobs()
    outp=os.path.join(EXP, "data", f"ctl_shard_{a.shard}.jsonl"); done=set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)['job_id'])
            except Exception: pass
    with open(outp,"a") as f:
        for i,j in enumerate(jobs):
            if i % a.nshards != a.shard or i in done: continue
            if a.mode != 'all' and j['kind'] != a.mode: continue
            t0=time.perf_counter()
            try:
                if j['kind']=='train':
                    # 32,000 NFE of repair training, vs 9,600 for the actor used before
                    r=repair_variant.run_repair(
                        {'reward':j['reward'],
                         'save_actor':os.path.join('results', j['ds'], 'actor_repair_model.pth')},
                        MV, 0, epochs=1000, batch=32)
                    rec=dict(job_id=i, kind='train', tag=j['tag'], hv=r['hv'], NFE=r['NFE'],
                             wall=time.perf_counter()-t0)
                else:
                    np.random.seed(700+j['seed']); torch.manual_seed(700+j['seed'])
                    p=dict(BASE); p['date_str']=j['ds']; p['random_init_actor']=False
                    out=run_intelligent_mutation_ga(MV, p)
                    rec=dict(job_id=i, kind='imga', tag=j['tag'], seed=j['seed'],
                             hv=float(out[4][-1]), NFE=out[5], wall=time.perf_counter()-t0)
            except Exception as e:
                rec=dict(job_id=i, kind=j['kind'], tag=j['tag'], seed=j.get('seed'), error=repr(e))
            f.write(json.dumps(rec)+"\n"); f.flush()
            print(f"[ctl {a.shard}] {i:3d} {j['kind']:<6} {j['tag']:<18} "
                  f"{rec.get('hv','ERR')} {rec.get('wall',0):.0f}s", flush=True)
