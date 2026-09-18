"""Trained-vs-untrained hybrid ablation, n=8. Usage: runner2.py --shard i --nshards N
Must be run with cwd = the abl/ workdir so results/ABL/ resolves."""
import sys, os, json, time, argparse, warnings, numpy as np, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
EXP=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP); sys.path.insert(0, os.path.dirname(EXP))
from optimization.genetic_algorithm import run_genetic_algorithm
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga
from optimization.warm_start_ga import run_warm_start_ga

MV=np.load(os.path.join(EXP,"max_values.npy"))
SEEDS=8; EPOCHS=500; BATCH=32
BASE=dict(num_epochs=EPOCHS, mini_batch_size=BATCH, gamma=0.999, lam=0.9, **{'lambda':0.9},
          learning_rate=3e-4, clip_ratio=0.05, update_iterations=5, target_kl=0.003,
          chunk_size=256, date_str='ABL', model_folder=None)

ARMS=[("plain GA",              run_genetic_algorithm,        False),
      ("IMGA untrained",        run_intelligent_mutation_ga,  True),
      ("IMGA trained",          run_intelligent_mutation_ga,  False),
      ("WarmStart untrained",   run_warm_start_ga,            True),
      ("WarmStart trained",     run_warm_start_ga,            False)]

def build_jobs():
    return [dict(name=n, fn=i, rnd=r, seed=s)
            for i,(n,_,r) in enumerate(ARMS) for s in range(SEEDS)]

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--shard",type=int); ap.add_argument("--nshards",type=int)
    a=ap.parse_args(); jobs=build_jobs()
    outp=os.path.join(EXP, "data", f"abl_shard_{a.shard}.jsonl")
    done=set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)['job_id'])
            except Exception: pass
    with open(outp,"a") as f:
        for i,j in enumerate(jobs):
            if i % a.nshards != a.shard or i in done: continue
            np.random.seed(700+j['seed'])
            import torch; torch.manual_seed(700+j['seed'])
            p=dict(BASE); p['random_init_actor']=j['rnd']
            t0=time.perf_counter()
            try:
                out=ARMS[j['fn']][1](MV, p)
                rec=dict(job_id=i, name=j['name'], seed=j['seed'], hv=float(out[4][-1]),
                         NFE=out[5], wall=time.perf_counter()-t0)
            except Exception as e:
                rec=dict(job_id=i, name=j['name'], seed=j['seed'], error=repr(e))
            f.write(json.dumps(rec)+"\n"); f.flush()
            print(f"[abl {a.shard}] {i:3d} {j['name']:<22} seed={j['seed']} "
                  f"{rec.get('hv','ERR') if 'hv' not in rec else '%.4f'%rec['hv']} {rec.get('wall',0):.0f}s", flush=True)
