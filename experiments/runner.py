"""Sharded job runner. Usage: runner.py --shard i --nshards N"""
import sys, os, json, time, argparse, warnings, numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import variant, repair_variant

MV = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "max_values.npy"))
FIX = {'fix_pe':True,'fix_position':True,'dropout':0.0,'amp':False}
SEEDS = 8

PPO_CONFIGS = [
    ("baseline",                  {}),
    ("entropy 0.01",              {'entropy_coef':0.01}),
    ("entropy 0.01 decay",        {'entropy_coef':0.01,'entropy_decay':True}),
    ("all fixes",                 dict(FIX)),
    ("all fixes + entropy decay", {**FIX,'entropy_coef':0.01,'entropy_decay':True}),
]
REPAIR_MODES = ['domination','front','hv']

def build_jobs():
    J=[]
    for name,cfg in PPO_CONFIGS:
        for s in range(SEEDS):
            J.append(dict(kind='ppo', name=name, cfg=cfg, seed=s, epochs=1000, batch=64))
    for m in REPAIR_MODES:
        for s in range(SEEDS):
            J.append(dict(kind='repair', name=f'repair {m}', cfg={'reward':m}, seed=s,
                          epochs=150, batch=32))
    for s in range(SEEDS):
        J.append(dict(kind='rs', name='random search', cfg={}, seed=s, epochs=1000, batch=64))
    # larger-budget repair arm (12,800 NFE), queued last
    for m in REPAIR_MODES:
        for s in range(6):
            J.append(dict(kind='repair', name=f'repair-big {m}', cfg={'reward':m}, seed=s,
                          epochs=400, batch=32))
    return J

def run_one(j):
    t0=time.perf_counter()
    if j['kind']=='ppo':
        r=variant.run_ppo(j['cfg'], MV, j['seed'], epochs=j['epochs'], batch=j['batch'])
        ent=np.array(r['entropy'])
        # epoch at which entropy first drops below half its starting value
        half=ent[:10].mean()/2
        below=np.where(ent<half)[0]
        return dict(hv=r['hv'], entropy_final=float(ent[-10:].mean()),
                    entropy_start=float(ent[:10].mean()),
                    collapse_epoch=int(below[0]) if len(below) else -1,
                    entropy_curve=[float(x) for x in ent[::20]],
                    ev=float(np.mean(r['ev'][-50:])), valid_frac=r['valid_frac'],
                    wall=time.perf_counter()-t0)
    if j['kind']=='repair':
        r=repair_variant.run_repair(j['cfg'], MV, j['seed'], epochs=j['epochs'], batch=j['batch'])
        return dict(hv=r['hv'], NFE=r['NFE'], hv_curve=r['hv_curve'][::50],
                    valid_frac=r['valid_frac'], wall=time.perf_counter()-t0)
    from optimization.random_search import run_random_search
    np.random.seed(5000+j['seed'])
    out=run_random_search({'num_epochs':j['epochs'],'mini_batch_size':j['batch']})
    return dict(hv=float(out[4][-1]), wall=time.perf_counter()-t0)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--shard",type=int); ap.add_argument("--nshards",type=int)
    a=ap.parse_args()
    jobs=build_jobs()
    outp=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"shard_{a.shard}.jsonl")
    done=set()
    if os.path.exists(outp):
        for line in open(outp):
            try: done.add(json.loads(line)['job_id'])
            except Exception: pass
    with open(outp,"a") as f:
        for i,j in enumerate(jobs):
            if i % a.nshards != a.shard or i in done: continue
            try:
                res=run_one(j)
                rec=dict(job_id=i, kind=j['kind'], name=j['name'], seed=j['seed'], **res)
            except Exception as e:
                rec=dict(job_id=i, kind=j['kind'], name=j['name'], seed=j['seed'], error=repr(e))
            f.write(json.dumps(rec)+"\n"); f.flush()
            print(f"[shard {a.shard}] {i:3d} {j['name']:<28} seed={j['seed']} "
                  f"{'hv=%.4f'%res['hv'] if 'hv' in locals().get('res',{}) else 'ERR'} "
                  f"{res.get('wall',0):.0f}s", flush=True)
