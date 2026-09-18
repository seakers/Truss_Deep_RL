"""Tiny-budget smoke test of every round-4 code path before the real sweep launches."""
import sys, os, warnings, numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib; matplotlib.use("Agg")
import torch; torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
WORK = os.path.join(HERE, "r4work_smoke"); os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
import variant, repair_variant
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga
from optimization.genetic_algorithm import run_genetic_algorithm

MV = np.load(os.path.join(HERE, "max_values.npy"))
FIX = {'fix_pe': True, 'fix_position': True, 'dropout': 0.0, 'amp': False}

print("-- synthesis: defaults still reproduce the repo (hv_reset_period=None)")
r = variant.run_ppo({}, MV, 0, epochs=3, batch=8); print("   baseline hv", round(r['hv'], 4))

for name, cfg in [("reset=1", {**FIX, 'reward_mode': 'hv', 'hv_reset_period': 1}),
                  ("reset=5", {**FIX, 'reward_mode': 'hv', 'hv_reset_period': 5}),
                  ("binary",  {**FIX, 'reward_mode': 'hv', 'hv_reset_period': 1,
                               'hv_binary': True}),
                  ("legacy 'batch' alias", {**FIX, 'reward_mode': 'hv',
                                            'hv_archive': 'batch'})]:
    r = variant.run_ppo(cfg, MV, 0, epochs=3, batch=8)
    print("   %-22s hv %.4f  ev %.3g" % (name, r['hv'], np.mean(r['ev'])))

print("-- repair: obs_mode and archive reset, and the saved actor loads into IMGA")
for tag, cfg in [("SMOKE_FULL", {'reward': 'domination', 'obs_mode': 'full'}),
                 ("SMOKE_DES",  {'reward': 'domination', 'obs_mode': 'design'}),
                 ("SMOKE_HVB",  {'reward': 'hv', 'obs_mode': 'design',
                                 'hv_reset_period': 1})]:
    p = os.path.join(WORK, "results", tag, "actor_repair_model.pth")
    r = repair_variant.run_repair({**cfg, 'save_actor': p}, MV, seed=0, epochs=4, batch=8)
    print("   %-12s hv %.4f  saved=%s" % (tag, r['hv'], os.path.exists(p)))

base = dict(num_epochs=4, mini_batch_size=8, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None)
base['lambda'] = 0.9
for tag in ["SMOKE_FULL", "SMOKE_DES", "SMOKE_HVB"]:
    p = dict(base, date_str=tag, random_init_actor=False)
    out = run_intelligent_mutation_ga(MV, p)
    print("   IMGA with %-12s hv %.4f NFE %d" % (tag, out[4][-1], out[5]))
p = dict(base, date_str='NONE', random_init_actor=True)
out = run_intelligent_mutation_ga(MV, p); print("   IMGA random init hv %.4f" % out[4][-1])
out = run_genetic_algorithm(MV, dict(base)); print("   plain GA hv %.4f" % out[4][-1])
print("ALL SMOKE PATHS OK")
