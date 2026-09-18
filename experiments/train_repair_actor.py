"""Train one repair actor + one synthesis actor into results/ABL/ for the ablation."""
import sys, os, warnings, numpy as np, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
EXP=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP))
from optimization.design_repair import run_design_repair_ppo
from optimization.ppo_rand_weights_state import run_ppo_rand_weights_state
MV=np.load(os.path.join(EXP,"max_values.npy"))
os.makedirs("results/ABL", exist_ok=True)
p=dict(num_epochs=300, mini_batch_size=32, gamma=0.999, learning_rate=3e-4, clip_ratio=0.05,
       update_iterations=5, target_kl=0.003, chunk_size=256, date_str='ABL', model_folder=None)
p['lambda']=0.9
np.random.seed(0)
r=run_ppo_rand_weights_state(MV, p, run_idx=0, best_hv_so_far=-np.inf)
print("synthesis actor trained, run HV =", r[-1])
np.random.seed(0)
r=run_design_repair_ppo(MV, p, run_idx=0, best_hv_so_far=-np.inf)
print("repair actor trained, run HV =", r[-1])
print("saved:", os.listdir("results/ABL"))
