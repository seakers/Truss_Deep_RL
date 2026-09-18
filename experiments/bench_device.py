"""Is the repair PPO faster on CPU than on GPU? Every forward in its rollout is a
batch-of-1 pass through a 16-wide transformer, which is pure kernel-launch overhead on a
GPU. Run the same short training both ways and compare."""
import sys, os, time, warnings, numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))

MODE = sys.argv[1] if len(sys.argv) > 1 else 'gpu'
if MODE == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(1)
print(MODE, "cuda visible:", torch.cuda.is_available())
import repair_variant

MV = np.load(os.path.join(HERE, "max_values.npy"))
t0 = time.perf_counter()
r = repair_variant.run_repair({'reward': 'domination', 'obs_mode': 'design'},
                              MV, seed=0, epochs=40, batch=32)      # 1,280 NFE
dt = time.perf_counter() - t0
print("%s: %.1fs for 1280 NFE -> projected %.0fs for 32,000 NFE   hv=%.4f"
      % (MODE, dt, dt * 25, r['hv']))
