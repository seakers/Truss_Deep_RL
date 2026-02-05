import itertools
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Import your optimization methods
from optimization.random_search import run_random_search
from optimization.ppo_rand_weights_uninformed import run_ppo_rand_weights_uninformed

def run_hyperparameter_sweep():
    # --- 1. Base Configuration ---
    base_params = {
        'num_epochs': 200,          
        'mini_batch_size': 8,
        'update_iterations': 10,
        'save_interval': 10,
        'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'model_folder': None,
        # Defaults (will be overwritten)
        'gamma': 0.99,
        'lambda': 0.95,
        'learning_rate': 1e-3,
        'target_kl': 0.01,
        'clip_ratio': 0.2,
        'run_num': 0
    }

    # Create base results directory
    base_dir = f"results/sweep_{base_params['date_str']}"
    os.makedirs(base_dir, exist_ok=True)
    print(f"Starting Hyperparameter Sweep. Results will be saved to: {base_dir}")

    # --- 2. Establish Baseline & Max Values ---
    print("\n" + "="*50)
    print("STEP 1: Establishing Baseline with Random Search")
    print("="*50 + "\n")
    
    rs_params = base_params.copy()
    rs_params['date_str'] = f"{base_dir.split('/')[-1]}/baseline_random_search"
    os.makedirs(f"results/{rs_params['date_str']}", exist_ok=True)

    # Capture Max Values
    _, _, _, _, _, _, max_values = run_random_search(rs_params)
    print(f"\nMax Values determined: {max_values}")

    # --- 3. Define Hyperparameter Grid ---
    grid = {
        'gamma': [0.99, 0.999],
        'lambda': [0.9, 0.95],
        'learning_rate': [3e-4, 1e-4],
        'target_kl': [0.003, 0.01],
        'clip_ratio': [0.05, 0.1],
        'run_num': [0,1,2,3,4]
    }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_runs = len(combinations)
    all_sweep_results = {} # To store data for comparison

    # --- 4. Iterate through Sweep ---
    for i, config in enumerate(combinations):
        print("\n" + "-"*50)
        print(f"Run {i+1}/{total_runs} | Config: {config}")
        print("-"*50 + "\n")

        current_params = base_params.copy()
        current_params.update(config)

        # Create unique directory name
        config_name = (
            f"g{config['gamma']}_"
            f"l{config['lambda']}_"
            f"lr{config['learning_rate']}_"
            f"kl{config['target_kl']}_"
            f"clip{config['clip_ratio']}_"
            f"run{config['run_num']}"
        )
        
        current_params['date_str'] = f"{base_dir.split('/')[-1]}/ppo_{config_name}"
        save_path = f"results/{current_params['date_str']}"
        os.makedirs(save_path, exist_ok=True)

        try:
            # Run PPO and capture outputs
            _, _, _, _, hypervolumes, _ = run_ppo_rand_weights_uninformed(max_values, current_params)
            
            # A. Save Hypervolume Data
            np.save(f"{save_path}/hypervolumes.npy", np.array(hypervolumes))
            
            # # B. Plot Individual Run
            # plt.figure(figsize=(10, 6))
            # plt.plot(hypervolumes, label=f"HV (PPO {config_name})", color='green')
            # plt.xlabel("Evaluations")
            # plt.ylabel("Hypervolume")
            # plt.title(f"Training Progress: {config_name}")
            # plt.legend()
            # plt.grid(True, alpha=0.3)
            # plt.savefig(f"{save_path}/hv_plot.png")
            # plt.close()

            # C. Store for Comparison
            all_sweep_results[config_name] = hypervolumes

            # Save config log
            with open(f"{save_path}/config_log.txt", "w") as f:
                f.write(str(current_params))
                
        except Exception as e:
            print(f"!!! Error in run {i+1}: {e}")
            continue

    # --- 5. Generate Comparison Plot ---
    print("\n" + "="*50)
    print("Generating Comparison Plot...")

    hv_finals = []
    
    plt.figure(figsize=(12, 8))
    
    for name, hv_data in all_sweep_results.items():
        # Optional: Smooth lines if data is noisy
        hv_finals.append((name, hv_data[-1]))
        plt.plot(hv_data, label=name, alpha=0.7, linewidth=1.5)

    hv_finals.sort(key=lambda x: x[1], reverse=True)
    print("\nHypervolume Final Results:")
    for name, final_hv in hv_finals:
        print(f"{name}: {final_hv:.4f}")

    plt.xlabel("Evaluations")
    plt.ylabel("Hypervolume")
    plt.title(f"Hyperparameter Sweep Comparison ({total_runs} Configs)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Save comparison in the base directory
    plt.savefig(f"{base_dir}/comparison_plot.png", dpi=300)
    plt.close()

    # Save dictionary of all results
    with open(f"{base_dir}/all_sweep_results.pkl", "wb") as f:
        pickle.dump(all_sweep_results, f)

    print(f"Sweep Complete. Comparison saved to {base_dir}/comparison_plot.png")
    print("="*50)

if __name__ == "__main__":
    run_hyperparameter_sweep()