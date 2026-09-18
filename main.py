import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import pickle
import gc
import torch

from optimization.random_search import run_random_search
from optimization.genetic_algorithm import run_genetic_algorithm
from optimization.ppo_rand_weights_state import run_ppo_rand_weights_state
from optimization.ppo_goal_conditioned import run_ppo_goal_conditioned
from optimization.warm_start_ga import run_warm_start_ga
from optimization.design_repair import run_design_repair_ppo
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga
from optimization.warm_start_intelligent_ga import run_warm_start_intelligent_ga
from optimization.transfer_learning_ppo_state import run_transfer_learning_ppo_state
from optimization.transfer_learning_design_repair import run_transfer_learning_design_repair
from optimization.adaptive_operator_selection_ga import run_aos_ga

from evaluation.truss_problem import TrussProblem


class OptimizationMethod:
    """Class to define optimization methods and their properties"""
    def __init__(self, name, function, color, requires_max_values=False, param_overrides=None):
        self.name = name
        self.function = function
        self.color = color
        self.requires_max_values = requires_max_values
        self.param_overrides = param_overrides or {}


def initialize_methods():
    """Initialize all optimization methods with their properties"""
    methods = [
        OptimizationMethod("Random Search", run_random_search, "blue", False),
        OptimizationMethod("Genetic Algorithm", run_genetic_algorithm, "orange", True),
        OptimizationMethod("Design Synthesis PPO", run_ppo_rand_weights_state, "purple", True),
        OptimizationMethod("Warm Start GA", run_warm_start_ga, "green", True),
        OptimizationMethod("Design Repair PPO", run_design_repair_ppo, "red", True),
        OptimizationMethod("Intelligent Mutation GA", run_intelligent_mutation_ga, "cyan", True),
        OptimizationMethod("Warm Start Intelligent GA", run_warm_start_intelligent_ga, "magenta", True),
        # Conditions the policy on a target cell in objective space instead of on a
        # random preference weighting. Works, but loses to the hv-reward synthesis PPO
        # (experiments/RESULTS_ROUND3.md, section 4), so it is off by default.
        # OptimizationMethod("Goal-Conditioned PPO", run_ppo_goal_conditioned, "brown", True),
        # Ablations: same hybrid with a randomly initialised actor, to isolate what the
        # training contributes.
        # OptimizationMethod("Warm Start GA (untrained)", run_warm_start_ga, "olive", True,
        #                    {'random_init_actor': True}),
        # OptimizationMethod("Intelligent Mutation GA (untrained)", run_intelligent_mutation_ga, "teal", True,
        #                    {'random_init_actor': True}),
        # OptimizationMethod("AOS GA", run_aos_ga, "darkgreen", True),
        # OptimizationMethod("Transfer Learning PPO State", run_transfer_learning_ppo_state, "lime", True),
        # OptimizationMethod("Transfer Learning Design Repair", run_transfer_learning_design_repair, "darkred", True),
    ]
    return methods


def setup_pretrained_artifacts(params):
    """
    If params['pretrained_artifacts_path'] is set, load max_values and copy
    actor model files from that folder into the current run's results folder.
    This allows hybrid GA methods to find their actor checkpoints via the
    normal params['date_str'] path, with no changes to those method files.

    Returns max_values array if loaded, else None.
    """
    source_path = params.get('pretrained_artifacts_path', None)
    if source_path is None:
        return None

    dest_path = params['save_path']  # e.g., results/2026-05-01_.../
    max_values = None

    # --- Load max_values ---
    mv_src = os.path.join(source_path, 'max_values.npy')
    if os.path.exists(mv_src):
        max_values = np.load(mv_src)
        # Also save into current run folder so load_max_values() finds it too
        np.save(os.path.join(dest_path, 'max_values.npy'), max_values)
        print(f"[Pretrained] Loaded max_values from {mv_src}")
    else:
        print(f"[Pretrained] WARNING: max_values.npy not found at {mv_src}")

    # --- Copy actor model files ---
    actor_files = [
        'actor_model.pth',        # Used by Warm Start GA
        'actor_repair_model.pth', # Used by Intelligent Mutation GA (Truss-specific)
    ]
    for fname in actor_files:
        src = os.path.join(source_path, fname)
        dst = os.path.join(dest_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[Pretrained] Copied {fname} -> {dst}")
        else:
            print(f"[Pretrained] WARNING: {fname} not found at {src}")

    return max_values


def initialize_storage(methods):
    """Initialize storage dictionaries for all methods"""
    storage = {}
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        storage[method_key] = {
            'all_runs_des': [],
            'all_runs_obj': [],
            'all_runs_pareto_front_des': [],
            'all_runs_pareto_front_obj': [],
            'all_runs_hypervolumes': [],
            'all_runs_NFE': []
        }
    return storage


def store_results(storage, method_name, results):
    """Store results for a given method"""
    method_key = method_name.replace(" ", "_").lower()
    all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE = results

    storage[method_key]['all_runs_des'].append(all_des)
    storage[method_key]['all_runs_obj'].append(all_obj)
    storage[method_key]['all_runs_pareto_front_des'].append(pareto_front_des)
    storage[method_key]['all_runs_pareto_front_obj'].append(pareto_front_obj)
    storage[method_key]['all_runs_hypervolumes'].append(hypervolumes)
    storage[method_key]['all_runs_NFE'].append(NFE)


def truncate_data(storage, methods):
    """Truncate the methods which can have different lengths for various reasons
    Just doing all the methods as an extra safety measure"""

    for method in methods:
        key = method.name.replace(" ", "_").lower()
        if key not in storage or not storage[key]['all_runs_NFE']:
            continue

        nfe = storage[key]['all_runs_NFE']
        min_nfe = min(nfe)

        storage[key]['all_runs_des'] = [des[:min_nfe] for des in storage[key]['all_runs_des']]
        storage[key]['all_runs_obj'] = [obj[:min_nfe] for obj in storage[key]['all_runs_obj']]
        storage[key]['all_runs_pareto_front_des'] = [pf[:min_nfe] for pf in storage[key]['all_runs_pareto_front_des']]
        storage[key]['all_runs_pareto_front_obj'] = [pf[:min_nfe] for pf in storage[key]['all_runs_pareto_front_obj']]
        storage[key]['all_runs_hypervolumes'] = [hv[:min_nfe] for hv in storage[key]['all_runs_hypervolumes']]
        storage[key]['all_runs_NFE'] = [min_nfe for _ in storage[key]['all_runs_NFE']]

    return storage


def calculate_statistics(hypervolumes_array):
    """Calculate statistics for hypervolumes"""
    return {
        'max': np.max(hypervolumes_array, axis=0),
        'min': np.min(hypervolumes_array, axis=0),
        'median': np.median(hypervolumes_array, axis=0),
        'q25': np.percentile(hypervolumes_array, 25, axis=0),
        'q75': np.percentile(hypervolumes_array, 75, axis=0)
    }


def plot_hypervolumes(storage, methods, params):
    """Plot hypervolume comparison for all methods"""
    plt.figure(figsize=(14, 8))

    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        if method_key not in storage or not storage[method_key]['all_runs_hypervolumes']:
            continue

        hypervolumes = np.array(storage[method_key]['all_runs_hypervolumes'])

        if hypervolumes.size > 0:
            stats = calculate_statistics(hypervolumes)

            # plt.plot(stats['max'], label=f'{method.name} Max', color=method.color, linestyle='--', alpha=0.8)
            # plt.plot(stats['min'], label=f'{method.name} Min', color=method.color, linestyle=':', alpha=0.8)
            for run_hv in hypervolumes:
                plt.plot(run_hv, color=method.color, linewidth=0.6, alpha=0.25)
            plt.plot(stats['median'], label=f'{method.name} Median (n={len(hypervolumes)})',
                     color=method.color, linewidth=2)
            plt.fill_between(range(len(stats['q25'])), stats['q25'], stats['q75'],
                             color=method.color, alpha=0.1)

    plt.title('Hypervolume Comparison Across All Methods', fontsize='x-large')
    plt.xlabel('Iterations', fontsize='x-large')
    plt.ylabel('Hypervolume', fontsize='x-large')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-large')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{params['date_str']}/hypervolume_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_configurations(storage, methods, params, max_designs=5):
    """Visualize configurations for final Pareto front designs from each method"""
    vis_problem = TrussProblem()

    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        method_safe_name = method.name.replace(" ", "_").lower()

        pareto_designs = storage[method_key]['all_runs_pareto_front_des']

        if (pareto_designs and len(pareto_designs) > 0 and
                pareto_designs[-1] and len(pareto_designs[-1]) > 0 and
                len(pareto_designs[-1][-1]) > 0):

            final_pareto_designs = pareto_designs[-1][-1]
            print(f"\nVisualizing {min(len(final_pareto_designs), max_designs)} designs for {method.name}...")

            for i, design in enumerate(final_pareto_designs):
                if isinstance(design, np.ndarray):
                    design = design.tolist()

                filename = f"results/{params['date_str']}/{method_safe_name}_design_{i}"

                try:
                    vis_problem.render(design, save_name=filename)
                except Exception as e:
                    print(f"Error visualizing {method.name} design {i}: {e}")

                if i >= max_designs - 1:
                    break


def plot_pareto_front_3d(storage, methods, params):
    """Create and save a 3D scatter plot of final Pareto fronts in objective space."""
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    plotted = False
    for method in methods:
        key = method.name.replace(" ", "_").lower()
        objs_runs = storage[key]['all_runs_pareto_front_obj']

        if not objs_runs or len(objs_runs) == 0:
            continue
        last_run = objs_runs[-1]
        if last_run is None:
            continue

        final_pfront = last_run[-1]
        if final_pfront is None:
            continue

        arr = np.array(final_pfront)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] >= 3:
            pts = arr[:, :3]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       label=method.name, color=method.color, s=40, alpha=0.9, edgecolors='k')
            plotted = True
        elif arr.shape[1] == 2:
            ax.scatter(arr[:, 0], arr[:, 1], np.zeros(arr.shape[0]),
                       label=method.name, color=method.color)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel('Mass (lbs)')
    ax.set_ylabel('Compliance')
    ax.set_zlabel('Neg Frequency (Hz)')
    ax.set_title('Pareto Fronts: Truss Optimization (3D)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"results/{params['date_str']}/pareto_front_3d.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_all_data(storage, methods, params):
    """Save all data to pickle file"""
    save_data = {}
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        save_data[method.name] = storage[method_key]
    save_data['params'] = params
    with open(f"results/{params['date_str']}/all_data.pkl", "wb") as f:
        pickle.dump(save_data, f)


def save_max_values(max_values, params):
    """Save max_values to disk so they persist across memory clears."""
    folder = f"results/{params['date_str']}"
    os.makedirs(folder, exist_ok=True)
    filepath = f"{folder}/max_values.npy"
    np.save(filepath, max_values)
    print(f"max_values saved to {filepath}")
    return filepath


def load_max_values(params):
    """Load max_values from disk."""
    filepath = f"results/{params['date_str']}/max_values.npy"
    if os.path.exists(filepath):
        max_values = np.load(filepath)
        print(f"max_values loaded from {filepath}")
        return max_values
    return None


def save_hv_and_pareto(method_name, run_idx, hypervolumes, pareto_front_obj, params):
    """
    Save hypervolumes and pareto front points to a separate lightweight file.
    Allows re-loading just HV/Pareto data for plotting without loading all designs.
    """
    folder = f"results/{params['date_str']}/hv_pareto_data"
    os.makedirs(folder, exist_ok=True)
    method_key = method_name.replace(" ", "_").lower()
    filepath = f"{folder}/{method_key}_run_{run_idx}.pkl"
    data = {
        'hypervolumes': hypervolumes,
        'pareto_front_obj': pareto_front_obj,
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    return filepath


def load_hv_and_pareto_for_plotting(methods, num_runs, params):
    """
    Load only HV and pareto data for final plotting, avoiding full data reload.
    Returns a lightweight storage dict sufficient for plot_hypervolumes.
    """
    folder = f"results/{params['date_str']}/hv_pareto_data"
    storage = {}

    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        storage[method_key] = {
            'all_runs_hypervolumes': [],
            'all_runs_pareto_front_obj': [],
            'all_runs_NFE': [],
            'all_runs_des': [],
            'all_runs_obj': [],
            'all_runs_pareto_front_des': [],
        }
        for run_idx in range(num_runs):
            filepath = f"{folder}/{method_key}_run_{run_idx}.pkl"
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                storage[method_key]['all_runs_hypervolumes'].append(data['hypervolumes'])
                storage[method_key]['all_runs_pareto_front_obj'].append(data['pareto_front_obj'])
                storage[method_key]['all_runs_NFE'].append(len(data['hypervolumes']))
            else:
                print(f"Warning: HV/Pareto file not found: {filepath}")

    return storage


def force_clear_memory():
    """Aggressively clear memory between methods/runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def save_single_run(method_name, run_idx, results, params):
    """Saves data for a single run to disk and returns the filename.
    Also saves HV and Pareto data separately for lightweight re-loading."""
    folder = f"results/{params['date_str']}/run_data"
    os.makedirs(folder, exist_ok=True)

    method_key = method_name.replace(" ", "_").lower()
    filename = f"{folder}/{method_key}_run_{run_idx}.pkl"

    all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE = results

    with open(filename, "wb") as f:
        pickle.dump(results, f)

    # Also save HV and Pareto separately (lightweight)
    save_hv_and_pareto(method_name, run_idx, hypervolumes, pareto_front_obj, params)

    return filename


def _model_key(method_name: str) -> str:
    """Return the model filename for a given method. Truss-specific filenames."""
    if method_name == "Design Synthesis PPO":
        return "actor_model.pth"
    elif method_name == "Design Repair PPO":
        return "actor_repair_model.pth"   # Truss-specific (not spacecraft)
    return ""


def _copy_best_model(method_name: str, run_idx: int, params: dict):
    """
    Copy the freshly-saved model from the current run into a
    'best_<filename>' file so it survives subsequent runs.
    """
    fname = _model_key(method_name)
    if not fname:
        return
    folder = f"results/{params['date_str']}"
    src = os.path.join(folder, fname)
    dst = os.path.join(folder, f"best_{fname}")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  [BestModel] Copied {fname} -> best_{fname} (run {run_idx})")
    else:
        print(f"  [BestModel] WARNING: {fname} not found at {src}")


def main():
    # Configuration
    num_runs = 6
    params = {
        'num_epochs': 1000,
        'mini_batch_size': 64,
        'gamma': 0.999,
        'lambda': 0.9,
        'learning_rate': 0.0003,
        'clip_ratio': 0.05,
        'update_iterations': 5,
        'target_kl': 0.003,
        'chunk_size': 256,
        # 'scalarized' is the original weighted-sum reward; 'hv' pays a design for the
        # hypervolume it adds to a running front, reset every hv_reset_period epochs.
        'reward_mode': 'hv',
        'hv_reset_period': 3,
        'hv_scale': 200.0,
        # Repair PPO observation. 'design' makes the state used for the PPO update
        # identical to the one the policy acted in and the one the hybrid GA deploys
        # it on; 'full' reproduces the old mismatched behaviour.
        'obs_mode': 'design',
        'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'model_folder': None,
        # Transfer learning params
        'transfer_pretrain_fraction': 0.3,
        'transfer_finetune_lr': 0.00015,
        'transfer_load_factors': [0.7, 0.85, 1.15, 1.3],
        # Set to None to run everything from scratch.
        # Set to a results folder string to load max_values and actor models from it.
        'pretrained_artifacts_path': 'results/2026-06-15_10-09-18',
    }

    os.makedirs(f"results/{params['date_str']}/", exist_ok=True)
    params['save_path'] = f"results/{params['date_str']}/"

    methods = initialize_methods()

    # Track file paths only — no large data in memory
    run_files = {method.name: [] for method in methods}

    # ---- Load pre-trained artifacts if a source path is configured ----
    max_values = setup_pretrained_artifacts(params)
    if max_values is not None:
        print(f"[Pretrained] max_values ready: {max_values}")
    else:
        max_values = None  # Will be set by Random Search runs

    # ================================================================
    # MAIN LOOP: iterate methods first, then runs (like main_adj)
    # ================================================================
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running method: {method.name} for {num_runs} run(s)")
        print(f"{'='*60}")

        best_hv = -np.inf           # Track best final HV across runs (for RL/repair models)
        all_run_max_values = []     # Collect max_values from each Random Search run

        saved_params = {k: params.get(k) for k in method.param_overrides}
        params.update(method.param_overrides)

        for run_idx in range(num_runs):
            print(f"\n--- {method.name} | Run {run_idx + 1}/{num_runs} ---")

            # --- Before each method: ensure memory is clean ---
            force_clear_memory()

            # --- Run the method ---
            if method.name == "Random Search":
                result = method.function(params)
                (all_des, all_obj, pareto_front_des,
                 pareto_front_obj, hypervolumes, NFE, run_max_values) = result
                all_run_max_values.append(run_max_values)

                filename = save_single_run(
                    method.name, run_idx,
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE),
                    params
                )
                run_files[method.name].append(filename)
                print(f"Results saved to {filename}")
                force_clear_memory()

            elif method.requires_max_values and max_values is not None:
                # Load max_values from disk if cleared from memory
                if max_values is None:
                    max_values = load_max_values(params)
                    if max_values is None:
                        raise ValueError(
                            f"Method {method.name} requires max_values but none found. "
                            "Ensure Random Search runs first."
                        )

                # Pass best_hv so the method knows what to beat for model saving
                if method.name in ("Design Synthesis PPO", "Design Repair PPO"):
                    result = method.function(
                        max_values, params,
                        run_idx=run_idx, best_hv_so_far=best_hv
                    )
                else:
                    result = method.function(max_values, params)

                if result is None:
                    print(f"Warning: {method.name} run {run_idx} returned None")
                    continue

                # Methods that track best model return best_hv alongside results
                if method.name in ("Design Synthesis PPO", "Design Repair PPO"):
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE, run_best_hv) = result

                    # Update global best HV for model selection
                    if run_best_hv > best_hv:
                        best_hv = run_best_hv
                        print(f"  New best HV for {method.name}: {best_hv:.6f} (run {run_idx})")
                        # Copy best model to a persistent "best_across_runs" file
                        _copy_best_model(method.name, run_idx, params)
                else:
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE) = result

                filename = save_single_run(
                    method.name, run_idx,
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE),
                    params
                )
                run_files[method.name].append(filename)
                print(f"Results saved to {filename}")
                force_clear_memory()

            else:
                if method.requires_max_values and max_values is None:
                    print(f"Skipping {method.name}: max_values not yet available.")
                    continue

            print(f"Memory cleared after {method.name} run {run_idx}")

        for k, v in saved_params.items():
            if v is None:
                params.pop(k, None)
            else:
                params[k] = v

        # ── Post-all-runs processing for this method ──────────────────────────
        if method.name == "Random Search":
            # Combine max values across all runs (element-wise max)
            combined_max = np.max(np.stack(all_run_max_values, axis=0), axis=0)
            max_values = combined_max
            save_max_values(max_values, params)
            print(f"\nCombined max_values across {num_runs} run(s): {max_values}")

        if method.name in ("Design Synthesis PPO", "Design Repair PPO") and best_hv > -np.inf:
            print(f"\nBest HV for {method.name} across all runs: {best_hv:.6f}")
            print(f"Best model saved to: results/{params['date_str']}/best_{_model_key(method.name)}")

    # ================================================================
    # POST-PROCESSING: Reload only what's needed for plots/stats
    # ================================================================
    print(f"\n{'='*60}")
    print("All runs complete. Loading HV/Pareto data for final analysis...")
    print(f"{'='*60}")

    force_clear_memory()

    # Option A: Lightweight reload (HV + Pareto only, for plotting)
    storage_light = load_hv_and_pareto_for_plotting(methods, num_runs, params)
    storage_light = truncate_data(storage_light, methods)
    plot_hypervolumes(storage_light, methods, params)
    del storage_light
    force_clear_memory()

    # Option B: Full reload for save_all_data and visualization (one method at a time)
    print("Building full dataset for save_all_data...")
    storage = initialize_storage(methods)
    for method in methods:
        for fpath in run_files[method.name]:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            store_results(storage, method.name, data)
            del data
            force_clear_memory()

    storage = truncate_data(storage, methods)
    save_all_data(storage, methods, params)
    plot_pareto_front_3d(storage, methods, params)

    # Visualization: one method at a time to limit memory
    for method in methods:
        try:
            visualize_configurations(storage, [method], params, max_designs=5)
        except Exception as e:
            print(f"Error visualizing {method.name}: {e}")
        force_clear_memory()

    del storage
    force_clear_memory()

    print(f"\nResults saved to: results/{params['date_str']}/")
    print("Generated files:")
    print("  - run_data/*.pkl          : Raw results per method per run")
    print("  - hv_pareto_data/*.pkl    : Lightweight HV/Pareto per method per run")
    print("  - max_values.npy          : Normalization constants from Random Search")
    print("  - all_data.pkl            : Complete aggregated results")
    print("  - hypervolume_comparison.png")
    print("  - pareto_front_3d.png")
    print("  - AOS operator statistics (if AOS GA was run)")


if __name__ == "__main__":
    main()