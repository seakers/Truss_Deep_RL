import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import pickle
import gc
import argparse
import torch

from optimization.random_search import run_random_search
from optimization.genetic_algorithm import run_genetic_algorithm
from optimization.design_synthesis import run_ppo_rand_weights_state
from optimization.warm_start_ga import run_warm_start_ga
from optimization.design_repair import run_design_repair_ppo
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga
from optimization.warm_start_intelligent_ga import run_warm_start_intelligent_ga
from optimization.transfer_learning import (
    run_transfer_learning_ppo_state,
    run_transfer_learning_design_repair,
)
from evaluation.truss_problem import TrussProblem
from utils.visualization import (plot_hypervolumes, plot_pareto_front_3d,
                                 visualize_configurations)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_arg_parser():
    """CLI overrides for the params dict. Every default mirrors the original
    hardcoded params so `python main.py` behaves identically, EXCEPT
    pretrained-artifacts-path which defaults to None (safe for HPC)."""
    p = argparse.ArgumentParser(description="Truss design optimization runner")

    # Run control
    p.add_argument('--num-runs', type=int, default=10)
    p.add_argument('--methods', type=str, default=None,
                   help="Comma-separated subset of method names to run "
                        "(default: all). E.g. 'Design Synthesis PPO,Design Repair PPO'")

    # PPO / optimization hyperparameters (defaults = original params)
    p.add_argument('--num-epochs',        type=int,   default=20)
    p.add_argument('--mini-batch-size',   type=int,   default=4)
    p.add_argument('--gamma',             type=float, default=0.999)
    p.add_argument('--lambda-gae',        type=float, default=0.9, dest='lambda_gae')
    p.add_argument('--learning-rate',     type=float, default=0.0003)
    p.add_argument('--clip-ratio',        type=float, default=0.05)
    p.add_argument('--update-iterations', type=int,   default=5)
    p.add_argument('--target-kl',         type=float, default=0.003)
    p.add_argument('--chunk-size',        type=int,   default=256)

    # Transfer-learning params (kept as fixed defaults; swept separately)
    p.add_argument('--load-node',   type=int,   default=16)
    p.add_argument('--load-factor', type=float, default=1.0)
    p.add_argument('--transfer-pretrain-fraction', type=float, default=0.3)
    p.add_argument('--transfer-finetune-lr',       type=float, default=0.00015)
    p.add_argument('--transfer-load-factors', type=str, default="0.7,0.85,1.15,1.3",
                   help="Comma-separated load factors for transfer sweep.")

    # Truss-specific
    p.add_argument('--overlap-constraint', dest='overlap_constraint',
                   action='store_true', default=True)
    p.add_argument('--no-overlap-constraint', dest='overlap_constraint',
                   action='store_false')

    # I/O
    p.add_argument('--model-folder', type=str, default=None,
                   help="Resume/warm-start checkpoint folder")
    p.add_argument('--date-str', type=str,
                   default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                   help="Override the results/<date_str> output dir "
                        "(use a stable tag per job for reproducibility)")
    p.add_argument('--results-root', type=str, default='results',
                   help="Root dir for outputs (point at $SCRATCH on HPC)")
    p.add_argument('--max-values-path', type=str, default=None,
                   help="Path to a precomputed max_values.npy (from a prior Random Search).")
    p.add_argument('--pretrained-artifacts-path', type=str, default=None,
                   help="Folder to load max_values + actor models from. "
                        "Default None = run from scratch (safe for HPC).")
    return p


def params_from_args(args):
    """Build the params dict from parsed CLI args."""
    return {
        'num_epochs':        args.num_epochs,
        'mini_batch_size':   args.mini_batch_size,
        'gamma':             args.gamma,
        'lambda':            args.lambda_gae,
        'learning_rate':     args.learning_rate,
        'clip_ratio':        args.clip_ratio,
        'update_iterations': args.update_iterations,
        'target_kl':         args.target_kl,
        'chunk_size':        args.chunk_size,
        'date_str':          args.date_str,
        'model_folder':      args.model_folder,
        # Transfer learning params
        'load_node':                   args.load_node,
        'load_factor':                 args.load_factor,
        'transfer_pretrain_fraction':  args.transfer_pretrain_fraction,
        'transfer_finetune_lr':        args.transfer_finetune_lr,
        'transfer_load_factors':       [float(x) for x in
                                        args.transfer_load_factors.split(',') if x != ''],
        'pretrained_artifacts_path':   args.pretrained_artifacts_path,
        'overlap_constraint':          args.overlap_constraint,
        'max_values_path':             args.max_values_path,
    }


class OptimizationMethod:
    """Class to define optimization methods and their properties"""
    def __init__(self, name, function, color, requires_max_values=False):
        self.name = name
        self.function = function
        self.color = color
        self.requires_max_values = requires_max_values


def initialize_methods():
    """Initialize all optimization methods with their properties"""
    methods = [
        OptimizationMethod("Random Search", run_random_search, "blue", False),
        OptimizationMethod("Genetic Algorithm", run_genetic_algorithm, "orange", True),
        OptimizationMethod("Design Synthesis PPO", run_ppo_rand_weights_state, "green", True),
        OptimizationMethod("Design Repair PPO", run_design_repair_ppo, "red", True),
        OptimizationMethod("Intelligent Mutation GA", run_intelligent_mutation_ga, "purple", True),
        OptimizationMethod("Warm Start GA", run_warm_start_ga, "brown", True),
        OptimizationMethod("Warm Start Intelligent GA", run_warm_start_intelligent_ga, "pink", True),
        OptimizationMethod("Transfer Synthesis PPO", run_transfer_learning_ppo_state, "cyan", True),
        OptimizationMethod("Transfer Repair PPO", run_transfer_learning_design_repair, "magenta", True),
    ]
    return methods


# PPO methods that take run_idx/best_hv_so_far and return a 7-tuple with best_hv
_MODEL_TRACKING = ("Design Synthesis PPO", "Design Repair PPO",
                   "Transfer Learning PPO State", "Transfer Learning Design Repair")


def setup_pretrained_artifacts(params):
    """If params['pretrained_artifacts_path'] is set, load max_values and copy
    actor model files from that folder into the current run's results folder."""
    source_path = params.get('pretrained_artifacts_path', None)
    if source_path is None:
        return None

    dest_path = params['save_path']
    max_values = None

    mv_src = os.path.join(source_path, 'max_values.npy')
    if os.path.exists(mv_src):
        max_values = np.load(mv_src)
        np.save(os.path.join(dest_path, 'max_values.npy'), max_values)
        print(f"[Pretrained] Loaded max_values from {mv_src}")
    else:
        print(f"[Pretrained] WARNING: max_values.npy not found at {mv_src}")

    actor_files = ['actor_model.pth', 'actor_repair_model.pth']
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
    storage = {}
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        storage[method_key] = {
            'all_runs_des': [], 'all_runs_obj': [],
            'all_runs_pareto_front_des': [], 'all_runs_pareto_front_obj': [],
            'all_runs_hypervolumes': [], 'all_runs_NFE': []
        }
    return storage


def store_results(storage, method_name, results):
    method_key = method_name.replace(" ", "_").lower()
    all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE = results
    storage[method_key]['all_runs_des'].append(all_des)
    storage[method_key]['all_runs_obj'].append(all_obj)
    storage[method_key]['all_runs_pareto_front_des'].append(pareto_front_des)
    storage[method_key]['all_runs_pareto_front_obj'].append(pareto_front_obj)
    storage[method_key]['all_runs_hypervolumes'].append(hypervolumes)
    storage[method_key]['all_runs_NFE'].append(NFE)


def truncate_data(storage, methods):
    """Truncate methods which can have different lengths across runs."""
    for method in methods:
        key = method.name.replace(" ", "_").lower()
        if key not in storage or not storage[key]['all_runs_NFE']:
            continue
        nfe = storage[key]['all_runs_NFE']
        min_nfe = min(nfe)
        storage[key]['all_runs_des'] = [d[:min_nfe] for d in storage[key]['all_runs_des']]
        storage[key]['all_runs_obj'] = [o[:min_nfe] for o in storage[key]['all_runs_obj']]
        storage[key]['all_runs_pareto_front_des'] = [pf[:min_nfe] for pf in storage[key]['all_runs_pareto_front_des']]
        storage[key]['all_runs_pareto_front_obj'] = [pf[:min_nfe] for pf in storage[key]['all_runs_pareto_front_obj']]
        storage[key]['all_runs_hypervolumes'] = [hv[:min_nfe] for hv in storage[key]['all_runs_hypervolumes']]
        storage[key]['all_runs_NFE'] = [min_nfe for _ in storage[key]['all_runs_NFE']]
    return storage


def calculate_statistics(hypervolumes_array):
    return {
        'max':    np.max(hypervolumes_array, axis=0),
        'min':    np.min(hypervolumes_array, axis=0),
        'median': np.median(hypervolumes_array, axis=0),
        'q25':    np.percentile(hypervolumes_array, 25, axis=0),
        'q75':    np.percentile(hypervolumes_array, 75, axis=0),
    }


def save_all_data(storage, methods, params):
    save_data = {}
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        save_data[method.name] = storage[method_key]
    save_data['params'] = params
    with open(f"{params['results_root']}/{params['date_str']}/all_data.pkl", "wb") as f:
        pickle.dump(save_data, f)


def save_max_values(max_values, params):
    folder = f"{params['results_root']}/{params['date_str']}"
    os.makedirs(folder, exist_ok=True)
    filepath = f"{folder}/max_values.npy"
    np.save(filepath, max_values)
    print(f"max_values saved to {filepath}")
    return filepath


def load_max_values(params):
    # Prefer an explicit --max-values-path if provided (sweep/tune jobs)
    mv_path = params.get('max_values_path', None)
    if mv_path and os.path.exists(mv_path):
        print(f"max_values loaded from {mv_path}")
        return np.load(mv_path)
    filepath = f"{params['results_root']}/{params['date_str']}/max_values.npy"
    if os.path.exists(filepath):
        print(f"max_values loaded from {filepath}")
        return np.load(filepath)
    return None


def save_hv_and_pareto(method_name, run_idx, hypervolumes, pareto_front_obj, params):
    folder = f"{params['results_root']}/{params['date_str']}/hv_pareto_data"
    os.makedirs(folder, exist_ok=True)
    method_key = method_name.replace(" ", "_").lower()
    filepath = f"{folder}/{method_key}_run_{run_idx}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump({'hypervolumes': hypervolumes, 'pareto_front_obj': pareto_front_obj}, f)
    return filepath


def load_hv_and_pareto_for_plotting(methods, num_runs, params):
    folder = f"{params['results_root']}/{params['date_str']}/hv_pareto_data"
    storage = {}
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        storage[method_key] = {
            'all_runs_hypervolumes': [], 'all_runs_pareto_front_obj': [],
            'all_runs_NFE': [], 'all_runs_des': [], 'all_runs_obj': [],
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def save_single_run(method_name, run_idx, results, params):
    folder = f"{params['results_root']}/{params['date_str']}/run_data"
    os.makedirs(folder, exist_ok=True)
    method_key = method_name.replace(" ", "_").lower()
    filename = f"{folder}/{method_key}_run_{run_idx}.pkl"
    all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE = results
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    save_hv_and_pareto(method_name, run_idx, hypervolumes, pareto_front_obj, params)
    return filename


def _model_key(method_name: str) -> str:
    """Truss-specific model filenames."""
    if method_name == "Design Synthesis PPO":
        return "actor_model.pth"
    elif method_name == "Design Repair PPO":
        return "actor_repair_model.pth"
    return ""


def _copy_best_model(method_name: str, run_idx: int, params: dict):
    fname = _model_key(method_name)
    if not fname:
        return
    folder = f"{params['results_root']}/{params['date_str']}"
    src = os.path.join(folder, fname)
    dst = os.path.join(folder, f"best_{fname}")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  [BestModel] Copied {fname} -> best_{fname} (run {run_idx})")
    else:
        print(f"  [BestModel] WARNING: {fname} not found at {src}")


def _completed_run_exists(method_name, run_idx, params):
    """A run counts as complete only if BOTH its full-results and lightweight
    HV/Pareto files are on disk. Guards against a half-written run killed by a
    wall-clock timeout."""
    method_key = method_name.replace(" ", "_").lower()
    run_file = f"results/{params['date_str']}/run_data/{method_key}_run_{run_idx}.pkl"
    hv_file  = f"results/{params['date_str']}/hv_pareto_data/{method_key}_run_{run_idx}.pkl"
    return os.path.exists(run_file) and os.path.exists(hv_file), run_file, hv_file


def _hv_from_completed_run(hv_file):
    """Read the final hypervolume from a completed run's lightweight file."""
    try:
        with open(hv_file, "rb") as f:
            data = pickle.load(f)
        hvs = data.get('hypervolumes')
        return float(hvs[-1]) if hvs is not None and len(hvs) else None
    except Exception as e:
        print(f"  [Resume] Could not read HV from {hv_file}: {e}")
        return None


def main():
    args = build_arg_parser().parse_args()
    params = params_from_args(args)
    params['results_root'] = args.results_root

    num_runs = args.num_runs

    os.makedirs(f"{params['results_root']}/{params['date_str']}/", exist_ok=True)
    params['save_path'] = f"{params['results_root']}/{params['date_str']}/"

    methods = initialize_methods()

    # Optional method subset (used by tune.slurm to run one method per task)
    if args.methods:
        wanted = {m.strip() for m in args.methods.split(',') if m.strip()}
        methods = [m for m in methods if m.name in wanted]
        print(f"Running method subset: {[m.name for m in methods]}")

    run_files = {method.name: [] for method in methods}

    # ---- Load pre-trained artifacts if configured ----
    max_values = setup_pretrained_artifacts(params)

    # ---- If a precomputed max_values path was passed (sweep/tune), load it ----
    if max_values is None and params.get('max_values_path'):
        max_values = load_max_values(params)
        if max_values is not None:
            print(f"[Pretrained] max_values ready from --max-values-path: {max_values}")

    best_hv_models = {
        "Design Synthesis PPO": -np.inf,
        "Design Repair PPO": -np.inf,
        "Transfer Synthesis PPO": -np.inf,
        "Transfer Repair PPO": -np.inf,
    }

    # ================================================================
    # MAIN LOOP: iterate methods first, then runs
    # ================================================================
    for method in methods:
        print(f"\n{'='*60}\nRunning method: {method.name} for {num_runs} run(s)\n{'='*60}")

        all_run_max_values = []

        for run_idx in range(num_runs):
            print(f"\n--- {method.name} | Run {run_idx + 1}/{num_runs} ---")

            # ── Resume guard: skip runs already fully saved to disk ──────────
            done, run_file, hv_file = _completed_run_exists(method.name, run_idx, params)
            if done:
                print(f"[Resume] Skipping {method.name} run {run_idx} (files already exist)")
                run_files[method.name].append(run_file)   # register for post-processing

                # Keep best-HV tracking honest so a LATER run can't promote a
                # worse model over this already-completed (and better) run's best_*.
                if method.name in best_hv_models:
                    hv = _hv_from_completed_run(hv_file)
                    if hv is not None and hv > best_hv_models[method.name]:
                        best_hv_models[method.name] = hv
                        print(f"  [Resume] Seeded best HV for {method.name}: {hv:.6f} (run {run_idx})")
                continue
            # ─────────────────────────────────────────────────────────────────


            force_clear_memory()

            if method.name == "Random Search":
                result = method.function(params)
                (all_des, all_obj, pareto_front_des,
                 pareto_front_obj, hypervolumes, NFE, run_max_values) = result
                all_run_max_values.append(run_max_values)
                filename = save_single_run(
                    method.name, run_idx,
                    (all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE),
                    params)
                run_files[method.name].append(filename)
                print(f"Results saved to {filename}")
                force_clear_memory()

            elif method.requires_max_values and max_values is not None:
                if method.name in best_hv_models:
                    result = method.function(max_values, params,
                                             run_idx=run_idx, best_hv_so_far=best_hv_models[method.name])
                else:
                    result = method.function(max_values, params)

                if result is None:
                    print(f"Warning: {method.name} run {run_idx} returned None")
                    continue

                if method.name in best_hv_models:
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE, run_best_hv) = result
                    if run_best_hv is not None and run_best_hv > best_hv_models[method.name]:
                        best_hv_models[method.name] = run_best_hv
                        print(f"  New best HV for {method.name}: {run_best_hv:.6f} (run {run_idx})")
                        _copy_best_model(method.name, run_idx, params)
                else:
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE) = result

                filename = save_single_run(
                    method.name, run_idx,
                    (all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE),
                    params)
                run_files[method.name].append(filename)
                print(f"Results saved to {filename}")
                force_clear_memory()

            else:
                if method.requires_max_values and max_values is None:
                    print(f"Skipping {method.name}: max_values not yet available.")
                    continue

            print(f"Memory cleared after {method.name} run {run_idx}")

        # ── Post-all-runs processing ──
        if method.name == "Random Search":
            combined_max = np.max(np.stack(all_run_max_values, axis=0), axis=0)
            max_values = combined_max
            save_max_values(max_values, params)
            print(f"\nCombined max_values across {num_runs} run(s): {max_values}")

        if method.name in best_hv_models and best_hv_models[method.name] > -np.inf:
            print(f"\nBest HV for {method.name} across all runs: {best_hv_models[method.name]:.6f}")
            print(f"Best model saved to: {params['results_root']}/{params['date_str']}/best_{_model_key(method.name)}")

    # ================================================================
    # POST-PROCESSING
    # ================================================================
    print(f"\n{'='*60}\nAll runs complete. Loading HV/Pareto data for final analysis...\n{'='*60}")
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

    for method in methods:
        try:
            visualize_configurations(storage, [method], params, max_designs=5)
        except Exception as e:
            print(f"Error visualizing {method.name}: {e}")
        force_clear_memory()

    del storage
    force_clear_memory()

    root = params['results_root']
    print(f"\nResults saved to: {root}/{params['date_str']}/")
    print("Generated files:")
    print("  - run_data/*.pkl          : Raw results per method per run")
    print("  - hv_pareto_data/*.pkl    : Lightweight HV/Pareto per method per run")
    print("  - max_values.npy          : Normalization constants from Random Search")
    print("  - all_data.pkl            : Complete aggregated results")
    print("  - hypervolume_comparison.html")
    print("  - pareto_front_3d.html")
    print("  - <method>_design_<i>.html   : Truss configuration renders")


if __name__ == "__main__":
    main()