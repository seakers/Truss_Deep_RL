import warnings
# import openpyxl
# warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import pickle

# Keep these imports; we will adapt the files they point to later
from optimization.random_search import run_random_search
from optimization.genetic_algorithm import run_genetic_algorithm
# from optimization.ppo_rand_weights_uninformed import run_ppo_rand_weights_uninformed
from optimization.ppo_rand_weights_state import run_ppo_rand_weights_state
from optimization.warm_start_ga import run_warm_start_ga
from optimization.design_repair import run_design_repair_ppo
from optimization.intelligent_mutation_ga import run_intelligent_mutation_ga

# Updated imports for Truss context
# (We will define these in the future 'utils/problem.py' and 'utils/visualization.py')
from evaluation.truss_problem import TrussProblem 
# from utils.visualization import visualize_truss_structure

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
        # OptimizationMethod("PPO Random Weights Uninformed", run_ppo_rand_weights_uninformed, "green", True),
        # OptimizationMethod("PPO Random Weights State", run_ppo_rand_weights_state, "red", True),
        # OptimizationMethod("Warm Start GA", run_warm_start_ga, "green", True),
        OptimizationMethod("Design Repair PPO", run_design_repair_ppo, "purple", True),
        OptimizationMethod("Intelligent Mutation GA", run_intelligent_mutation_ga, "cyan", True)
    ]
    return methods


def run_single_method(method, params, max_values=None):
    """Run a single optimization method"""
    if method.requires_max_values and max_values is not None:
        return method.function(max_values, params), None
    elif not method.requires_max_values:
        result = method.function(params)
        if method.name == "Random Search":
            # Random search returns max_values as the last element for normalization in other methods
            return result[:-1], result[-1]  # Return (method_results, max_values)
        return result, None
    else:
        raise ValueError(f"Method {method.name} requires max_values but none provided")


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
    """Truncate the methods which can have different lengths for various reasons"""
    min_nfe = None
    for method in methods:
        if method.name == "Genetic Algorithm" or method.name == "Warm Start GA" or method.name == "Design Repair PPO" or method.name == "Intelligent Mutation GA": # these three can have variable nfe
            key = method.name.replace(" ", "_").lower()
            if key not in storage or not storage[key]['all_runs_NFE']:
                continue
                
            nfe = storage[key]['all_runs_NFE']
            min_nfe = min(nfe)

            # Truncate lists to the minimum length found
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
            
            # Plot lines
            plt.plot(stats['max'], label=f'{method.name} Max', color=method.color, linestyle='--', alpha=0.8)
            plt.plot(stats['min'], label=f'{method.name} Min', color=method.color, linestyle=':', alpha=0.8)
            plt.plot(stats['median'], label=f'{method.name} Median', color=method.color, linewidth=2)
            
            # Fill IQR
            plt.fill_between(range(len(stats['q25'])), stats['q25'], stats['q75'], 
                           label=f'{method.name} IQR', color=method.color, alpha=0.1)
    
    plt.title('Hypervolume Comparison Across All Methods')
    plt.xlabel('Iterations')
    plt.ylabel('Hypervolume')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{params['date_str']}/hypervolume_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_configurations(storage, methods, params, max_designs=5):
    """Visualize configurations for final Pareto front designs from each method"""
    # Create the problem instance once to access visualization tools
    vis_problem = TrussProblem()
    
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        method_safe_name = method.name.replace(" ", "_").lower()
        
        pareto_designs = storage[method_key]['all_runs_pareto_front_des']
        
        # Check if we have valid results
        if (pareto_designs and len(pareto_designs) > 0 and 
            pareto_designs[-1] and len(pareto_designs[-1]) > 0 and 
            len(pareto_designs[-1][-1]) > 0):
            
            # Get the final Pareto front from the last run
            final_pareto_designs = pareto_designs[-1][-1]
            
            print(f"\nVisualizing {min(len(final_pareto_designs), max_designs)} designs for {method.name}...")
            
            for i, design in enumerate(final_pareto_designs):
                # Ensure design is the correct format (list of ints)
                if isinstance(design, np.ndarray):
                    design = design.tolist()
                
                # Define filename
                filename = f"results/{params['date_str']}/{method_safe_name}_design_{i}"
                
                # Visualize the truss
                vis_problem.render(design, save_name=filename)
                
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
            
        # Ensure we have at least 3 dims for 3D plot
        if arr.shape[1] >= 3:
            pts = arr[:, :3]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        label=method.name, color=method.color, s=40, alpha=0.9, edgecolors='k')
            plotted = True
        elif arr.shape[1] == 2:
            # Fallback for 2D if testing fewer objectives
             ax.scatter(pts[:, 0], pts[:, 1], np.zeros_like(pts[:,0]),
                        label=method.name, color=method.color)
             plotted = True

    if not plotted:
        plt.close(fig)
        return

    # Update labels to match Truss objectives
    ax.set_xlabel('Mass (lbs)')
    ax.set_ylabel('Compliance')
    ax.set_zlabel('Neg Frequency (Hz)') # or just Frequency if you flipped the sign in the plot
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


def main():
    # Configuration
    num_runs = 1
    
    params = {
        'num_epochs': 50,
        'mini_batch_size': 64,
        'gamma': 0.999,
        'lambda': 0.9,
        'learning_rate': 0.0003,
        'clip_ratio': 0.05,
        'update_iterations': 5,
        'target_kl': 0.003,
        'chunk_size': 256,
        'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'model_folder': None,  # Set to None to train from scratch
    }
    
    # Create results directory
    os.makedirs(f"results/{params['date_str']}/", exist_ok=True)
    params['save_path'] = f"results/{params['date_str']}/"

    # Initialize methods and storage
    methods = initialize_methods()
    storage = initialize_storage(methods)
    
    # Run optimization for each run
    for run in range(num_runs):
        print(f"\n\nStarting run {run + 1}/{num_runs}\n\n")
        max_values = None
        
        # Run each method
        for method in methods:
            print(f"\n\nRunning {method.name}...")
            
            # Pass params and max_values (for normalization if needed)
            results, method_max_values = run_single_method(method, params, max_values)
            store_results(storage, method.name, results)
            
            # Capture max values from Random Search to normalize objectives for GA/RL
            if method.name == "Random Search" and method_max_values is not None:
                max_values = method_max_values
                print(f"Max values found (for normalization): {max_values}")
    
    # Truncate data to ensure consistent plotting
    storage = truncate_data(storage, methods)
    
    # Save, Plot, Visualize
    save_all_data(storage, methods, params)
    plot_hypervolumes(storage, methods, params)
    plot_pareto_front_3d(storage, methods, params)
    visualize_configurations(storage, methods, params, max_designs=5)
    
    print(f"\nResults saved to: results/{params['date_str']}/")

if __name__ == "__main__":
    main()