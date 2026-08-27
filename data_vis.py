from main import *

methods = initialize_methods()
num_runs = 10
params = {"date_str": "2026-07-31_agg"}

run_files = {method.name: [] for method in methods}
for method in methods:
    method_key = method.name.replace(" ", "_").lower()
    for run_idx in range(num_runs):
        fpath = f"results/{params['date_str']}/run_data/{method_key}_run_{run_idx}.pkl"
        if os.path.exists(fpath):
            run_files[method.name].append(fpath)
        else:
            print(f"  [WARNING] Run file not found: {fpath}")
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

# # Option B: Full reload for save_all_data and visualization (one method at a time)
# print("Building full dataset for save_all_data...")
# storage = initialize_storage(methods)
# for method in methods:
#     for fpath in run_files[method.name]:
#         with open(fpath, "rb") as f:
#             data = pickle.load(f)
#         store_results(storage, method.name, data)
#         del data
#         force_clear_memory()

# storage = truncate_data(storage, methods)
# # save_all_data(storage, methods, params)
# plot_pareto_front_3d(storage, methods, params)

# # Visualization: one method at a time to limit memory
# for method in methods:
#     try:
#         visualize_configurations(storage, [method], params, max_designs=5)
#     except Exception as e:
#         print(f"Error visualizing {method.name}: {e}")
#     force_clear_memory()

# del storage
# force_clear_memory()

print(f"\nResults saved to: results/{params['date_str']}/")
print("Generated files:")
print("  - run_data/*.pkl          : Raw results per method per run")
print("  - hv_pareto_data/*.pkl    : Lightweight HV/Pareto per method per run")
print("  - max_values.npy          : Normalization constants from Random Search")
print("  - all_data.pkl            : Complete aggregated results")
print("  - hypervolume_comparison.png")
print("  - pareto_front_3d.png")
print("  - AOS operator statistics (if AOS GA was run)")























# # aggregate_runs.py
# #
# # Collects up to TARGET_RUNS runs per method from a list of source directories,
# # then generates all the same output files that main.py produces:
# #   - run_data/*.pkl
# #   - hv_pareto_data/*.pkl
# #   - max_values.npy
# #   - all_data.pkl
# #   - hypervolume_comparison.png
# #   - pareto_front_3d.png
# #   - per-method design visualizations

# import os
# import pickle
# import numpy as np

# # ---------------------------------------------------------------------------
# # Import everything we need from main so we don't duplicate logic
# # ---------------------------------------------------------------------------
# from main import (
#     initialize_methods,
#     initialize_storage,
#     store_results,
#     truncate_data,
#     save_single_run,
#     save_hv_and_pareto,
#     save_max_values,
#     save_all_data,
#     load_hv_and_pareto_for_plotting,
#     plot_hypervolumes,
#     plot_pareto_front_3d,
#     visualize_configurations,
#     force_clear_memory,
# )

# # ===========================================================================
# # CONFIGURATION  –  edit these two blocks as needed
# # ===========================================================================

# TARGET_RUNS = 10          # Total runs we want per method

# # Source directories searched IN ORDER.  Runs are collected from the top
# # of this list first until TARGET_RUNS is reached.
# SOURCE_DIRS = [
#     "results/2026-06-10_08-44-37",
#     "results/2026-06-10_16-09-53",
#     "results/2026-06-15_10-09-18",
#     "results/2026-06-29_08-49-13",
#     "results/2026-07-15_08-49-44",
#     "results/2026-07-27_10-59-20",
# ]

# # Output folder inside results/
# AGG_DATE_STR = "2026-07-31_agg"

# # ===========================================================================


# def discover_runs(source_dirs: list[str], method_key: str, target: int) -> list[str]:
#     """
#     Walk source_dirs in order and collect run_data pkl files for method_key
#     until we have `target` files.  Returns a list of absolute file paths.

#     Files are expected at:
#         <source_dir>/run_data/<method_key>_run_<idx>.pkl
#     """
#     collected: list[str] = []

#     for src in source_dirs:
#         run_data_dir = os.path.join(src, "run_data")
#         if not os.path.isdir(run_data_dir):
#             print(f"  [discover] No run_data folder in {src}, skipping.")
#             continue

#         # Gather all matching files and sort by run index so order is stable
#         candidates = sorted(
#             [
#                 f for f in os.listdir(run_data_dir)
#                 if f.startswith(method_key) and f.endswith(".pkl")
#             ],
#             key=lambda name: int(name.replace(method_key + "_run_", "").replace(".pkl", ""))
#             if name.replace(method_key + "_run_", "").replace(".pkl", "").isdigit()
#             else 9999,
#         )

#         for fname in candidates:
#             if len(collected) >= target:
#                 break
#             full_path = os.path.join(run_data_dir, fname)
#             collected.append(full_path)
#             print(f"  [discover] Found: {full_path}")

#         if len(collected) >= target:
#             break

#     return collected


# def load_run_file(filepath: str):
#     """Load a single run pkl and return its tuple of results."""
#     with open(filepath, "rb") as f:
#         return pickle.load(f)


# def find_last_max_values(source_dirs: list[str]) -> np.ndarray | None:
#     """
#     Load max_values.npy from the LAST source directory that contains one.
#     Returns None if none is found in any directory.
#     """
#     last_found = None
#     for src in source_dirs:
#         candidate = os.path.join(src, "max_values.npy")
#         if os.path.exists(candidate):
#             last_found = candidate   # keep overwriting so we end on the last

#     if last_found is not None:
#         max_values = np.load(last_found)
#         print(f"[max_values] Loaded from {last_found}: {max_values}")
#         return max_values

#     print("[max_values] WARNING: max_values.npy not found in any source directory.")
#     return None


# # ===========================================================================
# # MAIN AGGREGATION ROUTINE
# # ===========================================================================

# def aggregate():
#     # ------------------------------------------------------------------
#     # 1.  Setup params and output directory
#     # ------------------------------------------------------------------
#     params = {"date_str": AGG_DATE_STR}
#     out_root = f"results/{AGG_DATE_STR}"
#     os.makedirs(out_root, exist_ok=True)
#     print(f"\nOutput directory: {out_root}")

#     # ------------------------------------------------------------------
#     # 2.  Load methods the same way main does
#     # ------------------------------------------------------------------
#     methods = initialize_methods()
#     storage = initialize_storage(methods)

#     # ------------------------------------------------------------------
#     # 3.  Load max_values from the last source dir that has one
#     # ------------------------------------------------------------------
#     max_values = find_last_max_values(SOURCE_DIRS)
#     if max_values is not None:
#         save_max_values(max_values, params)

#     # ------------------------------------------------------------------
#     # 4.  Collect runs for every method
#     # ------------------------------------------------------------------
#     for method in methods:
#         method_key = method.name.replace(" ", "_").lower()
#         print(f"\n{'='*60}")
#         print(f"Collecting runs for: {method.name}")
#         print(f"{'='*60}")

#         run_files = discover_runs(SOURCE_DIRS, method_key, TARGET_RUNS)

#         if not run_files:
#             print(f"  No run files found for {method.name}, skipping.")
#             continue

#         if len(run_files) < TARGET_RUNS:
#             print(
#                 f"  WARNING: Only found {len(run_files)}/{TARGET_RUNS} runs "
#                 f"for {method.name}."
#             )

#         # ------------------------------------------------------------------
#         # 4a.  Load each run, write it into the aggregated run_data folder
#         #      and hv_pareto_data folder, then store in memory-storage.
#         # ------------------------------------------------------------------
#         for new_run_idx, src_path in enumerate(run_files):
#             print(f"\n  --- {method.name} | Aggregated run {new_run_idx} (from {src_path}) ---")

#             try:
#                 results = load_run_file(src_path)
#             except Exception as e:
#                 print(f"  ERROR loading {src_path}: {e}")
#                 continue

#             # results tuple is:
#             #   (all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE)
#             (all_des, all_obj, pareto_front_des,
#              pareto_front_obj, hypervolumes, NFE) = results

#             # Write run_data pkl with the new sequential index
#             save_single_run(
#                 method.name,
#                 new_run_idx,
#                 (all_des, all_obj, pareto_front_des,
#                  pareto_front_obj, hypervolumes, NFE),
#                 params,
#             )

#             # Write hv_pareto_data pkl
#             save_hv_and_pareto(
#                 method.name,
#                 new_run_idx,
#                 hypervolumes,
#                 pareto_front_obj,
#                 params,
#             )

#             # Accumulate into in-memory storage
#             store_results(
#                 storage,
#                 method.name,
#                 (all_des, all_obj, pareto_front_des,
#                  pareto_front_obj, hypervolumes, NFE),
#             )

#             force_clear_memory()

#     # ------------------------------------------------------------------
#     # 5.  Truncate runs that may have different lengths (same as main) [1]
#     # ------------------------------------------------------------------
#     print("\nTruncating variable-length runs...")
#     storage = truncate_data(storage, methods)

#     # ------------------------------------------------------------------
#     # 6.  Save aggregated all_data.pkl
#     # ------------------------------------------------------------------
#     print("\nSaving all_data.pkl...")
#     save_all_data(storage, methods, params)

#     # ------------------------------------------------------------------
#     # 7.  Plot hypervolume comparison [1]
#     # ------------------------------------------------------------------
#     print("\nPlotting hypervolumes...")
#     # Reload from lightweight hv_pareto files to keep memory manageable,
#     # exactly as main does for the final plotting step.
#     num_runs_per_method = {
#         method.name.replace(" ", "_").lower(): len(
#             discover_runs(SOURCE_DIRS, method.name.replace(" ", "_").lower(), TARGET_RUNS)
#         )
#         for method in methods
#     }
#     # Use the max collected count as num_runs for the loader
#     max_collected = max(num_runs_per_method.values()) if num_runs_per_method else TARGET_RUNS

#     hv_storage = load_hv_and_pareto_for_plotting(methods, max_collected, params)
#     plot_hypervolumes(hv_storage, methods, params)
#     force_clear_memory()

#     # ------------------------------------------------------------------
#     # 8.  Plot 3-D Pareto front [1]
#     # ------------------------------------------------------------------
#     print("\nPlotting 3D Pareto front...")
#     plot_pareto_front_3d(storage, methods, params)
#     force_clear_memory()

#     # ------------------------------------------------------------------
#     # 9.  Visualize design configurations (one method at a time) [1]
#     # ------------------------------------------------------------------
#     print("\nVisualizing design configurations...")
#     for method in methods:
#         try:
#             visualize_configurations(storage, [method], params, max_designs=5)
#         except Exception as e:
#             print(f"Error visualizing {method.name}: {e}")
#         force_clear_memory()

#     # ------------------------------------------------------------------
#     # 10.  Summary
#     # ------------------------------------------------------------------
#     print(f"\n{'='*60}")
#     print(f"Aggregation complete!  Results saved to: {out_root}/")
#     print("Generated files:")
#     print("  - run_data/*.pkl          : Raw results per method per run")
#     print("  - hv_pareto_data/*.pkl    : Lightweight HV/Pareto per method per run")
#     print("  - max_values.npy          : Normalization constants (from last source dir)")
#     # print("  - all_data.pkl            : Complete aggregated results")
#     print("  - hypervolume_comparison.png")
#     print("  - pareto_front_3d.png")
#     print("  - <method>_design_<i>.png : Pareto design visualizations")
#     print(f"{'='*60}")


# if __name__ == "__main__":
#     aggregate()











# from main import *

# # ============================================================
# # CONFIGURATION
# # ============================================================

# # Folder containing 10 runs for ALL methods (including non-GA-hybrid)
# PRIMARY_FOLDER = "results/2026-04-30_16-42-01"
# PRIMARY_NUM_RUNS = 10

# # GA hybrid methods split across multiple folders
# # Each entry: (folder_path, num_runs_in_that_folder)
# GA_HYBRID_FOLDERS = [
#     ("results/2026-05-08_11-17-33", 1),
#     ("results/2026-05-09_08-39-25", 8),
#     ("results/2026-05-13_08-50-17", 1),
# ]

# AOS_FOLDER = [
#     ("results/2026-05-15_09-12-33", 10),
# ]

# # These are the GA hybrid method names as they appear in your codebase
# GA_HYBRID_METHODS = {
#     "Warm Start GA",
#     "Intelligent Mutation GA",
#     "Warm Start Intelligent GA",
# }

# AOS_METHODS = {
#     "AOS GA",
# }

# OUTPUT_FOLDER = "results/combined_visualization"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# # ============================================================
# # HELPER: Load HV/Pareto from a specific folder + run indices
# # ============================================================

# def load_hv_pareto_from_folder(methods, folder, num_runs):
#     """
#     Load HV and Pareto data from a specific results folder.
#     Returns a storage dict in the same format as load_hv_and_pareto_for_plotting.
#     """
#     hv_pareto_folder = os.path.join(folder, "hv_pareto_data")
#     storage = {}

#     for method in methods:
#         method_key = method.name.replace(" ", "_").lower()
#         storage[method_key] = {
#             'all_runs_hypervolumes': [],
#             'all_runs_pareto_front_obj': [],
#             'all_runs_NFE': [],
#             # These are not in lightweight HV/Pareto files, left empty
#             'all_runs_des': [],
#             'all_runs_obj': [],
#             'all_runs_pareto_front_des': [],
#         }

#         for run_idx in range(num_runs):
#             filepath = os.path.join(hv_pareto_folder, f"{method_key}_run_{run_idx}.pkl")
#             if os.path.exists(filepath):
#                 with open(filepath, "rb") as f:
#                     data = pickle.load(f)
#                 storage[method_key]['all_runs_hypervolumes'].append(data['hypervolumes'])
#                 storage[method_key]['all_runs_pareto_front_obj'].append(data['pareto_front_obj'])
#                 storage[method_key]['all_runs_NFE'].append(len(data['hypervolumes']))
#             else:
#                 print(f"  [WARNING] Not found: {filepath}")

#     return storage


# # ============================================================
# # STEP 1: Load primary folder (all methods, 10 runs)
# # ============================================================

# print("Loading primary folder...")
# all_methods = initialize_methods()
# primary_params = {'date_str': PRIMARY_FOLDER.replace("results/", "")}

# # Use the existing utility directly
# storage_combined = load_hv_and_pareto_for_plotting(all_methods, PRIMARY_NUM_RUNS, primary_params)

# # Remove GA hybrid data from the primary folder load —
# # we'll replace it with the aggregated version from the split folders
# for method in all_methods:
#     if method.name in GA_HYBRID_METHODS or method.name in AOS_METHODS:
#         key = method.name.replace(" ", "_").lower()
#         storage_combined[key] = {
#             'all_runs_hypervolumes': [],
#             'all_runs_pareto_front_obj': [],
#             'all_runs_NFE': [],
#             'all_runs_des': [],
#             'all_runs_obj': [],
#             'all_runs_pareto_front_des': [],
#         }

# print("Primary folder loaded.")


# # ============================================================
# # STEP 2: Load and aggregate GA hybrid runs from split folders
# # ============================================================

# print("Loading GA hybrid split folders...")

# # Only load the GA hybrid methods from the split folders
# ga_hybrid_method_objs = [m for m in all_methods if m.name in GA_HYBRID_METHODS]
# aos_method_objs = [m for m in all_methods if m.name in AOS_METHODS]

# for folder, num_runs in GA_HYBRID_FOLDERS:
#     print(f"  Loading {num_runs} run(s) from {folder}...")
#     folder_storage = load_hv_pareto_from_folder(ga_hybrid_method_objs, folder, num_runs)

#     for method in ga_hybrid_method_objs:
#         key = method.name.replace(" ", "_").lower()
#         storage_combined[key]['all_runs_hypervolumes'].extend(
#             folder_storage[key]['all_runs_hypervolumes']
#         )
#         storage_combined[key]['all_runs_pareto_front_obj'].extend(
#             folder_storage[key]['all_runs_pareto_front_obj']
#         )
#         storage_combined[key]['all_runs_NFE'].extend(
#             folder_storage[key]['all_runs_NFE']
#         )

# for folder, num_runs in AOS_FOLDER:
#     print(f"  Loading {num_runs} run(s) from {folder}...")
#     folder_storage = load_hv_pareto_from_folder(aos_method_objs, folder, num_runs)

#     for method in aos_method_objs:
#         key = method.name.replace(" ", "_").lower()
#         storage_combined[key]['all_runs_hypervolumes'].extend(
#             folder_storage[key]['all_runs_hypervolumes']
#         )
#         storage_combined[key]['all_runs_pareto_front_obj'].extend(
#             folder_storage[key]['all_runs_pareto_front_obj']
#         )
#         storage_combined[key]['all_runs_NFE'].extend(
#             folder_storage[key]['all_runs_NFE']
#         )

# # Verify run counts
# print("\nRun counts after aggregation:")
# for method in all_methods:
#     key = method.name.replace(" ", "_").lower()
#     n = len(storage_combined[key]['all_runs_hypervolumes'])
#     print(f"  {method.name}: {n} runs")


# # ============================================================
# # STEP 3: Truncate (handles variable-length GA runs)
# # ============================================================

# storage_combined = truncate_data(storage_combined, all_methods)


# # ============================================================
# # STEP 4: Plot — reuse your post-visualization function
# # ============================================================

# def plot_hypervolumes_combined(storage, methods, output_folder):
#     plt.figure(figsize=(14, 8))

#     for method in methods:
#         method_key = method.name.replace(" ", "_").lower()
#         if method_key not in storage or not storage[method_key]['all_runs_hypervolumes']:
#             print(f"  [SKIP] No data for {method.name}")
#             continue

#         if 'transfer_learning' in method_key:
#             continue

#         runs = storage[method_key]['all_runs_hypervolumes']

#         # Defensive truncation — handles any ragged runs regardless of truncate_data
#         min_len = min(len(r) for r in runs)
#         runs_truncated = [r[:min_len] for r in runs]

#         hypervolumes = np.array(runs_truncated)

#         if hypervolumes.size > 0:
#             stats = calculate_statistics(hypervolumes)

#             label = method.name
#             if method.name == 'PPO Random Weights State':
#                 label = 'Design Synthesis PPO'
#             if method.name == 'Warm Start Intelligent GA':
#                 label = 'Warm Start Intelligent Mutation GA'

#             plt.plot(stats['median'], label=label, color=method.color, linewidth=2)
#             plt.fill_between(
#                 range(len(stats['q25'])),
#                 stats['q25'], stats['q75'],
#                 color=method.color, alpha=0.1
#             )

#     plt.title('Hypervolume Comparison Across All Methods', fontsize='xx-large')
#     plt.xlabel('Iterations', fontsize='xx-large')
#     plt.ylabel('Hypervolume', fontsize='xx-large')
#     plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='xx-large')
#     plt.tick_params(axis='both', labelsize='xx-large')
#     plt.grid(True)
#     plt.tight_layout()

#     out_path = os.path.join(output_folder, "hypervolume_comparison_combined.png")
#     plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"\nPlot saved to: {out_path}")


# plot_hypervolumes_combined(storage_combined, all_methods, OUTPUT_FOLDER)
# print("Done!")