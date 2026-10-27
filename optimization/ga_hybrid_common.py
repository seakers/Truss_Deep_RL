"""Shared custom NSGA-II core for the hybrid (RL-informed) GA methods on the
TrussProblem. No pygad/pymoo. Reuses the operators from genetic_algorithm.py
and exposes run_hybrid_ga(...) with pluggable init-population and mutation hooks.

Truss interface is static: evaluate(design) takes one arg [10]; there is no
reset()/filter_des_space. Output format matches the RL methods via
postprocess_run [2][3].
"""
import numpy as np

from optimization.genetic_algorithm import (
    _fast_nondominated_sort, _crowding_distance, _crossover, _random_gene)
from utils.pareto import postprocess_run


def run_hybrid_ga(max_values, params, eval_func, init_population_fn=None,
                  mutation_fn=None):
    """Custom NSGA-II with optional warm-start (init_population_fn) and
    intelligent-mutation (mutation_fn) hooks. Inference-only on the actors.

    init_population_fn(static_ranges, pop_size, rng) -> list of gene vectors
    mutation_fn(child) -> mutated gene vector (list)
    """
    rng = np.random.default_rng()

    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']

    static_ranges = eval_func._design_space_template
    num_objectives = eval_func.num_objectives
    objective_min_max = eval_func.objective_min_max
    min_mask = np.array([mm.lower() == 'min' for mm in objective_min_max[:num_objectives]])

    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    NFE = 0

    def eval_and_store(raw):
        nonlocal NFE
        design = [int(x) for x in raw]
        obj_values, is_constrained, cvals = eval_func.evaluate(design)
        NFE += 1
        all_des.append(design)
        all_obj.append(obj_values)                   # raw -> HV/Pareto
        all_constraints.append(is_constrained)
        all_constraint_vals.append(cvals)
        obj = np.array(obj_values, dtype=float)
        viol = float(np.sum(cvals)) if is_constrained else 0.0
        return design, obj, viol

    # --- init population (warm-start hook or random) ---
    if init_population_fn is not None:
        raw_pop = init_population_fn(static_ranges, pop_size, rng)
    else:
        raw_pop = [[_random_gene(v, rng) for v in static_ranges] for _ in range(pop_size)]

    population, pop_obj, pop_viol = [], [], []
    for raw in raw_pop:
        d, obj, viol = eval_and_store(raw)
        population.append(d); pop_obj.append(obj); pop_viol.append(viol)
    pop_obj, pop_viol = np.array(pop_obj), np.array(pop_viol)

    # --- generations ---
    for gen in range(n_gen):
        if gen % max(1, n_gen // 10) == 0:
            print(f"GA generation {gen}/{n_gen}")

        fronts = _fast_nondominated_sort(pop_obj, min_mask, pop_viol)
        rank = np.zeros(len(population), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r

        def tournament():
            i, j = rng.integers(0, len(population), size=2)
            return i if rank[i] <= rank[j] else j

        offspring = []
        for _ in range(pop_size):
            p1, p2 = population[tournament()], population[tournament()]
            child = _crossover(np.array(p1, dtype=float), np.array(p2, dtype=float), rng)
            child = [int(x) for x in child]
            if mutation_fn is not None:                 # intelligent mutation hook
                child = mutation_fn(child)
            else:
                for i, var in enumerate(static_ranges):
                    if rng.random() < 0.1:
                        child[i] = _random_gene(var, rng)
            offspring.append(child)

        off_designs, off_obj, off_viol = [], [], []
        for raw in offspring:
            d, obj, viol = eval_and_store(raw)
            off_designs.append(d); off_obj.append(obj); off_viol.append(viol)

        combined = population + off_designs
        combined_obj = np.vstack([pop_obj, np.array(off_obj)])
        combined_viol = np.concatenate([pop_viol, np.array(off_viol)])
        c_fronts = _fast_nondominated_sort(combined_obj, min_mask, combined_viol)

        new_pop, new_obj, new_viol = [], [], []
        for front in c_fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(combined[idx])
                    new_obj.append(combined_obj[idx])
                    new_viol.append(combined_viol[idx])
            else:
                cd = _crowding_distance(combined_obj[front])
                for k in np.argsort(-cd)[: pop_size - len(new_pop)]:
                    new_pop.append(combined[front[k]])
                    new_obj.append(combined_obj[front[k]])
                    new_viol.append(combined_viol[front[k]])
                break
        population, pop_obj, pop_viol = new_pop, np.array(new_obj), np.array(new_viol)

    print(f"Total NFE: {NFE}")
    return postprocess_run(
        all_des, all_obj, all_constraints, all_constraint_vals,
        objective_min_max, max_values,
    )


def intelligent_mutation_operator(design, repair_actor, device, params, eval_func):
    """One repair trajectory (inference only) as the GA mutation operator.
    Follows the RepairActor usage in design_repair/intelligent_mutation_ga:
    sample (var_idx, new_value, stop) one gene at a time until the stop token
    [4][7][8]. The truss RepairActor.sample_action takes just the state tensor.
    """
    import torch
    current = list(design)
    stop = False
    with torch.no_grad():
        while not stop:
            state = torch.tensor(current, dtype=torch.float32, device=device)
            var_idx, new_value, stop_decision, _, _ = repair_actor.sample_action(state)
            new_design = list(current)
            new_design[int(var_idx)] = int(new_value)
            current = new_design
            if int(stop_decision) == 1:                 # stop token
                stop = True
    return current