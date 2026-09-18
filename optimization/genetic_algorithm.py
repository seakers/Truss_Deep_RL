import numpy as np
import pygad
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.indicators.hv import HV

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress

def run_genetic_algorithm(max_values, params):
    # print("\n\nRunning Genetic Algorithm...\n\n")
    
    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']
    
    setup_problem = TrussProblem()

    num_genes = setup_problem.n_elements
    gene_space = [ range(len(setup_problem.area_map)) for _ in range(num_genes) ]
    
    min_mask = np.array([True, True, False, True, True])
    
    all_des = []
    all_obj = []
    all_constraints = []
    all_constraint_vals = []
    NFE = 0
    num_objectives = setup_problem.num_objectives
    max_objectives = max_values[:num_objectives]
    objective_min_max = setup_problem.objective_min_max
    ref_point = np.ones(num_objectives)
    hv_ind = HV(ref_point=ref_point)

    def fitness_func(ga_instance, solution, solution_idx):
        discrete_solution = [int(x) for x in solution]
        
        # 1. Evaluate
        obj_values, is_constrained, constraint_vals = setup_problem.evaluate(discrete_solution)
        
        nonlocal all_des, all_obj, all_constraints, NFE, all_constraint_vals
        all_des.append(discrete_solution)
        all_obj.append(obj_values)
        # eval_function_temp.print_design()
        all_constraints.append(is_constrained)
        all_constraint_vals.append(constraint_vals)
        NFE += 1
        
        if is_constrained:
            obj_arr = -np.array(np.append(max_objectives,constraint_vals), dtype=float) # for minimization objectives
            obj_arr[~min_mask] = 0.0 # For maximization objectives, zero fitness
            obj_arr[-2:] = constraint_vals  # Append constraint values as is
            obj_arr[-2:][min_mask[-2:]] = -obj_arr[-2:][min_mask[-2:]]  # Negate if minimization
            return obj_arr
        else:
            # Return negative objectives for maximization (PyGAD maximizes fitness)
            obj_arr = np.array(obj_values + constraint_vals, dtype=float)
            obj_arr[min_mask] = -obj_arr[min_mask]
            return obj_arr

    # Stop on evaluation count, not generation count: elitism carries individuals over
    # unevaluated, so generations cost different numbers of evaluations across methods.
    nfe_budget = params.get('nfe_budget')

    def _on_generation(ga):
        if nfe_budget is not None and NFE >= nfe_budget:
            return "stop"

    # Initialize GA
    ga_instance = pygad.GA(
        num_generations=n_gen,
        num_parents_mating=int(pop_size/2),
        fitness_func=fitness_func,
        sol_per_pop=pop_size,
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type=params.get('parent_selection_type', 'tournament'),
        K_tournament=params.get('K_tournament', 3),
        crossover_type="single_point",
        mutation_type="random",
        # Tuned defaults; these were 10% mutation and no elitism.
        mutation_num_genes=params.get('mutation_num_genes', 1),
        keep_elitism=params.get('keep_elitism', 8),
        keep_parents=0,          # ignored by pygad whenever keep_elitism > 0
        save_solutions=False,
        on_generation=_on_generation,
    )

    ga_instance.run()

    # Post-Processing
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_constraint_vals = np.array(all_constraint_vals)

    ref_values = max_objectives * ref_point # max point for min objectives, zero for max objectives
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values
    print(f"Number of valid designs (False in all_constraints): {np.sum(all_constraints == False)}")
    norm_obj = all_obj / max_objectives
    hv_obj = deepcopy(norm_obj)
    for i in range(num_objectives):
        if objective_min_max[i].lower() == 'max':
            hv_obj[:,i] = 1 - norm_obj[:,i]

    pareto_front_des, pareto_front_obj, hypervolumes = track_pareto_progress(
        hv_obj, all_des, all_constraints, hv_ind)


    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE