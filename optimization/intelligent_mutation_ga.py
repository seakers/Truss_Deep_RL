import numpy as np
import pygad
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.indicators.hv import HV
import torch

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress
from optimization.transformer_design_repair import Actor, Critic

def run_intelligent_mutation_ga(max_values, params):
    # print("\n\nRunning Genetic Algorithm...\n\n")
    
    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']
    
    setup_problem = TrussProblem()

    num_genes = setup_problem.n_elements
    gene_space = [ range(len(setup_problem.area_map)) for _ in range(num_genes) ]
    
    min_mask = np.array([True, True, False, True, True])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_total_objs = len(min_mask)
    actor = get_models(device, params, num_total_objs)
    
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


    use_stress = params.get('use_stress', False)
    max_edits = params.get('max_edits', 200)

    def intelligent_mutation(offspring, ga_instance):
        nonlocal NFE

        for ind, design in enumerate(offspring):
            stop_token = False
            new_design = design.copy()
            n_edits = 0

            # pygad mutates before evaluating, so the utilisation channel needs its own
            # solve. Counted as an NFE, since it costs the same as one.
            if use_stress:
                _, _, _, util = setup_problem.evaluate(
                    [int(x) for x in design], return_utilisation=True)
                NFE += 1

            while not stop_token and n_edits < max_edits:
                n_edits += 1

                if use_stress:
                    state = np.stack([np.asarray(design, dtype=float),
                                      np.asarray(util, dtype=float)], axis=-1)
                else:
                    state = np.asarray(design, dtype=float)

                # Sample actions from actor
                _, var_idx, _, value_idx, _, stop_decision = actor.sample_action(
                    torch.tensor(state, dtype=torch.float32).to(device)
                )

                # Create new design
                new_design = design.copy()
                new_design[var_idx.item()] = value_idx.item()
                design = new_design.copy()

                # Check if stop decision was made
                if stop_decision.item() == 1:  # Stop decision is 1 (stop)
                    stop_token = True
                elif use_stress:
                    # the design moved, so the utilisation the operator is reading must
                    # move with it, otherwise later edits are made on stale physics
                    _, _, _, util = setup_problem.evaluate(
                        [int(x) for x in design], return_utilisation=True)
                    NFE += 1

            offspring[ind] = new_design

        return offspring

    # Stop on evaluation count, not generation count: elitism carries individuals over
    # unevaluated and the stress-aware operator spends extra solves per offspring.
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
        # Same knobs as genetic_algorithm.py, but defaults stay at the historical values
        # so earlier comparisons still reproduce. Pass the tuned ones explicitly.
        parent_selection_type=params.get('parent_selection_type', 'sss'),
        K_tournament=params.get('K_tournament', 3),
        crossover_type="single_point",
        mutation_type=intelligent_mutation,
        keep_elitism=params.get('keep_elitism', 0),
        keep_parents=0,
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


def get_models(device, params, num_total_objs):
    # Briefly instantiate problem to get input dimensions
    problem = TrussProblem()
    num_actions = problem.n_elements
    action_length = len(problem.area_map)

    # Actor input: action_length (sequence length)
    actor = Actor(device=device, params=params, num_variables=num_actions, action_length=action_length, num_objectives=num_total_objs)

    actor.to(device)
    
    # Initialize lazy layers. Shape must match what the operator is fed at deployment:
    # [1, 72, 2] with the utilisation channel, [1, 77] without.
    if params.get('use_stress', False):
        input_dummy = torch.zeros((1, num_actions, 2), dtype=torch.float32).to(device)
    else:
        input_dummy = torch.zeros((1, num_actions+num_total_objs), dtype=torch.float32).to(device)
    actor(input_dummy)

    if not params.get('random_init_actor', False):
        actor.load_state_dict(torch.load(f"results/{params['date_str']}/actor_repair_model.pth"))

    return actor