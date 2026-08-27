import numpy as np
import pygad
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.indicators.hv import HV
import torch

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import is_pareto_efficient
from optimization.transformer_rand_weights_state import Actor, Critic


def run_warm_start_ga(max_values, params):
    # print("\n\nRunning Genetic Algorithm...\n\n")
    
    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']
    
    setup_problem = TrussProblem()

    num_genes = setup_problem.n_elements
    gene_space = [ range(len(setup_problem.area_map)) for _ in range(num_genes) ]
    
    min_mask = np.array([True, True, False, True, True])

    init_designs = sample_designs(params, len(min_mask), num_genes)
    
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

    # Initialize GA
    ga_instance = pygad.GA(
        num_generations=n_gen,
        num_parents_mating=int(pop_size/2),
        fitness_func=fitness_func,
        sol_per_pop=pop_size,
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type="nsga2",
        mutation_type="random",
        mutation_probability=0.1,
        initial_population=init_designs,
        keep_elitism=0,
        keep_parents=0,
        save_solutions=False,
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

    pareto_front_des = []
    pareto_front_obj = []
    hypervolumes = []

    for obj in range(NFE):
        if all_constraints[obj] == False or len(hypervolumes) == 0: # check this to save computational time
            pfront_obj = hv_obj[:obj+1]
            pfront_des = all_des[:obj+1]
            pfront_mask = is_pareto_efficient(pfront_obj, return_mask=True)
            if pfront_mask[-1]:
                pareto_front_obj.append(pfront_obj[pfront_mask])
                pareto_front_des.append(pfront_des[pfront_mask])
                hypervolumes.append(hv_ind(pfront_obj[pfront_mask]))
            else:
                hypervolumes.append(hypervolumes[-1])
                pareto_front_obj.append(pareto_front_obj[-1])
                pareto_front_des.append(pareto_front_des[-1])
        else:
            hypervolumes.append(hypervolumes[-1])
            pareto_front_obj.append(pareto_front_obj[-1])
            pareto_front_des.append(pareto_front_des[-1])

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE


def get_models(device, params, num_total_objs):
    # Briefly instantiate problem to get input dimensions
    problem = TrussProblem()
    num_actions = problem.n_elements
    action_length = len(problem.area_map)

    # Actor input: action_length (sequence length)
    actor = Actor(device=device, params=params, action_length=action_length)

    actor.to(device)
    
    # Initialize lazy layers
    input_dummy = torch.zeros((1, num_actions+num_total_objs), dtype=torch.float32).to(device)
    actor(input_dummy)

    actor.load_state_dict(torch.load(f"results/{params['date_str']}/actor_model.pth"))

    return actor


def sample_designs(params, num_total_objs, num_actions):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = get_models(device, params, num_total_objs)
    mini_batch_size = params['mini_batch_size']
    # Generate Weights for all 5 dimensions
    weights_nonnorm = np.random.rand(mini_batch_size, num_total_objs)
    weights_nonnorm[:, -2:] = 1.0  # Equal weights for constraint values
    weights = weights_nonnorm / weights_nonnorm.sum(axis=1, keepdims=True)

    designs = [[] for _ in range(mini_batch_size)]
    observation = [[] for _ in range(mini_batch_size)]

    for idx in range(mini_batch_size):
        observation[idx] = weights[idx].tolist()  # Initialize observation with weights

    # --- 1. Sample Actor ---
    with torch.no_grad():
        for i in range(num_actions):
            log_probs_out, sel_actions_out = actor.sample_action(observation)
            log_probs_out = log_probs_out.tolist()
            sel_actions_out = sel_actions_out.tolist()

            for idx, action in enumerate(sel_actions_out):
                observation[idx].append(action)
                des_val = int(action)
                designs[idx].append(des_val)

    return designs
