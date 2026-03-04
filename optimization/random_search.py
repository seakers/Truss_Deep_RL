import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.indicators.hv import HV
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import is_pareto_efficient

def run_random_search(params):

    num_exec = params['num_epochs'] * params['mini_batch_size']

    all_des = []
    all_obj = []
    all_constraints = []
    all_constraint_vals = []
    NFE = 0
    setup_eval_function = TrussProblem()
    num_actions = setup_eval_function.n_elements
    action_range = len(setup_eval_function.area_map)
    num_objectives = setup_eval_function.num_objectives
    objective_min_max = setup_eval_function.objective_min_max
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])
    ref_point = np.ones(num_objectives)
    # hv_grid = HypervolumeGrid(refPoint=ref_point)
    hv_ind = HV(ref_point=ref_point)
    
    print(f"Starting Random Search for {num_exec} evaluations...")

    for run in range(num_exec):
        if run % (params['mini_batch_size'] * 10) == 0:
            print(f"Execution {run+1}/{num_exec}")
            
        design = np.random.randint(0, action_range, num_actions).tolist()
        
        # 1. Evaluate & Capture Constraint Values
        objectives, is_constrained, constraint_vals = setup_eval_function.evaluate(design)

        all_des.append(design)
        all_obj.append(objectives)
        all_constraints.append(is_constrained)
        all_constraint_vals.append(constraint_vals)
        NFE += 1

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_constraint_vals = np.array(all_constraint_vals)
    max_values = np.max(np.concatenate((all_obj, all_constraint_vals), axis=1), axis=0) * 1.1 + 1e-6  # Add a small epsilon to avoid division by zero
    max_objectives = max_values[:num_objectives]
    ref_values = max_objectives * ref_point
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

    print(f"Max values used for normalization: {max_values}")
            
    # Return 5-dim all_obj so max_values can be reused by PPO if needed
    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE, max_values