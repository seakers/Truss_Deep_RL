import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from evaluation.truss_problem import TrussProblem
from utils.pareto import postprocess_run

def run_random_search(params):

    num_exec = params['num_epochs'] * params['mini_batch_size']

    all_des = []
    all_obj = []
    all_constraints = []
    all_constraint_vals = []
    NFE = 0
    setup_eval_function = TrussProblem(overlap_constraint=params['overlap_constraint'])
    design_space = setup_eval_function.design_space
    num_actions = setup_eval_function.n_elements
    action_range = len(setup_eval_function.area_map)
    num_objectives = setup_eval_function.num_objectives
    objective_min_max = setup_eval_function.objective_min_max
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])
    
    print(f"Starting Random Search for {num_exec} evaluations...")

    for run in range(num_exec):
        if run % (params['mini_batch_size'] * 10) == 0:
            print(f"Execution {run+1}/{num_exec}")
            
        design = [int(np.random.choice(v['range'])) for v in design_space]
        
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

    post_process_package = postprocess_run(
        all_des, all_obj, all_constraints, all_constraint_vals,
        objective_min_max, max_values,
    )

    return (*post_process_package, max_values)