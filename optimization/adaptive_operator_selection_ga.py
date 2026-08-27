# optimization/adaptive_operator_selection_ga.py

import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.indicators.hv import HV
import torch

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import is_pareto_efficient
from optimization.transformer_rand_weights_state import Actor as WarmStartActor
from optimization.transformer_design_repair import Actor as RepairActor


class AdaptiveOperatorSelector:
    """
    Sliding-window credit assignment + probability matching for AOS.
    """
    def __init__(self, operators, window_size=50, min_prob=0.1, alpha=0.8):
        self.operators = operators
        self.n_operators = len(operators)
        self.window_size = window_size
        self.min_prob = min_prob
        self.alpha = alpha
        self.probabilities = np.ones(self.n_operators) / self.n_operators
        self.credit_history = []
        self.quality = np.ones(self.n_operators)
        self.usage_count = np.zeros(self.n_operators)

    def select_operator(self):
        idx = np.random.choice(self.n_operators, p=self.probabilities)
        self.usage_count[idx] += 1
        return idx, self.operators[idx]

    def update(self, operator_idx, reward):
        self.credit_history.append((operator_idx, reward))
        if len(self.credit_history) > self.window_size:
            self.credit_history = self.credit_history[-self.window_size:]
        self.quality[operator_idx] = (
            self.alpha * self.quality[operator_idx] +
            (1 - self.alpha) * max(reward, 0)
        )
        total_quality = np.sum(self.quality)
        if total_quality > 0:
            raw_probs = self.quality / total_quality
        else:
            raw_probs = np.ones(self.n_operators) / self.n_operators
        self.probabilities = np.maximum(raw_probs, self.min_prob)
        self.probabilities /= self.probabilities.sum()


# ---- Crossover operators (integer-valued truss designs) ----

def single_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def two_point_crossover(parent1, parent2):
    n = len(parent1)
    p1, p2 = sorted(np.random.choice(range(1, n), 2, replace=False))
    child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
    return child1, child2

def uniform_crossover(parent1, parent2):
    mask = np.random.rand(len(parent1)) < 0.5
    child1 = [p1 if m else p2 for p1, p2, m in zip(parent1, parent2, mask)]
    child2 = [p2 if m else p1 for p1, p2, m in zip(parent1, parent2, mask)]
    return child1, child2

def blend_crossover(parent1, parent2, action_range):
    """Probabilistic blend — pick from either parent or nearby integer."""
    child1, child2 = [], []
    for p1, p2 in zip(parent1, parent2):
        if np.random.rand() < 0.5:
            child1.append(p1)
            child2.append(p2)
        else:
            lo, hi = min(p1, p2), max(p1, p2)
            child1.append(np.random.randint(max(0, lo - 1), min(action_range, hi + 2)))
            child2.append(np.random.randint(max(0, lo - 1), min(action_range, hi + 2)))
    return child1, child2


# ---- Mutation operators ----

def random_mutation(individual, action_range, mutation_rate=0.1):
    child = individual.copy()
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            child[i] = np.random.randint(0, action_range)
    return child

def neighbor_mutation(individual, action_range, mutation_rate=0.1):
    """Creep mutation: move to adjacent area index."""
    child = individual.copy()
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            delta = np.random.choice([-1, 1])
            child[i] = np.clip(child[i] + delta, 0, action_range - 1)
    return child

def swap_mutation(individual, action_range, mutation_rate=0.15):
    """Swap two random gene positions."""
    child = individual.copy()
    if np.random.rand() < mutation_rate and len(child) >= 2:
        i, j = np.random.choice(len(child), 2, replace=False)
        child[i], child[j] = child[j], child[i]
    return child

def gaussian_discrete_mutation(individual, action_range, mutation_rate=0.1, sigma=2.0):
    """Gaussian perturbation rounded to nearest integer."""
    child = individual.copy()
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            new_val = child[i] + int(np.round(np.random.normal(0, sigma)))
            child[i] = np.clip(new_val, 0, action_range - 1)
    return child


# ---- Selection helpers ----

def dominates(obj_a, obj_b, min_mask):
    """Returns True if obj_a dominates obj_b given min/max mask.
    For min objectives: lower is better. For max objectives: higher is better."""
    a = np.array(obj_a, dtype=float)
    b = np.array(obj_b, dtype=float)
    # Flip max objectives so we can do uniform "lower is better" comparison
    a_adj = np.where(min_mask, a, -a)
    b_adj = np.where(min_mask, b, -b)
    return np.all(a_adj <= b_adj) and np.any(a_adj < b_adj)


def compute_crowding_distance(objectives):
    """Compute crowding distance for NSGA-II selection."""
    n = len(objectives)
    if n <= 2:
        return np.full(n, np.inf)
    obj_array = np.array(objectives)
    num_obj = obj_array.shape[1]
    distances = np.zeros(n)
    for m in range(num_obj):
        sorted_indices = np.argsort(obj_array[:, m])
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        obj_range = obj_array[sorted_indices[-1], m] - obj_array[sorted_indices[0], m]
        if obj_range == 0:
            continue
        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (
                obj_array[sorted_indices[i + 1], m] - obj_array[sorted_indices[i - 1], m]
            ) / obj_range
    return distances


def fast_non_dominated_sort(objectives, min_mask):
    """Fast non-dominated sorting (NSGA-II). Returns list of fronts (lists of indices)."""
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q], min_mask):
                dominated_solutions[p].append(q)
            elif dominates(objectives[q], objectives[p], min_mask):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # Remove last empty front


def nsga2_selection(objectives, constraints, select_size, min_mask):
    """
    NSGA-II style selection: feasible preferred, then
    non-dominated sorting + crowding distance.
    """
    n = len(objectives)
    obj_array = np.array(objectives)
    con_array = np.array(constraints, dtype=bool)

    feasible_indices = [i for i in range(n) if not con_array[i]]
    infeasible_indices = [i for i in range(n) if con_array[i]]

    selected = []

    if len(feasible_indices) >= select_size:
        fronts = fast_non_dominated_sort(obj_array[feasible_indices], min_mask)
        for front in fronts:
            actual_indices = [feasible_indices[f] for f in front]
            if len(selected) + len(actual_indices) <= select_size:
                selected.extend(actual_indices)
            else:
                remaining = select_size - len(selected)
                front_objs = [objectives[i] for i in actual_indices]
                cd = compute_crowding_distance(front_objs)
                sorted_by_cd = np.argsort(-cd)
                for idx in sorted_by_cd[:remaining]:
                    selected.append(actual_indices[idx])
                break
    else:
        selected = feasible_indices.copy()
        remaining = select_size - len(selected)
        if remaining > 0 and len(infeasible_indices) > 0:
            fill = np.random.choice(
                infeasible_indices,
                size=min(remaining, len(infeasible_indices)),
                replace=False
            ).tolist()
            selected.extend(fill)

    while len(selected) < select_size:
        selected.append(np.random.randint(0, n))

    return selected[:select_size]


def tournament_selection(population, fitness_values, min_mask, tournament_size=3):
    """Tournament selection for two parents."""
    pop_size = len(population)
    selected_indices = []
    for _ in range(2):
        candidates = np.random.choice(pop_size, size=min(tournament_size, pop_size), replace=False)
        best_candidate = candidates[0]
        for c in candidates[1:]:
            if dominates(fitness_values[c], fitness_values[best_candidate], min_mask):
                best_candidate = c
        selected_indices.append(best_candidate)
    return selected_indices


# ---- Main AOS-GA runner ----

def run_aos_ga(max_values, params):
    print("\n\nRunning GA with Adaptive Operator Selection...\n\n")

    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']

    eval_function = TrussProblem()
    num_objectives = eval_function.num_objectives
    objective_min_max = eval_function.objective_min_max
    num_actions = eval_function.n_elements
    action_range = len(eval_function.area_map)
    max_objectives = max_values[:num_objectives]
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])
    min_mask_obj = min_mask[:num_objectives]

    ref_point = np.ones(num_objectives)
    hv_ind = HV(ref_point=ref_point)

    # ---- Load learned operator models ----
    repair_actor, repair_device = _load_repair_actor(params)
    ws_actor, ws_device, ws_num_actions, ws_num_objs = _load_warmstart_actor(params)

    # ---- Define operator pools ----
    crossover_operators = [
        ('single_point', single_point_crossover),
        ('two_point',    two_point_crossover),
        ('uniform',      uniform_crossover),
        ('blend',        lambda p1, p2: blend_crossover(p1, p2, action_range)),
    ]
    mutation_operators = [
        ('random',       lambda ind: random_mutation(ind, action_range)),
        ('neighbor',     lambda ind: neighbor_mutation(ind, action_range)),
        ('swap',         lambda ind: swap_mutation(ind, action_range)),
        ('gaussian',     lambda ind: gaussian_discrete_mutation(ind, action_range)),
        # Learned repair: iteratively edits the individual [3]
        ('intelligent',  lambda ind: intelligent_mutation_operator(
                             ind, repair_actor, repair_device)),
        # Learned restart: ignores individual, generates from scratch [1]
        ('restart',      lambda ind: restart_mutation_operator(
                             ind, ws_actor, ws_device,
                             ws_num_actions, ws_num_objs, action_range)),
    ]

    # ---- Initialize AOS selectors ----
    crossover_aos = AdaptiveOperatorSelector(
        operators=[name for name, _ in crossover_operators],
        window_size=50, min_prob=0.1, alpha=0.8
    )
    mutation_aos = AdaptiveOperatorSelector(
        operators=[name for name, _ in mutation_operators],
        window_size=50, min_prob=0.1, alpha=0.8
    )

    # ---- Tracking ----
    all_des = []
    all_obj = []
    all_constraints = []
    all_constraint_vals = []
    NFE = 0

    crossover_prob_history = []
    mutation_prob_history = []

    # ---- Initialize population randomly (same as GA [4]) ----
    population = []
    pop_objectives = []
    pop_constraints = []

    for _ in range(pop_size):
        design = np.random.randint(0, action_range, num_actions).tolist()
        obj_values, is_constrained, constraint_vals = eval_function.evaluate(design)
        NFE += 1

        population.append(design)
        pop_objectives.append(obj_values if not is_constrained else max_objectives.tolist())
        pop_constraints.append(is_constrained)

        all_des.append(design)
        all_obj.append(obj_values)
        all_constraints.append(is_constrained)
        all_constraint_vals.append(constraint_vals)

    # Build initial HV tracking
    # We'll compute HV improvement per-generation for credit assignment
    def compute_current_hv(des_list, obj_list, con_list):
        """Compute current HV from all valid designs seen so far."""
        valid_obj = []
        for o, c in zip(obj_list, con_list):
            if not c:
                norm = np.array(o) / max_objectives
                hv_o = norm.copy()
                for i in range(num_objectives):
                    if objective_min_max[i].lower() == 'max':
                        hv_o[i] = 1 - norm[i]
                valid_obj.append(hv_o)
        if len(valid_obj) == 0:
            return 0.0
        valid_obj = np.array(valid_obj)
        pfront_mask = is_pareto_efficient(valid_obj, return_mask=True)
        if np.any(pfront_mask):
            return hv_ind(valid_obj[pfront_mask])
        return 0.0

    prev_hv = compute_current_hv(all_des, all_obj, all_constraints)

    # ---- Generational loop ----
    for gen in range(n_gen):
        if gen % 10 == 0:
            print(f"AOS-GA Gen {gen + 1}/{n_gen} | "
                  f"CX probs: {np.round(crossover_aos.probabilities, 3)} | "
                  f"MUT probs: {np.round(mutation_aos.probabilities, 3)} | "
                  f"HV: {prev_hv:.6f}")

        crossover_prob_history.append(crossover_aos.probabilities.copy())
        mutation_prob_history.append(mutation_aos.probabilities.copy())

        offspring_population = []
        offspring_objectives = []
        offspring_constraints = []
        offspring_constraint_vals = []
        offspring_cx_ops = []
        offspring_mut_ops = []

        # ---- Generate offspring ----
        i = 0
        while i < pop_size:
            parent_indices = tournament_selection(population, pop_objectives, min_mask_obj)
            parent1 = population[parent_indices[0]].copy()
            parent2 = population[parent_indices[1]].copy()

            # Adaptive crossover selection
            cx_idx, cx_name = crossover_aos.select_operator()
            cx_func = crossover_operators[cx_idx][1]
            child1, child2 = cx_func(parent1, parent2)

            # Adaptive mutation selection for each child
            mut_idx1, _ = mutation_aos.select_operator()
            mut_func1 = mutation_operators[mut_idx1][1]
            child1 = mut_func1(child1)

            mut_idx2, _ = mutation_aos.select_operator()
            mut_func2 = mutation_operators[mut_idx2][1]
            child2 = mut_func2(child2)

            # Evaluate both children
            for child, mut_idx_used in [(child1, mut_idx1), (child2, mut_idx2)]:
                if i >= pop_size:
                    break

                # Ensure valid discrete values
                child = [np.clip(int(round(c)), 0, action_range - 1) for c in child]

                obj_values, is_constrained, constraint_vals = eval_function.evaluate(child)
                NFE += 1

                offspring_population.append(child)
                offspring_objectives.append(obj_values if not is_constrained else max_objectives.tolist())
                offspring_constraints.append(is_constrained)
                offspring_constraint_vals.append(constraint_vals)
                offspring_cx_ops.append(cx_idx)
                offspring_mut_ops.append(mut_idx_used)

                all_des.append(child)
                all_obj.append(obj_values)
                all_constraints.append(is_constrained)
                all_constraint_vals.append(constraint_vals)

                i += 1

        # ---- Credit assignment: HV improvement this generation ----
        current_hv = compute_current_hv(all_des, all_obj, all_constraints)
        gen_hv_improvement = max(0, current_hv - prev_hv)

        # Distribute credit proportionally to operators used this generation
        # Count how many offspring each operator produced
        cx_counts = np.zeros(crossover_aos.n_operators)
        mut_counts = np.zeros(mutation_aos.n_operators)
        for cx_op in offspring_cx_ops:
            cx_counts[cx_op] += 1
        for mut_op in offspring_mut_ops:
            mut_counts[mut_op] += 1

        # Per-operator credit = (gen improvement) * (fraction of offspring from that operator)
        total_offspring = len(offspring_population)
        if total_offspring > 0 and gen_hv_improvement > 0:
            for op_idx in range(crossover_aos.n_operators):
                if cx_counts[op_idx] > 0:
                    credit = gen_hv_improvement * (cx_counts[op_idx] / total_offspring)
                    crossover_aos.update(op_idx, credit)
            for op_idx in range(mutation_aos.n_operators):
                if mut_counts[op_idx] > 0:
                    credit = gen_hv_improvement * (mut_counts[op_idx] / total_offspring)
                    mutation_aos.update(op_idx, credit)
        else:
            # No improvement: give zero credit to all used operators
            for op_idx in set(offspring_cx_ops):
                crossover_aos.update(op_idx, 0.0)
            for op_idx in set(offspring_mut_ops):
                mutation_aos.update(op_idx, 0.0)

        prev_hv = current_hv

        # ---- Survival selection: NSGA-II style ----
        combined_pop = population + offspring_population
        combined_obj = pop_objectives + offspring_objectives
        combined_con = pop_constraints + offspring_constraints

        selected_indices = nsga2_selection(combined_obj, combined_con, pop_size, min_mask_obj)

        population = [combined_pop[i] for i in selected_indices]
        pop_objectives = [combined_obj[i] for i in selected_indices]
        pop_constraints = [combined_con[i] for i in selected_indices]

    # ---- Post-processing (same as GA [4]) ----
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_constraint_vals = np.array(all_constraint_vals)

    ref_values = max_objectives * ref_point
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values
    print(f"Valid designs: {np.sum(~all_constraints)}")

    from copy import deepcopy
    norm_obj = all_obj / max_objectives
    hv_obj = deepcopy(norm_obj)
    for i in range(num_objectives):
        if objective_min_max[i].lower() == 'max':
            hv_obj[:, i] = 1 - norm_obj[:, i]

    pareto_front_des, pareto_front_obj, hypervolumes = [], [], []
    for obj_idx in range(NFE):
        if not all_constraints[obj_idx] or len(hypervolumes) == 0:
            pfront_obj = hv_obj[:obj_idx + 1]
            pfront_des = all_des[:obj_idx + 1]
            pfront_mask = is_pareto_efficient(pfront_obj, return_mask=True)
            if pfront_mask[-1]:
                pareto_front_obj.append(pfront_obj[pfront_mask])
                pareto_front_des.append(pfront_des[pfront_mask])
                hypervolumes.append(hv_ind(pfront_obj[pfront_mask]))
            else:
                hypervolumes.append(hypervolumes[-1] if hypervolumes else 0)
                pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])
                pareto_front_des.append(pareto_front_des[-1] if pareto_front_des else [])
        else:
            hypervolumes.append(hypervolumes[-1] if hypervolumes else 0)
            pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])
            pareto_front_des.append(pareto_front_des[-1] if pareto_front_des else [])

    print(f"Total NFE: {NFE}")

    # ---- Plot AOS statistics ----
    _plot_aos_statistics(
        crossover_prob_history, mutation_prob_history,
        crossover_aos, mutation_aos,
        [name for name, _ in crossover_operators],
        [name for name, _ in mutation_operators],
        params
    )

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE


def intelligent_mutation_operator(individual, actor, device):
    """
    Design repair mutation: iteratively modifies the individual using
    the trained repair actor until the actor emits a stop token.
    Matches the (individual) -> individual signature used by AOS.
    """
    design = list(individual).copy()  # ensure it's a plain list

    stop_token = False
    while not stop_token:
        _, var_idx, _, value_idx, _, stop_decision = actor.sample_action(
            torch.tensor(design, dtype=torch.float32).to(device)
        )
        design[var_idx.item()] = value_idx.item()

        if stop_decision.item() == 1:
            stop_token = True

    return design


def restart_mutation_operator(individual, warmstart_actor, device,
                              num_actions, num_total_objs, action_range):
    """
    Restart mutation: ignores the input individual entirely and returns
    a brand-new design sampled from the trained from-scratch generator.
    Weights are sampled randomly, matching sample_designs() in warm_start_ga.py [1].
    """
    # Sample a single random weight vector
    # have to set mini batch to 2 for dimensionality reasons, but we'll only use the first one and ignore the second [1]
    weights_nonnorm = np.random.rand(2,num_total_objs)
    weights_nonnorm[:, -2:] = 1.0   # equal weight for constraint dims [1]
    weights = weights_nonnorm / weights_nonnorm.sum(axis=1, keepdims=True)

    design = []
    observation = [[] for _ in range(2)]

    for idx in range(2):
        observation[idx] = weights[idx].tolist()  # Initialize observation with weights

    with torch.no_grad():
        for i in range(num_actions):
            _, sel_action = warmstart_actor.sample_action(observation)
            for idx, action in enumerate(sel_action):
                observation[idx].append(action)
            design.append(int(sel_action[0]))

    # Clip just in case [2]
    design = [np.clip(v, 0, action_range - 1) for v in design]
    return design


def _load_repair_actor(params):
    """Load the trained design repair actor model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem = TrussProblem()
    num_actions = problem.n_elements
    action_length = len(problem.area_map)

    # min_mask has 5 elements (3 objectives + 2 constraints), matching intelligent_mutation_ga.py
    num_total_objs = 5

    actor = RepairActor(
        device=device,
        params=params,
        num_variables=num_actions,
        action_length=action_length,
        num_objectives=num_total_objs
    )
    actor.to(device)

    # Initialize lazy layers
    input_dummy = torch.zeros((1, num_actions + num_total_objs), dtype=torch.float32).to(device)
    actor(input_dummy)

    actor.load_state_dict(
        torch.load(f"results/{params['date_str']}/actor_repair_model.pth")
    )
    actor.eval()
    return actor, device


def _load_warmstart_actor(params):
    """Load the trained from-scratch generator actor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem = TrussProblem()
    num_actions = problem.n_elements
    action_length = len(problem.area_map)
    num_total_objs = 5  # 3 objectives + 2 constraints, matching warm_start_ga.py [1]

    actor = WarmStartActor(
        device=device,
        params=params,
        action_length=action_length
    )
    actor.to(device)

    # Initialize lazy layers
    input_dummy = torch.zeros(
        (1, num_actions + num_total_objs), dtype=torch.float32
    ).to(device)
    actor(input_dummy)

    actor.load_state_dict(
        torch.load(f"results/{params['date_str']}/actor_model.pth")
    )
    actor.eval()
    return actor, device, num_actions, num_total_objs


def _plot_aos_statistics(cx_history, mut_history, cx_aos, mut_aos,
                         cx_names, mut_names, params):
    """Plot how operator selection probabilities evolved over generations."""
    date_str = params['date_str']
    cx_history = np.array(cx_history)
    mut_history = np.array(mut_history)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, name in enumerate(cx_names):
        axes[0].plot(cx_history[:, i], label=name)
    axes[0].set_title('Crossover Operator Selection Probabilities')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Probability')
    axes[0].legend()
    axes[0].grid(True)

    for i, name in enumerate(mut_names):
        axes[1].plot(mut_history[:, i], label=name)
    axes[1].set_title('Mutation Operator Selection Probabilities')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Probability')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'results/{date_str}/aos_operator_statistics.png', dpi=300)
    plt.close()

    print("\n--- AOS Final Statistics ---")
    print(f"Crossover usage:  {dict(zip(cx_names, cx_aos.usage_count.astype(int)))}")
    print(f"Crossover probs:  {dict(zip(cx_names, np.round(cx_aos.probabilities, 4)))}")
    print(f"Crossover quality:{dict(zip(cx_names, np.round(cx_aos.quality, 6)))}")
    print(f"Mutation usage:   {dict(zip(mut_names, mut_aos.usage_count.astype(int)))}")
    print(f"Mutation probs:   {dict(zip(mut_names, np.round(mut_aos.probabilities, 4)))}")
    print(f"Mutation quality: {dict(zip(mut_names, np.round(mut_aos.quality, 6)))}")
    print("--- End AOS Statistics ---\n")