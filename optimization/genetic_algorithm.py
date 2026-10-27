"""Custom NSGA-II GA for the discrete 72-bar TrussProblem, using utils.pareto
(no pymoo/pygad). Replaces the previous PyGAD implementation [3].

Core NSGA-II operators (constrained non-dominated sort, crowding distance,
tournament, single-point-style crossover, random mutation) are self-contained.
The truss interface is static: evaluate(design) takes one arg and returns
(objectives, is_constrained, constraint_vals) [10]; no reset()/filter_des_space.
Output format matches the RL methods via postprocess_run [2][3].
"""
import numpy as np

from evaluation.truss_problem import TrussProblem
from utils.pareto import postprocess_run


# ---- NSGA-II operators ----
def _dominates(a, b, min_mask):
    better = np.where(min_mask, a < b, a > b)
    worse = np.where(min_mask, a > b, a < b)
    return np.any(better) and not np.any(worse)


def _constrained_dominates(a, b, viol_a, viol_b, min_mask, tol=1e-9):
    """Deb constrained-domination: feasibility first, then Pareto dominance."""
    a_feasible = viol_a <= tol
    b_feasible = viol_b <= tol
    if a_feasible and not b_feasible:
        return True
    if b_feasible and not a_feasible:
        return False
    if not a_feasible and not b_feasible:
        return viol_a < viol_b
    return _dominates(a, b, min_mask)


def _fast_nondominated_sort(objs, min_mask, viol):
    n = len(objs)
    S = [[] for _ in range(n)]
    ndom = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _constrained_dominates(objs[p], objs[q], viol[p], viol[q], min_mask):
                S[p].append(q)
            elif _constrained_dominates(objs[q], objs[p], viol[q], viol[p], min_mask):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding_distance(objs):
    n, m = objs.shape
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(objs[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = objs[order[-1], k] - objs[order[0], k]
        if span == 0:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (objs[order[i + 1], k] - objs[order[i - 1], k]) / span
    return dist


def _random_gene(var, rng):
    """Truss genes are all discrete ('Area' or 'Sel') [10]."""
    if var["type"] == "continuous":
        return float(rng.uniform(var["range"][0], var["range"][1]))
    return int(rng.choice(np.array(var["range"])))


def _crossover(p1, p2, rng):
    mask = rng.random(len(p1)) < 0.5
    return [p1[i] if mask[i] else p2[i] for i in range(len(p1))]


def _mutate(genes, gene_space, rng, rate=0.1):
    child = list(genes)
    for i, var in enumerate(gene_space):
        if rng.random() < rate:
            child[i] = _random_gene(var, rng)
    return child


def run_genetic_algorithm(max_values, params):
    print("\n\nRunning Genetic Algorithm (custom NSGA-II)...\n\n")
    rng = np.random.default_rng()

    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']

    setup_problem = TrussProblem(overlap_constraint=params.get('overlap_constraint', False))
    template_des_space = setup_problem._design_space_template

    num_objectives = setup_problem.num_objectives
    objective_min_max = setup_problem.objective_min_max
    # min_mask over objectives only (3) for domination; constraints handled via viol
    min_mask = np.array([mm.lower() == 'min' for mm in objective_min_max[:num_objectives]])

    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    NFE = 0

    # ---- evaluation (truss static interface: evaluate(design), one arg) [10] ----
    def evaluate(raw_solution):
        nonlocal NFE
        discrete_solution = [int(x) for x in raw_solution]
        obj_values, is_constrained, constraint_vals = setup_problem.evaluate(discrete_solution)
        NFE += 1

        all_des.append(discrete_solution)
        all_obj.append(obj_values)                 # raw objectives -> HV/Pareto
        all_constraints.append(is_constrained)
        all_constraint_vals.append(constraint_vals)

        obj = np.array(obj_values, dtype=float)
        viol = float(np.sum(constraint_vals)) if is_constrained else 0.0
        return discrete_solution, obj, viol

    # ---- init population ----
    population = [[_random_gene(v, rng) for v in template_des_space] for _ in range(pop_size)]
    pop_repaired, pop_obj, pop_viol = [], [], []
    for genes in population:
        d, obj, viol = evaluate(genes)
        pop_repaired.append(d); pop_obj.append(obj); pop_viol.append(viol)
    population, pop_obj, pop_viol = pop_repaired, np.array(pop_obj), np.array(pop_viol)

    # ---- generations ----
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
            offspring.append(_mutate(_crossover(p1, p2, rng), template_des_space, rng))

        off_repaired, off_obj, off_viol = [], [], []
        for genes in offspring:
            d, obj, viol = evaluate(genes)
            off_repaired.append(d); off_obj.append(obj); off_viol.append(viol)

        combined = population + off_repaired
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

    # ---- post-processing (same helper/format as RL methods) [2][3] ----
    return postprocess_run(
        all_des, all_obj, all_constraints, all_constraint_vals,
        objective_min_max, max_values,
    )