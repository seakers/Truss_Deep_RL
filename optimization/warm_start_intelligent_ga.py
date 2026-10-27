import numpy as np
import torch

from evaluation.truss_problem import TrussProblem
from optimization.design_synthesis import get_models as get_model_warm_start
from optimization.design_repair import get_models as get_model_mutation
from optimization.ga_hybrid_common import run_hybrid_ga, intelligent_mutation_operator


def run_warm_start_intelligent_ga(max_values, params):
    print("\n\nRunning Warm-Start Intelligent GA...\n\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_problem = TrussProblem(overlap_constraint=params.get('overlap_constraint', False))
    folder = f"results/{params['date_str']}"

    # Repair actor for the mutation operator (unchanged filename) [7][8]
    repair_actor, _ = get_model_mutation(device, params, setup_problem, load_models=folder)
    repair_actor.eval()

    # Synthesis actor for warm-start seeding (unchanged filename) [6][7]
    def init_population_fn(static_ranges, pop_size, rng):
        actor, _ = get_model_warm_start(device, params, setup_problem, load_models=folder)
        actor.eval()
        num_objectives = setup_problem.num_objectives          # 3 [5][10]
        designs = []
        with torch.no_grad():
            for _ in range(pop_size):
                w = rng.random(num_objectives); w = w / w.sum()  # [2]
                genes, _ = actor.sample_action(w, setup_problem)  # [5]
                designs.append([int(x) for x in genes])
        return designs

    def mutation_fn(child):
        return intelligent_mutation_operator(child, repair_actor, device, params,
                                             eval_func=setup_problem)

    return run_hybrid_ga(max_values, params, eval_func=setup_problem,
                         init_population_fn=init_population_fn, mutation_fn=mutation_fn)