import numpy as np
import torch

from evaluation.truss_problem import TrussProblem
from optimization.design_synthesis import get_models
from optimization.ga_hybrid_common import run_hybrid_ga


def run_warm_start_ga(max_values, params):
    print("\n\nRunning Warm-Start GA...\n\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_problem = TrussProblem(overlap_constraint=params['overlap_constraint'])

    def init_population_fn(static_ranges, pop_size, rng):
        # Load synthesis actor (unchanged filename via get_models) [2]
        actor, _ = get_models(device, params, setup_problem,
                              load_models=f"results/{params['date_str']}")
        actor.eval()

        num_objectives = setup_problem.num_objectives          # 3 — actor uses these [5][10]
        designs = []
        with torch.no_grad():
            for _ in range(pop_size):
                w = rng.random(num_objectives)
                w = w / w.sum()                                 # normalized weights [2]
                genes, _ = actor.sample_action(w, setup_problem)  # full gene vector [5]
                designs.append([int(x) for x in genes])
        return designs

    return run_hybrid_ga(max_values, params, eval_func=setup_problem,
                         init_population_fn=init_population_fn)   # random mutation