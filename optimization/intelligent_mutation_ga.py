import torch

from evaluation.truss_problem import TrussProblem
from optimization.design_repair import get_models
from optimization.ga_hybrid_common import run_hybrid_ga, intelligent_mutation_operator


def run_intelligent_mutation_ga(max_values, params):
    print("\n\nRunning Intelligent Mutation GA...\n\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_problem = TrussProblem(overlap_constraint=params.get('overlap_constraint', False))

    # Repair actor for the mutation operator (unchanged filename via get_models) [4][8]
    repair_actor, _ = get_models(device, params, setup_problem,
                                 load_models=f"results/{params['date_str']}")
    repair_actor.eval()

    def mutation_fn(child):
        return intelligent_mutation_operator(child, repair_actor, device, params,
                                             eval_func=setup_problem)

    return run_hybrid_ga(max_values, params, eval_func=setup_problem,
                         mutation_fn=mutation_fn)