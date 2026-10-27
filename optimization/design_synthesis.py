import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
from copy import deepcopy

from optimization.ppo_architectures import Actor, Critic
from evaluation.truss_problem import TrussProblem
from utils.pareto import postprocess_run

def run_ppo_rand_weights_state(max_values, params, run_idx=0, best_hv_so_far=-np.inf):
    """
    Run PPO with updated reward handling for 3 Objectives + 2 Constraint Values.
    """
    # print("\n\nRunning PPO Optimization...\n\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    epochs = params['num_epochs']
    date_str = params['date_str']

    problem = TrussProblem(
        overlap_constraint=params['overlap_constraint'],
        load_factor=params.get('load_factor', 1.0),     # NEW
        load_node=params.get('load_node', 16),          # NEW
    )
    num_objectives = problem.num_objectives
    objective_min_max = problem.objective_min_max
    max_objectives = max_values[:num_objectives]
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])

    # Update model initialization to handle 5 output dimensions
    actor, critic = get_models(device, params, problem)
    ctx = params.get('context_vector')                  # NEW
    actor.set_context(ctx); critic.set_context(ctx)     # NEW — no-op if ctx is None [4]

    NFE = 0
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    all_actor_loss, all_critic_loss, all_avg_obj, all_kl = [], [], [], []

    for epoch in range(epochs):
        if epoch % 10 == 9:
            print(f"Epoch {epoch + 1}/{epochs}")
            
        actor, critic, NFE, epoch_data, stats = run_epoch(
            actor, critic, NFE, max_values, min_mask, device, params, problem
        )

        all_des.extend(epoch_data['des'])
        all_obj.extend(epoch_data['obj'])
        all_constraints.extend(epoch_data['constraints'])
        all_constraint_vals.extend(epoch_data['constraint_vals'])
        
        all_actor_loss.append(stats['actor_loss'])
        all_critic_loss.append(stats['critic_loss'])
        all_avg_obj.append(stats['avg_obj'])
        all_kl.append(stats['kl'])

    post_process_package = postprocess_run(
        all_des, all_obj, all_constraints, all_constraint_vals,
        objective_min_max, max_values,
    )

    all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE = post_process_package

    # ── Best-HV model saving ──────────────────────────────────────────────────
    run_best_hv = hypervolumes[-1] if hypervolumes else 0.0

    # Always save the latest model (overwrites previous run's file)
    torch.save(actor.state_dict(),
            f'results/{date_str}/actor_model.pth')
    torch.save(critic.state_dict(),
            f'results/{date_str}/critic_model.pth')

    print(f"  [RL Informed Env] Run HV = {run_best_hv:.6f} | "
        f"Best so far = {best_hv_so_far:.6f}")

    # Simple plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.plot(all_actor_loss, label='Actor Loss', color='blue'); plt.title('Actor Loss')
    plt.subplot(1, 3, 2); plt.plot(all_critic_loss, label='Critic Loss', color='orange'); plt.title('Critic Loss')
    plt.subplot(1, 3, 3); plt.plot(all_kl, label='KL Divergence', color='green'); plt.title('KL Divergence')
    plt.tight_layout()
    plt.savefig(f'results/{date_str}/ppo_training_results_run_{run_idx}.png')

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE, run_best_hv


def get_models(device, params, problem, load_models=None):
    gene_space = problem.design_space
    unique_des_space = problem.unique_design_space
    actor = Actor(device, params, gene_space, unique_des_space,
                  problem.num_objectives).to(device)
    critic = Critic(device, params, problem.num_objectives,
                    problem.num_constraints).to(device)

    if params['model_folder'] is not None:
        actor.load_state_dict(torch.load(f"{params['model_folder']}/actor_model.pth"))
        critic.load_state_dict(torch.load(f"{params['model_folder']}/critic_model.pth"))
        print("Loaded pre-trained models.")

    if load_models is not None:
        actor.load_state_dict(torch.load(f"{load_models}/actor_model.pth"))
        critic.load_state_dict(torch.load(f"{load_models}/critic_model.pth"))
        print("Loaded pre-trained models from specified path.")

    return actor, critic


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def run_epoch(actor, critic, NFE, max_values, min_mask, device, params, eval_function):
    rng = np.random.default_rng()
    mini_batch_size = params['mini_batch_size']
    num_objectives = eval_function.num_objectives
    constraint_penalty = params.get('constraint_penalty', 1.0)
    max_vals = np.asarray(max_values, dtype=float)[:num_objectives]
    min_mask_obj = min_mask[:num_objectives]
    obj_worst = np.where(min_mask_obj, max_vals, 0.0)

    batch_weights, batch_genes, batch_logprobs = [], [], []
    batch_rewards, batch_targets = [], []
    epoch_des, epoch_obj, epoch_constraints, epoch_constraint_vals = [], [], [], []

    def scalarize(objectives, cvals, weights, violated):
        obj = np.asarray(objectives, dtype=float)
        if violated:
            obj = obj_worst.copy()
        signed = np.where(min_mask_obj, -obj, obj)
        return float(np.dot(weights, signed)) - constraint_penalty * float(np.sum(cvals))

    for _ in range(mini_batch_size):
        w = rng.random(num_objectives); w = w / w.sum()
        genes, log_prob = actor.sample_action(w, eval_function)
        objectives, is_constrained, cvals = eval_function.evaluate(genes)
        NFE += 1
        reward = scalarize(objectives, cvals, w, is_constrained)

        batch_weights.append(w); batch_genes.append(genes)
        batch_logprobs.append(float(log_prob.item())); batch_rewards.append(reward)
        batch_targets.append(list(objectives) + list(cvals))
        epoch_des.append(list(genes)); epoch_obj.append(objectives)
        epoch_constraints.append(is_constrained); epoch_constraint_vals.append(cvals)

    values = []
    for genes, w in zip(batch_genes, batch_weights):
        pred = critic.value(genes).cpu().numpy()
        values.append(scalarize(pred[:num_objectives], pred[num_objectives:], w,
                                violated=float(np.sum(pred[num_objectives:])) > 0.0))
    advantages = np.asarray(batch_rewards) - np.asarray(values)
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # --- Updates: target-KL early stop on the actor, then the critic ---
    actor_loss, kl = 0.0, 0.0
    for _ in range(params['update_iterations']):
        actor_loss, kl = actor.ppo_update(
            batch_weights, batch_genes, None, batch_logprobs, advantages)
        if kl > params['target_kl']:
            break

    critic_loss = critic.ppo_update(batch_genes, batch_targets)

    epoch_data = {'des': epoch_des, 'obj': epoch_obj,
                  'constraints': epoch_constraints,
                  'constraint_vals': epoch_constraint_vals}
    stats = {'actor_loss': actor_loss, 'critic_loss': critic_loss, 'kl': kl,
             'avg_obj': np.mean(np.asarray(epoch_obj, dtype=float), axis=0)}
    return actor, critic, NFE, epoch_data, stats