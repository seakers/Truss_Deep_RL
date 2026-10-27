import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
from copy import deepcopy

from evaluation.truss_problem import TrussProblem
from optimization.ppo_architectures import RepairActor, RepairCritic
from utils.pareto import postprocess_run, is_pareto_efficient

def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def run_design_repair_ppo(max_values, params, run_idx=0, best_hv_so_far=-np.inf):
    """
    Run PPO for design repair: modify existing designs one variable at a time.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    epochs = params['num_epochs']
    mini_batch_size = params['mini_batch_size']
    
    eval_function = TrussProblem(
        overlap_constraint=params['overlap_constraint'],
        load_factor=params.get('load_factor', 1.0),     # NEW
        load_node=params.get('load_node', 16),          # NEW
    )

    num_objectives = eval_function.num_objectives
    objective_min_max = eval_function.objective_min_max
    num_actions = eval_function.n_elements
    action_range = len(eval_function.area_map)
    
    max_objectives = max_values[:num_objectives]
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])
    
    actor, critic = get_models(device, params, eval_function)
    ctx = params.get('context_vector')                  # NEW
    actor.set_context(ctx); critic.set_context(ctx)     # NEW — no-op if ctx is None [4]
    
    NFE = 0
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    all_actor_loss, all_critic_loss, all_avg_reward, all_kl = [], [], [], []
      
    while NFE < epochs * mini_batch_size:
        epoch = NFE // mini_batch_size
        if epoch % 10 == 9:
            print(f"Epoch {epoch + 1}/{epochs}, NFE: {NFE}")

        if NFE == 0:
            # Generate initial random designs
            print("Generating initial designs...")
            initial_designs, initial_objs, all_des, all_obj, all_constraints, NFE = generate_initial_designs(
                all_des, all_obj, all_constraints, 
                eval_function, mini_batch_size, num_actions, action_range, NFE
            )
        else:
            initial_designs, initial_objs, all_des, all_obj, all_constraints, NFE = sample_next_batch(
                all_des, all_obj, all_constraints, 
                eval_function, mini_batch_size, num_actions, action_range, NFE
            )
        
        # Run epoch with current batch of designs
        epoch_data, stats, all_des, all_obj, all_constraints, NFE = run_repair_epoch(
            actor,
            critic,
            all_des,
            all_obj,
            all_constraints,
            initial_designs,
            initial_objs,
            eval_function,
            mini_batch_size,
            NFE,
            max_objectives,
            min_mask,
            device,
            params
        )
        
        # Store results
        all_des.extend(epoch_data['des'])
        all_obj.extend(epoch_data['obj'])
        all_constraints.extend(epoch_data['constraints'])
        all_constraint_vals.extend(epoch_data['constraint_vals'])
        
        all_actor_loss.append(stats['actor_loss'])
        all_critic_loss.append(stats['critic_loss'])
        all_avg_reward.append(stats['avg_reward'])
        all_kl.append(stats['kl'])
        
        # Sample new initial designs for next epoch (mix of random and previous designs)
    
    post_process_package = postprocess_run(
        all_des, all_obj, all_constraints, all_constraint_vals,
        objective_min_max, max_values,
    )

    all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE = post_process_package
    
    # ── Best-HV model saving ──────────────────────────────────────────────────
    run_best_hv = hypervolumes[-1] if hypervolumes else 0.0

    # Always save the latest model (main() will copy to best_* if it wins)
    torch.save(actor.state_dict(),
               f'results/{params["date_str"]}/actor_repair_model.pth')
    torch.save(critic.state_dict(),
               f'results/{params["date_str"]}/critic_repair_model.pth')

    print(f"  [Design Repair] Run HV = {run_best_hv:.6f} | "
          f"Best so far = {best_hv_so_far:.6f}")
    
    plot_training_results(all_actor_loss, all_critic_loss, all_kl, all_avg_reward, params, suffix=f"_run_{run_idx}")
    
    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE, run_best_hv



def get_models(device, params, problem, load_models=None):
    gene_space = problem.design_space
    unique_des_space = problem.unique_design_space
    actor = RepairActor(device, params, gene_space, unique_des_space,
                        problem.num_objectives).to(device)
    critic = RepairCritic(device, params, len(gene_space)).to(device)

    if params['model_folder'] is not None:
        actor.load_state_dict(torch.load(f"{params['model_folder']}/actor_repair_model.pth"))
        critic.load_state_dict(torch.load(f"{params['model_folder']}/critic_repair_model.pth"))
        print("Loaded pre-trained models.")

    if load_models is not None:
        actor.load_state_dict(torch.load(f"{load_models}/actor_repair_model.pth"))
        critic.load_state_dict(torch.load(f"{load_models}/critic_repair_model.pth"))
        print("Loaded pre-trained models from specified path.")
        
    return actor, critic


def encode_design(design, gene_space):
    enc = []
    for i, val in enumerate(design):
        opts = list(gene_space[i]['range'])
        enc.append(opts.index(val) / max(1, len(opts) - 1))
    return enc


def run_repair_epoch(actor, critic, all_des, all_obj, all_constraints, designs, objs, eval_function, mini_batch_size, 
                     NFE, max_objectives, min_mask, device, params):
    """
    Run one epoch: modify each design in batch, collect trajectories, update models.
    """
    num_objectives = len(max_objectives)
    num_actions = eval_function.n_elements
    action_range = len(eval_function.area_map)
    gene_space = eval_function.design_space
    
    # Storage for trajectories
    batch_observations = [[] for _ in range(mini_batch_size)]
    batch_actions = [[] for _ in range(mini_batch_size)]  # (var_idx, new_value)
    batch_logprobs = [[] for _ in range(mini_batch_size)]
    batch_rewards = [[] for _ in range(mini_batch_size)]
    batch_designs = [[] for _ in range(mini_batch_size)]
    batch_infos = [[] for _ in range(mini_batch_size)]
    batch_states = [[] for _ in range(mini_batch_size)]
    
    epoch_des = []
    epoch_obj = []
    epoch_constraints = []
    epoch_constraint_vals = []
    
    # Run modification steps
    for idx in range(len(designs)):
        stop_token = False
        current_design = designs[idx]

        while not stop_token:
            current_obj = objs[idx]
            
            state = torch.tensor(encode_design(current_design, gene_space),
                                dtype=torch.float32, device=device)
            var_idx, new_value, stop_decision, log_prob, info = actor.sample_action(state)

            new_design = list(current_design)
            new_design[var_idx] = new_value
            new_obj, new_constraints, new_constraint_vals = eval_function.evaluate(new_design)
            NFE += 1

            all_des.append(new_design)
            all_obj.append(new_obj)
            all_constraints.append(new_constraints)
            
            epoch_des.append(new_design)
            epoch_obj.append(new_obj)
            epoch_constraints.append(new_constraints)
            epoch_constraint_vals.append(new_constraint_vals)
            
            # Calculate reward (dense)
            reward = calculate_step_reward(
                current_obj, new_obj, 
                new_constraints,
                min_mask, params
            )
            
            # Store trajectory
            batch_states[idx].append(state.cpu().numpy())
            batch_infos[idx].append(info)
            batch_logprobs[idx].append(float(log_prob.item()))
            batch_rewards[idx].append(reward)
            batch_designs[idx].append(new_design)
            
            new_obs = new_design + new_obj + list(new_constraint_vals)
            current_design = new_design
            batch_observations[idx].append(new_obs)
                    
            # Check if stop decision was made
            if int(stop_decision) == 1:  # Stop decision is 1 (stop)
                stop_token = True
        
    
    # --- 4. Advantages and returns (scalar critic, per-trajectory GAE) ---
    gamma = params['gamma']
    lam = params['lambda']

    critic_values = [np.array([critic.value(s) for s in batch_states[idx]], dtype=float)
                     for idx in range(mini_batch_size)]
    all_advantages, all_returns = calculate_gae(
        batch_rewards, critic_values, gamma, lam
    )

    # --- 5. Flatten trajectories (ppo_update takes flat lists) ---
    traj_states, traj_infos, traj_logprobs = [], [], []
    for idx in range(mini_batch_size):
        traj_states.extend(batch_states[idx])
        traj_infos.extend(batch_infos[idx])
        traj_logprobs.extend(batch_logprobs[idx])

    advantages = np.concatenate(all_advantages)
    returns = np.concatenate(all_returns)
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # --- 6. Updates ---
    actor_loss, kl = 0.0, 0.0
    for _ in range(params['update_iterations']):
        actor_loss, kl = actor.ppo_update(
            traj_states, traj_infos, traj_logprobs, advantages
        )
        if kl > params['target_kl']:
            break

    critic_loss = critic.ppo_update(traj_states, returns)
    
    avg_reward = np.mean([sum(r) for r in batch_rewards])
    
    epoch_data = {
        'des': epoch_des,
        'obj': epoch_obj,
        'constraints': epoch_constraints,
        'constraint_vals': epoch_constraint_vals
    }
    
    stats = {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'kl': kl,
        'avg_reward': avg_reward
    }
    
    return epoch_data, stats, all_des, all_obj, all_constraints,NFE


def calculate_step_reward(parent_obj, child_obj, is_constrained, min_mask, params):
    """
    Calculate reward for a single modification step.
    Phase 1: Simple domination (+1 if child dominates parent)
    Phase 2: Could switch to hypervolume improvement
    """
    step_penalty = params.get('step_penalty', 0.01)
    
    if is_constrained:
        return -1.0 - step_penalty
    
    # Check if child dominates parent
    parent_arr = np.array(parent_obj)
    child_arr = np.array(child_obj)
    
    # For minimization: child < parent is better
    # For maximization: child > parent is better
    better = np.where(min_mask[:len(parent_obj)], 
                     child_arr < parent_arr, 
                     child_arr > parent_arr)
    worse = np.where(min_mask[:len(parent_obj)], 
                    child_arr > parent_arr, 
                    child_arr < parent_arr)
    
    if np.any(better) and not np.any(worse):
        return 1.0 - step_penalty  # Child dominates parent
    elif np.any(worse) and not np.any(better):
        return -0.5 - step_penalty  # Parent dominates child
    else:
        return -step_penalty  # Non-dominated


def calculate_gae(batch_rewards, batch_values, gamma, lam):
    """Calculate Generalized Advantage Estimation"""
    mini_batch_size = len(batch_rewards)
    all_advantages = []
    all_returns = []
    
    for idx in range(mini_batch_size):
        rewards = np.array(batch_rewards[idx])
        values = np.array(batch_values[idx].copy())
        values = np.append(values, values[-1])
        
        deltas = rewards + gamma * values[1:] - values[:-1]
        advantages = discounted_cumulative_sums(deltas, gamma * lam)
        returns = discounted_cumulative_sums(rewards, gamma)
        
        all_advantages.append(advantages)
        all_returns.append(returns)
    
    # Normalize advantages
    all_advantages_flat = np.concatenate(all_advantages)
    adv_mean, adv_std = np.mean(all_advantages_flat), np.std(all_advantages_flat)
    all_advantages = [(adv - adv_mean) / (adv_std + 1e-8) for adv in all_advantages]
    
    return all_advantages, all_returns


def generate_initial_designs(all_des, all_obj, all_constraints, eval_function, num_designs, num_actions, action_range, NFE):
    """Generate random initial designs"""
    valid_designs = []
    obj_values = []
    gene_space = eval_function.design_space
    while len(valid_designs) < num_designs:
        design = [int(np.random.choice(v['range'])) for v in gene_space]
        objs, is_constrained, constraint_vals = eval_function.evaluate(design)
        NFE += 1
        if not is_constrained:
            valid_designs.append(design)
            obj_values.append(objs)
        all_des.append(design)
        all_obj.append(objs)
        all_constraints.append(is_constrained)
            
    return valid_designs, obj_values, all_des, all_obj, all_constraints, NFE


def sample_next_batch(all_des, all_obj, all_constraints, eval_function, 
                     num_designs, num_actions, action_range, NFE):
    """
    Sample designs for next epoch (mix of random and good previous designs)
    """
    designs = []
    obj_values = []
    gene_space = eval_function.design_space
    
    # 10% random, 90% from previous results
    num_random = num_designs // 10
    num_from_previous = num_designs - num_random
    
    # Random designs
    while len(designs) < num_random:
        design = [int(np.random.choice(v['range'])) for v in gene_space]
        objs, is_constrained, constraint_vals = eval_function.evaluate(design)
        NFE += 1
        if not is_constrained:
            designs.append(design)
            obj_values.append(objs)
        all_des.append(design)
        all_obj.append(objs)
        all_constraints.append(is_constrained)
    
    # Sample from valid pfront designs
    valid_indices = [i for i, c in enumerate(all_constraints) if not c]
    pareto_mask = is_pareto_efficient(np.array(all_obj)[valid_indices])
    pareto_indices = np.nonzero(pareto_mask)[0]
    if len(pareto_indices) > 0:
        sampled_indices = np.random.choice(pareto_indices, 
                                          min(num_from_previous, len(pareto_indices)), 
                                          replace=False)
        for idx in sampled_indices:
            designs.append(all_des[idx])
            obj_values.append(all_obj[idx])
    
    # Fill remaining with random if needed
    while len(designs) < num_designs:
        design = [int(np.random.choice(v['range'])) for v in gene_space]
        objs, is_constrained, constraint_vals = eval_function.evaluate(design)
        NFE += 1
        if not is_constrained:
            designs.append(design)
            obj_values.append(objs)
        all_des.append(design)
        all_obj.append(objs)
        all_constraints.append(is_constrained)
    return designs, obj_values, all_des, all_obj, all_constraints, NFE


def plot_training_results(actor_loss, critic_loss, kl, avg_reward, params, suffix=""):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.plot(actor_loss, label='Actor Loss', color='blue'); plt.title('Actor Loss')
    plt.subplot(1, 3, 2); plt.plot(critic_loss, label='Critic Loss', color='orange'); plt.title('Critic Loss')
    plt.subplot(1, 3, 3); plt.plot(kl, label='KL Divergence', color='green'); plt.title('KL Divergence')
    plt.tight_layout()
    plt.savefig(f"results/{params['date_str']}/repair_training_results{suffix}.png")