import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from copy import deepcopy

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import is_pareto_efficient, track_pareto_progress
from optimization.transformer_design_repair import Actor, Critic

def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def run_design_repair_ppo(max_values, params, run_idx=0, best_hv_so_far=-np.inf):
    """
    Run PPO for design repair: modify existing designs one variable at a time.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    epochs = params['num_epochs']
    mini_batch_size = params['mini_batch_size']
    
    eval_function = TrussProblem()
    num_objectives = eval_function.num_objectives
    objective_min_max = eval_function.objective_min_max
    num_actions = eval_function.n_elements
    action_range = len(eval_function.area_map)
    
    max_objectives = max_values[:num_objectives]
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])
    
    ref_point = np.ones(num_objectives)
    hv_ind = HV(ref_point=ref_point)

    num_total_objs = len(max_values)    
    actor, critic = get_models(device, params, num_total_objs)
    
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
    
    # Post-process results (similar to your existing code)
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    
    ref_values = max_objectives * ref_point
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values
    
    print(f"Number of valid designs: {np.sum(all_constraints == False)}")
    
    norm_obj = all_obj / max_objectives
    hv_obj = deepcopy(norm_obj)
    for i in range(num_objectives):
        if objective_min_max[i].lower() == 'max':
            hv_obj[:,i] = 1 - norm_obj[:,i]
    
    # Calculate Pareto front and hypervolume over time
    pareto_front_des, pareto_front_obj, hypervolumes = track_pareto_progress(
        hv_obj, all_des, all_constraints, hv_ind
    )
    
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


def get_models(device, params, num_total_objs):
    problem = TrussProblem()
    num_actions = problem.n_elements
    action_length = len(problem.area_map)

    actor = Actor(device=device, params=params, num_variables=num_actions,
                  action_length=action_length, num_objectives=num_total_objs)

    # CHANGE: remove the spurious +2. num_total_objs is already 5 (3 obj + 2 cv)
    # so input_dim = 72 + 5 = 77, matching new_design + new_obj + new_constraint_vals
    critic = Critic(device=device, params=params, num_objectives=num_total_objs,
                    input_dim=num_actions + num_total_objs)  # 72 + 5 = 77

    actor.to(device)
    critic.to(device)

    # Dummy tensors must also match the corrected 77 dims -- or [1, 72, 2] when the
    # per-bar utilisation channel is on, since then the token count is the design only.
    if params.get('use_stress', False):
        input_dummy = torch.zeros((1, num_actions, 2), dtype=torch.float32).to(device)
    else:
        input_dummy = torch.zeros((1, num_actions + num_total_objs), dtype=torch.float32).to(device)
    actor(input_dummy)
    critic(input_dummy)  # Same dummy works for both now

    if params['model_folder'] is not None:
        actor.load_state_dict(torch.load(f"{params['model_folder']}/actor_repair_model.pth"))
        critic.load_state_dict(torch.load(f"{params['model_folder']}/critic_repair_model.pth"))
        print("Loaded pre-trained models.")

    return actor, critic


def run_repair_epoch(actor, critic, all_des, all_obj, all_constraints, designs, objs, eval_function, mini_batch_size, 
                     NFE, max_objectives, min_mask, device, params):
    """
    Run one epoch: modify each design in batch, collect trajectories, update models.
    """
    num_objectives = len(max_objectives)
    num_actions = eval_function.n_elements
    action_range = len(eval_function.area_map)
    
    # Storage for trajectories
    batch_observations = [[] for _ in range(mini_batch_size)]
    batch_actions = [[] for _ in range(mini_batch_size)]  # (var_idx, new_value)
    batch_logprobs = [[] for _ in range(mini_batch_size)]
    batch_rewards = [[] for _ in range(mini_batch_size)]
    batch_designs = [[] for _ in range(mini_batch_size)]
    
    epoch_des = []
    epoch_obj = []
    epoch_constraints = []
    epoch_constraint_vals = []
    
    use_stress = params.get('use_stress', False)
    max_edits = params.get('max_edits', 200)

    # Run modification steps
    for idx in range(len(designs)):
        stop_token = False
        current_design = designs[idx]
        # These recompute a solve already paid for when this design was sampled, so they
        # cost wall-clock but no NFE. Only needed for the channels/rewards that use them.
        if use_stress or params.get('reward_shape', 'domination') == 'potential':
            _, _, current_cv, current_util = eval_function.evaluate(
                current_design, return_utilisation=True)
        else:
            current_cv, current_util = None, None
        current_obj = objs[idx]
        n_edits = 0

        while not stop_token and n_edits < max_edits:
            n_edits += 1

            # Sample actions from actor. With use_stress the state is [tokens, 2]:
            # the design in channel 0 and the per-bar utilisation in channel 1.
            if use_stress:
                state = np.stack([np.asarray(current_design, dtype=float),
                                  np.asarray(current_util, dtype=float)], axis=-1)
            else:
                state = np.asarray(current_design, dtype=float)
            var_log_prob, var_idx, value_log_prob, value_idx, stop_log_prob, stop_decision = actor.sample_action(
                torch.tensor(state, dtype=torch.float32).to(device)
            )

            # Create new design
            new_design = current_design.copy()
            new_design[var_idx.item()] = value_idx.item()

            # Evaluate new design
            new_obj, new_constraints, new_constraint_vals, new_util = eval_function.evaluate(
                new_design, return_utilisation=True)
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
                min_mask, params,
                parent_cv=current_cv, child_cv=new_constraint_vals,
                max_objectives=max_objectives
            )

            # Store trajectory
            batch_actions[idx].append([var_idx.item(), value_idx.item(), stop_decision.item()])
            batch_logprobs[idx].append([var_log_prob.item(), value_log_prob.item(), stop_log_prob.item()])
            batch_rewards[idx].append(reward)
            batch_designs[idx].append(new_design)
            
            # Must match the state sample_action was called on, or ppo_update recomputes
            # log-probs at states the policy never visited. 'full' keeps the old
            # mismatched behaviour for comparison.
            if use_stress:
                # must match the state the policy acted in, channel for channel
                new_obs = np.stack([np.asarray(new_design, dtype=float),
                                    np.asarray(new_util, dtype=float)], axis=-1).tolist()
            elif params.get('obs_mode', 'design') == 'full':
                new_obs = new_design + new_obj + list(new_constraint_vals)
            else:
                new_obs = list(new_design)
            current_design = new_design
            current_util = new_util
            if params.get('reward_shape', 'domination') == 'potential':
                # shaping must compare CONSECUTIVE states for the telescoping to hold
                current_obj, current_cv = new_obj, new_constraint_vals
            # otherwise current_obj stays at the episode's starting design, which is what
            # the domination reward compares against
            batch_observations[idx].append(new_obs)
                    
            # Check if stop decision was made
            if stop_decision.item() == 1:  # Stop decision is 1 (stop)
                stop_token = True
        
    
    # Get critic values for all observations
    critic_values = [[] for _ in range(mini_batch_size)]
    with torch.no_grad():
        for idx in range(mini_batch_size):
            obs_tensor = torch.tensor([batch_observations[idx]], dtype=torch.float32).to(device)
            values = np.array(critic.sample_critic(obs_tensor).tolist())
            critic_values[idx] = values
    
    # Calculate advantages and returns
    gamma = params['gamma']
    lam = params['lambda']
    all_advantages, all_returns = calculate_gae(
        batch_rewards, critic_values, gamma, lam
    )
    
    # --- 5. Prepare Tensors ---
    observation_tensors = []
    action_tensors = []
    logprob_tensors = []
    advantage_tensors = []
    return_tensors = []
    
    for batch_idx in range(mini_batch_size):
        observation_tensors.append(torch.tensor(batch_observations[batch_idx], dtype=torch.float32).to(device))
        action_tensors.append(torch.tensor(batch_actions[batch_idx], dtype=torch.float32).to(device))
        logprob_tensors.append(torch.tensor(batch_logprobs[batch_idx], dtype=torch.float32).to(device))
        advantage_tensors.append(torch.tensor(all_advantages[batch_idx], dtype=torch.float32).to(device))
        return_tensors.append(torch.tensor(all_returns[batch_idx].copy(), dtype=torch.float32).to(device))


    # --- 6. Updates ---
    targetkl = params['target_kl']
    kl = 0
    actor_loss = 0
    
    for _ in range(params['update_iterations']):
        actor_loss, kl = actor.ppo_update(
            observation_tensors, 
            action_tensors, 
            logprob_tensors,
            advantage_tensors
        )
        if kl > targetkl:
            break

    critic_loss = 0
    for _ in range(params['update_iterations']):
        critic_loss = critic.ppo_update(
            observation_tensors, 
            return_tensors, 
        )
    
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


def design_potential(obj, constraint_vals, max_objectives, min_mask, lam=1.0):
    """Phi(design): higher is better, a function of the design's own objectives only.

    Each objective is normalised so 0 is ideal and 1 is the reference, the maximisation
    objective is flipped, and constraint violation is folded in. Being a function of the
    state alone is what makes the shaped reward below policy-invariant.
    """
    z = np.asarray(obj, dtype=float) / np.asarray(max_objectives, dtype=float)
    z = np.where(min_mask[:len(z)], z, 1.0 - z)          # flip maximisation axes
    viol = float(np.sum(np.asarray(constraint_vals, dtype=float)))
    return -(float(np.mean(np.clip(z, 0.0, None))) + lam * viol)


def calculate_step_reward(parent_obj, child_obj, is_constrained, min_mask, params,
                          parent_cv=None, child_cv=None, max_objectives=None):
    """
    Calculate reward for a single modification step.

    reward_shape='domination' (default) is the original: +1 if the child dominates the
    parent, -0.5 if dominated, -0.01 for a tie, -1 if infeasible. Every edit has negative
    expected reward, so this punishes episode length.

    reward_shape='potential' pays the change in design quality instead, which telescopes
    over an episode so the total does not depend on how many edits it took. This is
    potential-based shaping (Ng, Harada & Russell 1999).
    """
    shape = params.get('reward_shape', 'domination')

    if shape == 'potential':
        if max_objectives is None or parent_cv is None or child_cv is None:
            raise ValueError("reward_shape='potential' needs parent_cv, child_cv and "
                             "max_objectives")
        scale = params.get('potential_scale', 10.0)
        phi_new = design_potential(child_obj, child_cv, max_objectives, min_mask)
        phi_old = design_potential(parent_obj, parent_cv, max_objectives, min_mask)

        # Explicit price per edit, since the shaping term is length-neutral. This does
        # break potential-invariance and so changes the optimal policy, which is intended:
        # the operator should prefer the same improvement in fewer edits.
        step_cost = params.get('step_cost', 0.0)
        return scale * (phi_new - phi_old) - step_cost

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
    while len(valid_designs) < num_designs:
        design = np.random.randint(0, action_range, num_actions).tolist()
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
    
    # 10% random, 90% from previous results
    num_random = num_designs // 10
    num_from_previous = num_designs - num_random
    
    # Random designs
    while len(designs) < num_random:
        design = np.random.randint(0, action_range, num_actions).tolist()
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
        design = np.random.randint(0, action_range, num_actions).tolist()
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