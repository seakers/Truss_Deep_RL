import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from copy import deepcopy

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import is_pareto_efficient
from optimization.transformer_design_repair import Actor, Critic

def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def run_design_repair_ppo(max_values, params):
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
    
    actor, critic = get_models(device, params, num_objectives)
    
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
    pareto_front_des, pareto_front_obj, hypervolumes = calculate_pareto_progress(
        all_des, all_obj, all_constraints, hv_obj, hv_ind, NFE
    )
    
    # Save and plot results
    torch.save(actor.state_dict(), f'results/{params["date_str"]}/actor_repair_model.pth')
    torch.save(critic.state_dict(), f'results/{params["date_str"]}/critic_repair_model.pth')
    plot_training_results(all_actor_loss, all_critic_loss, all_kl, all_avg_reward, params)
    
    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE


def get_models(device, params, num_total_objs):
    # Briefly instantiate problem to get input dimensions
    problem = TrussProblem()
    num_actions = problem.n_elements
    action_length = len(problem.area_map)

    # Actor input: action_length (sequence length)
    actor = Actor(device=device, params=params, num_variables=num_actions, action_length=action_length, num_objectives=num_total_objs)
    # Critic output: total_dims (5) to match weights vector
    critic = Critic(device=device, params=params, num_objectives=num_total_objs, input_dim=num_actions+num_total_objs)

    actor.to(device)
    critic.to(device)
    
    # Initialize lazy layers
    input_dummy = torch.zeros((1, num_actions+num_total_objs), dtype=torch.float32).to(device)
    actor(input_dummy)
    critic(input_dummy)

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
    
    # Run modification steps
    for idx in range(len(designs)):
        stop_token = False
        current_design = designs[idx]

        while not stop_token:
            current_obj = objs[idx]
            
            # Sample actions from actor
            var_log_prob, var_idx, value_log_prob, value_idx, stop_log_prob, stop_decision = actor.sample_action(
                torch.tensor(current_design, dtype=torch.float32).to(device)
            )
            
            # Create new design
            new_design = current_design.copy()
            new_design[var_idx.item()] = value_idx.item()
            
            # Evaluate new design
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
            batch_actions[idx].append([var_idx.item(), value_idx.item(), stop_decision.item()])
            batch_logprobs[idx].append([var_log_prob.item(), value_log_prob.item(), stop_log_prob.item()])
            batch_rewards[idx].append(reward)
            batch_designs[idx].append(new_design)
            
            new_obs = new_design + new_obj
            current_design = new_design
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
        values = np.append(values, values[-1][-1])
        
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


def calculate_pareto_progress(all_des, all_obj, all_constraints, hv_obj, hv_ind, NFE):
    """Calculate Pareto front and hypervolume over time (from your original code)"""
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
    
    return pareto_front_des, pareto_front_obj, hypervolumes


def plot_training_results(actor_loss, critic_loss, kl, avg_reward, params):
    """Plot training metrics"""
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1); plt.plot(actor_loss); plt.title('Actor Loss')
    plt.subplot(1, 4, 2); plt.plot(critic_loss); plt.title('Critic Loss')
    plt.subplot(1, 4, 3); plt.plot(kl); plt.title('KL Divergence')
    plt.subplot(1, 4, 4); plt.plot(avg_reward); plt.title('Avg Reward per Episode')
    plt.tight_layout()
    plt.savefig(f"results/{params['date_str']}/repair_training_results.png")