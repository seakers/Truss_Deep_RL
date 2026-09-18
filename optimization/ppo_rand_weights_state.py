import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from copy import deepcopy

from optimization.transformer_rand_weights_state import Actor, Critic
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress, RunningParetoArchive

def run_ppo_rand_weights_state(max_values, params, run_idx=0, best_hv_so_far=-np.inf):
    """
    Run PPO with updated reward handling for 3 Objectives + 2 Constraint Values.
    """
    # print("\n\nRunning PPO Optimization...\n\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    epochs = params['num_epochs']
    date_str = params['date_str']

    temp_eval = TrussProblem()
    num_objectives = temp_eval.num_objectives
    objective_min_max = temp_eval.objective_min_max
    max_objectives = max_values[:num_objectives]
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])

    # Update model initialization to handle 5 output dimensions
    actor, critic = get_models(device, params, len(max_values))

    NFE = 0
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    all_actor_loss, all_critic_loss, all_avg_obj, all_kl = [], [], [], []
    ref_point = np.ones(num_objectives)
    hv_ind = HV(ref_point=ref_point)

    # Running front for the hypervolume-contribution reward. Built regardless, but
    # only read when params['reward_mode'] == 'hv'.
    def to_hv(obj_values):
        n = np.array(obj_values, dtype=float) / max_objectives
        p = np.where(min_mask[:num_objectives], n, 1.0 - n)
        return np.clip(p, 0.0, 1.0)          # points past the reference add no volume

    archive = RunningParetoArchive(hv_ind)
    # How often the archive resets. 0 never resets, so designs are scored against the
    # whole run; a period of n scores them against the last n epochs instead.
    reset_period = params.get('hv_reset_period')
    if reset_period is None:
        # back-compat: the older 'hv_archive' flag is equivalent to a period of 1
        reset_period = 1 if params.get('hv_archive', 'global') == 'batch' else 0

    for epoch in range(epochs):
        if reset_period and epoch % reset_period == 0:
            archive = RunningParetoArchive(hv_ind)
        if epoch % 10 == 9:
            print(f"Epoch {epoch + 1}/{epochs}")
            
        actor, critic, NFE, epoch_data, stats = run_epoch(
            actor, critic, NFE, max_values, min_mask, device, params,
            archive=archive, to_hv=to_hv
        )

        all_des.extend(epoch_data['des'])
        all_obj.extend(epoch_data['obj'])
        all_constraints.extend(epoch_data['constraints'])
        all_constraint_vals.extend(epoch_data['constraint_vals'])
        
        all_actor_loss.append(stats['actor_loss'])
        all_critic_loss.append(stats['critic_loss'])
        all_avg_obj.append(stats['avg_obj'])
        all_kl.append(stats['kl'])

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_constraint_vals = np.array(all_constraint_vals)

    ref_values = max_objectives * ref_point 
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values
    print(f"Number of valid designs (False in all_constraints): {np.sum(all_constraints == False)}")
    norm_obj = all_obj / max_objectives
    hv_obj = deepcopy(norm_obj)
    for i in range(num_objectives):
        if objective_min_max[i].lower() == 'max':
            hv_obj[:,i] = 1 - norm_obj[:,i]

    pareto_front_des, pareto_front_obj, hypervolumes = track_pareto_progress(
        hv_obj, all_des, all_constraints, hv_ind)
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


def get_models(device, params, num_total_objs):
    # Briefly instantiate problem to get input dimensions
    problem = TrussProblem()
    num_actions = problem.n_elements
    action_length = len(problem.area_map)

    # Actor input: action_length (sequence length)
    actor = Actor(device=device, params=params, action_length=action_length)
    # Critic output: total_dims (5) to match weights vector
    critic = Critic(device=device, params=params, num_objectives=num_total_objs, input_dim=num_actions+num_total_objs)

    actor.to(device)
    critic.to(device)
    
    # Initialize lazy layers
    input_dummy = torch.zeros((1, num_actions+num_total_objs), dtype=torch.float32).to(device)
    actor(input_dummy)
    critic(input_dummy)

    if params['model_folder'] is not None:
        actor.load_state_dict(torch.load(f"{params['model_folder']}/actor_model.pth"))
        critic.load_state_dict(torch.load(f"{params['model_folder']}/critic_model.pth"))
        print("Loaded pre-trained models.")

    return actor, critic


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def run_epoch(actor, critic, NFE, max_values, min_mask, device, params, eval_function_override=None,
              archive=None, to_hv=None):
    mini_batch_size = params['mini_batch_size']
    reward_mode = params.get('reward_mode', 'scalarized')
    hv_scale = params.get('hv_scale', 200.0)
    infeasible_penalty = params.get('infeasible_penalty', 0.0)
    if reward_mode == 'hv' and (archive is None or to_hv is None):
        raise ValueError("reward_mode='hv' needs an archive and a to_hv mapping")

    # Use override if provided, otherwise create default
    if eval_function_override is not None:
        eval_function = eval_function_override
    else:
        eval_function = TrussProblem()
    
    # Total output dimensions (Objectives + Constraints)
    num_total_objs = len(max_values) 
    max_objectives = max_values[:eval_function.num_objectives]
    
    # Extract dimensions
    num_actions = eval_function.n_elements
    
    rewards = [[] for _ in range(mini_batch_size)]
    actions = [[] for _ in range(mini_batch_size)]
    logprobs = [[] for _ in range(mini_batch_size)]
    designs = [[] for _ in range(mini_batch_size)]
    observation = [[] for _ in range(mini_batch_size)]
    
    epoch_des = []
    epoch_obj = []
    epoch_constraints = []
    epoch_constraint_vals = []
    objectives_raw = []

    # Generate Weights for all 5 dimensions
    weights_nonnorm = np.random.rand(mini_batch_size, num_total_objs)
    if params.get('pin_constraint_weights', True):
        weights_nonnorm[:, -2:] = 1.0  # Equal weights for constraint values
    weights = weights_nonnorm / weights_nonnorm.sum(axis=1, keepdims=True)

    for idx in range(mini_batch_size):
        observation[idx] = weights[idx].tolist()  # Initialize observation with weights

    # --- 1. Sample Actor ---
    with torch.no_grad():
        for i in range(num_actions):
            log_probs_out, sel_actions_out = actor.sample_action(observation)
            log_probs_out = log_probs_out.tolist()
            sel_actions_out = sel_actions_out.tolist()

            for idx, action in enumerate(sel_actions_out):
                actions[idx].append(action)
                logprobs[idx].append(log_probs_out[idx])
                observation[idx].append(action)
                rewards[idx].append(0.) 
                des_val = int(action)
                designs[idx].append(des_val)

    # --- 2. Evaluate & Calculate Rewards ---
    for idx, des in enumerate(designs):
        # Unpack explicitly including constraint_vals
        obj_values, constraints, constraint_vals = eval_function.evaluate(des)
        NFE += 1
        epoch_des.append(des)
        epoch_obj.append(obj_values)
        epoch_constraints.append(constraints)
        epoch_constraint_vals.append(constraint_vals)
        objectives_raw.append(obj_values)

        if reward_mode == 'hv':
            # Reward is the hypervolume this design adds to the running front. Infeasible
            # designs score 0 by default; infeasible_penalty grades them by violation.
            if constraints:
                step_reward = (-infeasible_penalty * float(np.sum(constraint_vals))
                               if infeasible_penalty else 0.0)
            else:
                step_reward = archive.offer(to_hv(obj_values))[1] * hv_scale
        else:
            obj_arr = np.array(obj_values + constraint_vals, dtype=float)
            max_obj_arr = np.array(max_values, dtype=float)

            if constraints:
                # For constrained designs: give -max_obj for minimization objectives, 0 for maximization ones
                reward_vec = np.where(min_mask, -np.ones(num_total_objs), 0.0)
                reward_vec[-2:] = np.array(constraint_vals)
                reward_vec[-2:][min_mask[-2:]] = -reward_vec[-2:][min_mask[-2:]]

            else:
                # For valid designs: negate only minimization objectives, keep maximization objectives positive (normalized)
                norm = obj_arr / max_obj_arr
                reward_vec = np.where(min_mask, -norm, norm)

            step_reward = float(np.dot(weights[idx], reward_vec))

        rewards[idx][-1] = rewards[idx][-1] + step_reward


    # --- 3. Sample Critic ---
    critic_values = []
    with torch.no_grad():
        for action_idx in range(num_actions):
            critic_observations = []
            for idx in range(mini_batch_size):
                critic_observations.append(observation[idx][:action_idx + 1 + num_total_objs])  # Include weights in observation
                
            crit_vals = critic.sample_critic(critic_observations)
            crit_vals = np.array(crit_vals.tolist())
            
            # Critic predicts vector of 5 values, weighted sum here
            critic_values.append(np.sum(np.multiply(weights, crit_vals), axis=1))

    # Reshape critic values
    values = [[] for _ in range(mini_batch_size)]
    for act_idx, act_vals in enumerate(critic_values):
        for batch_idx, val in enumerate(act_vals):
            values[batch_idx].append(val)

    for idx in range(mini_batch_size):
        values[idx].append(values[idx][-1]) 

    # --- 4. Calculate Advantages ---
    gamma = params['gamma']
    lam = params['lambda']
    all_advantages = [[] for _ in range(mini_batch_size)]
    all_returns = [[] for _ in range(mini_batch_size)]
    
    for idx in range(mini_batch_size):
        d_reward = np.array(rewards[idx])
        d_value = np.array(values[idx])
        
        deltas = d_reward + gamma * d_value[1:] - d_value[:-1]
        adv_tensor = discounted_cumulative_sums(deltas, gamma * lam)
        all_advantages[idx] = adv_tensor

        ret_tensor = discounted_cumulative_sums(d_reward, gamma * lam)
        all_returns[idx] = np.array(ret_tensor, dtype=np.float32)

    all_advantages_flat = np.concatenate(all_advantages)
    advantage_mean, advantage_std = np.mean(all_advantages_flat), np.std(all_advantages_flat)
    all_advantages = [(adv - advantage_mean) / (advantage_std + 1e-8) for adv in all_advantages]

    # --- 5. Prepare Tensors ---
    observation_tensor = []
    action_pos_tensor = []
    action_tensor = []
    logprob_tensor = []
    advantage_tensor = []
    return_tensor = []
    weights_tensor = []
    
    for batch_idx in range(mini_batch_size):
        obs = observation[batch_idx]
        for step in range(num_actions):
            obs_fragment = obs[:step+1+num_total_objs]  # Include weights in observation
            padded_obs = obs_fragment + [0] * (num_actions+num_total_objs - len(obs_fragment))
            
            observation_tensor.append(padded_obs)
            # The slot the actor read when it chose this action: it saw the weights plus
            # `step` earlier actions and took its distribution from the last position.
            action_pos_tensor.append(num_total_objs + step - 1)
            action_tensor.append(actions[batch_idx][step])
            logprob_tensor.append(logprobs[batch_idx][step])
            advantage_tensor.append(all_advantages[batch_idx][step])
            return_tensor.append(all_returns[batch_idx][step])
            weights_tensor.append(weights[batch_idx])

    observation_tensor = torch.tensor(observation_tensor, dtype=torch.float32).to(device)
    action_pos_tensor = torch.tensor(action_pos_tensor, dtype=torch.long).to(device)
    action_tensor = torch.tensor(action_tensor, dtype=torch.float32).to(device)
    logprob_tensor = torch.tensor(logprob_tensor, dtype=torch.float32).to(device)
    advantage_tensor = torch.tensor(advantage_tensor, dtype=torch.float32).to(device)
    return_tensor = torch.tensor(return_tensor, dtype=torch.float32).to(device)
    weights_tensor = torch.tensor(np.array(weights_tensor), dtype=torch.float32).to(device)

    # --- 6. Updates ---
    targetkl = params['target_kl']
    final_kl = 0
    final_actor_loss = 0
    
    for _ in range(params['update_iterations']):
        actor_loss, kl = actor.ppo_update(
            observation_tensor, 
            action_tensor, 
            logprob_tensor,
            advantage_tensor,
            action_pos_tensor
        )
        final_actor_loss = actor_loss
        final_kl = kl
        if kl > targetkl:
            break

    final_critic_loss = 0
    for _ in range(params['update_iterations']):
        final_critic_loss = critic.ppo_update(
            observation_tensor, 
            return_tensor, 
            weights_tensor
        )

    avg_obj_val = np.mean(objectives_raw, axis=0)

    epoch_data = {
        'des': epoch_des,
        'obj': epoch_obj,
        'constraints': epoch_constraints,
        'constraint_vals': epoch_constraint_vals
    }
    
    stats = {
        'actor_loss': final_actor_loss,
        'critic_loss': final_critic_loss,
        'kl': final_kl,
        'avg_obj': avg_obj_val
    }

    return actor, critic, NFE, epoch_data, stats