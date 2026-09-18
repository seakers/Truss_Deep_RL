"""
Goal-conditioned synthesis PPO.

Instead of conditioning the policy on a random preference weighting like
ppo_rand_weights_state.py, this conditions it on a destination in objective space. A
coarse occupancy grid tracks which regions have been reached, and each episode is aimed
at a frontier cell: one that is empty but adjacent to a reached one. Frontier cells
rather than arbitrary empty ones, so the policy is not asked to reach regions no truss
can physically occupy.

Each minibatch gets a different destination per row, which spreads the front by
construction. Related to Go-Explore, MAP-Elites and Pareto Conditioned Networks.
"""
import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
from itertools import product
from pymoo.indicators.hv import HV
from copy import deepcopy

from optimization.transformer_rand_weights_state import Actor, Critic, _amp_ctx
from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress, RunningParetoArchive

COND_DIM = 5          # [target_0, target_1, target_2, coverage, archive_hv]


class ObjectiveGrid:
    """Coarse occupancy map over normalised objective space (the behaviour descriptor)."""

    def __init__(self, resolution=3, ndim=3):
        self.res = resolution
        self.ndim = ndim
        self.occ = np.zeros((resolution,) * ndim, dtype=bool)

    def cell_of(self, point):
        idx = (np.asarray(point, dtype=float) * self.res).astype(int)
        return tuple(np.clip(idx, 0, self.res - 1))

    def mark(self, point):
        self.occ[self.cell_of(point)] = True

    def coverage(self):
        return float(self.occ.mean())

    def frontier_cells(self):
        """Empty cells adjacent to a reached one; everything if nothing is reached yet."""
        occupied = list(zip(*np.nonzero(self.occ)))
        if not occupied:
            return list(np.ndindex(self.occ.shape))
        out = [c for c in np.ndindex(self.occ.shape)
               if not self.occ[c]
               and any(max(abs(a - b) for a, b in zip(c, o)) <= 1 for o in occupied)]
        if out:
            return out
        empty = [c for c in np.ndindex(self.occ.shape) if not self.occ[c]]
        return empty or list(np.ndindex(self.occ.shape))   # grid full: aim anywhere

    def sample_target(self):
        cells = self.frontier_cells()
        cell = cells[np.random.randint(len(cells))]
        return (np.asarray(cell, dtype=float) + 0.5) / self.res      # cell centre


class ScalarCritic(Critic):
    """Critic with a plain scalar value head.

    The random-weight method reads its value out as ``sum(-v * weights)``, which only makes
    sense when the conditioning vector is a weighting. Here it is a destination, so the
    value is predicted directly.
    """

    def __init__(self, device, params, input_dim):
        super().__init__(device, params, num_objectives=1, input_dim=input_dim)

    def ppo_update(self, observation, return_tensor, weights_tensor=None):
        self.optimizer.zero_grad()
        with _amp_ctx(self):
            value_loss = torch.mean((self(observation).squeeze(-1) - return_tensor) ** 2)

        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return value_loss.item()


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def get_models(device, params):
    problem = TrussProblem()
    num_actions = problem.n_elements
    action_length = len(problem.area_map)

    actor = Actor(device=device, params=params, action_length=action_length)
    critic = ScalarCritic(device=device, params=params, input_dim=num_actions + COND_DIM)
    actor.to(device)
    critic.to(device)

    dummy = torch.zeros((1, num_actions + COND_DIM), dtype=torch.float32).to(device)
    actor(dummy)
    critic(dummy)

    if params.get('model_folder') is not None:
        actor.load_state_dict(torch.load(f"{params['model_folder']}/actor_model.pth"))
        critic.load_state_dict(torch.load(f"{params['model_folder']}/critic_model.pth"))
        print("Loaded pre-trained models.")

    return actor, critic


def run_ppo_goal_conditioned(max_values, params, run_idx=0, best_hv_so_far=-np.inf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = params['num_epochs']
    date_str = params['date_str']

    problem = TrussProblem()
    num_objectives = problem.num_objectives
    objective_min_max = problem.objective_min_max
    max_objectives = np.asarray(max_values[:num_objectives], dtype=float)
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])

    actor, critic = get_models(device, params)
    hv_ind = HV(ref_point=np.ones(num_objectives))
    grid = ObjectiveGrid(resolution=params.get('grid_resolution', 3), ndim=num_objectives)
    archive = RunningParetoArchive(hv_ind)

    def to_descriptor(obj_values):
        """Objective vector -> point in [0,1]^3, all axes minimised."""
        n = np.asarray(obj_values, dtype=float) / max_objectives
        return np.clip(np.where(min_mask[:num_objectives], n, 1.0 - n), 0.0, 1.0)

    NFE = 0
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    all_actor_loss, all_critic_loss, all_kl, all_coverage = [], [], [], []

    for epoch in range(epochs):
        if epoch % 10 == 9:
            print(f"Epoch {epoch + 1}/{epochs}")

        actor, critic, NFE, epoch_data, stats = run_epoch(
            actor, critic, NFE, grid, archive, to_descriptor, device, params
        )

        all_des.extend(epoch_data['des'])
        all_obj.extend(epoch_data['obj'])
        all_constraints.extend(epoch_data['constraints'])
        all_constraint_vals.extend(epoch_data['constraint_vals'])

        all_actor_loss.append(stats['actor_loss'])
        all_critic_loss.append(stats['critic_loss'])
        all_kl.append(stats['kl'])
        all_coverage.append(stats['coverage'])

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_constraint_vals = np.array(all_constraint_vals)

    ref_values = max_objectives * np.ones(num_objectives)
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values
    print(f"Number of valid designs (False in all_constraints): {np.sum(all_constraints == False)}")
    print(f"Objective-space cells reached: {grid.occ.sum()}/{grid.occ.size}")

    norm_obj = all_obj / max_objectives
    hv_obj = deepcopy(norm_obj)
    for i in range(num_objectives):
        if objective_min_max[i].lower() == 'max':
            hv_obj[:, i] = 1 - norm_obj[:, i]

    pareto_front_des, pareto_front_obj, hypervolumes = track_pareto_progress(
        hv_obj, all_des, all_constraints, hv_ind)
    run_best_hv = hypervolumes[-1] if hypervolumes else 0.0

    torch.save(actor.state_dict(), f'results/{date_str}/actor_model.pth')
    torch.save(critic.state_dict(), f'results/{date_str}/critic_model.pth')

    print(f"  [Goal-Conditioned PPO] Run HV = {run_best_hv:.6f} | "
          f"Best so far = {best_hv_so_far:.6f}")

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1); plt.plot(all_actor_loss, color='blue'); plt.title('Actor Loss')
    plt.subplot(1, 4, 2); plt.plot(all_critic_loss, color='orange'); plt.title('Critic Loss')
    plt.subplot(1, 4, 3); plt.plot(all_kl, color='green'); plt.title('KL Divergence')
    plt.subplot(1, 4, 4); plt.plot(all_coverage, color='brown'); plt.title('Grid Coverage')
    plt.tight_layout()
    plt.savefig(f'results/{date_str}/goal_conditioned_training_run_{run_idx}.png')
    plt.close()

    return (all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE,
            run_best_hv)


def run_epoch(actor, critic, NFE, grid, archive, to_descriptor, device, params,
              eval_function_override=None):
    mini_batch_size = params['mini_batch_size']
    eval_function = eval_function_override or TrussProblem()
    num_actions = eval_function.n_elements

    goal_coef = params.get('goal_coef', 1.0)        # dense: did you go where you were sent
    front_bonus = params.get('front_bonus', 1.0)    # sparse: did you advance the front
    hv_scale = params.get('hv_scale', 0.0)          # optional raw hypervolume contribution

    rewards = [[] for _ in range(mini_batch_size)]
    actions = [[] for _ in range(mini_batch_size)]
    logprobs = [[] for _ in range(mini_batch_size)]
    designs = [[] for _ in range(mini_batch_size)]
    observation = [[] for _ in range(mini_batch_size)]

    epoch_des, epoch_obj, epoch_constraints, epoch_constraint_vals = [], [], [], []

    # one destination per row: 64 rows, 64 different targets, 64 different trusses
    targets = np.array([grid.sample_target() for _ in range(mini_batch_size)])
    coverage = grid.coverage()
    cond = np.concatenate(
        [targets, np.full((mini_batch_size, 1), coverage),
         np.full((mini_batch_size, 1), archive.hv)], axis=1)

    for idx in range(mini_batch_size):
        observation[idx] = cond[idx].tolist()

    with torch.no_grad():
        for _ in range(num_actions):
            log_probs_out, sel_actions_out = actor.sample_action(observation)
            log_probs_out = log_probs_out.tolist()
            sel_actions_out = sel_actions_out.tolist()

            for idx, action in enumerate(sel_actions_out):
                actions[idx].append(action)
                logprobs[idx].append(log_probs_out[idx])
                observation[idx].append(action)
                rewards[idx].append(0.)
                designs[idx].append(int(action))

    max_dist = float(np.sqrt(eval_function.num_objectives))     # diagonal of the unit cube
    for idx, des in enumerate(designs):
        obj_values, constraints, constraint_vals = eval_function.evaluate(des)
        NFE += 1
        epoch_des.append(des)
        epoch_obj.append(obj_values)
        epoch_constraints.append(constraints)
        epoch_constraint_vals.append(constraint_vals)

        descriptor = to_descriptor(obj_values)

        # Dense term, paid for feasible and infeasible designs alike: an infeasible truss
        # still has a mass and a frequency, so distance to target is meaningful there.
        goal_bonus = 1.0 - min(1.0, float(np.linalg.norm(descriptor - targets[idx])) / max_dist)
        step_reward = goal_coef * goal_bonus

        # Sparse term: only a feasible design can advance the front, and only advancing it
        # counts. Unlike the raw hypervolume delta (which decays to ~1e-8 once the front
        # matures) this keeps a usable magnitude.
        if not constraints:
            joined, delta_hv = archive.offer(descriptor)
            grid.mark(descriptor)
            if joined:
                step_reward += front_bonus
            step_reward += hv_scale * delta_hv

        rewards[idx][-1] = rewards[idx][-1] + step_reward

    critic_values = []
    with torch.no_grad():
        for action_idx in range(num_actions):
            critic_observations = [observation[idx][:action_idx + 1 + COND_DIM]
                                   for idx in range(mini_batch_size)]
            crit_vals = critic.sample_critic(critic_observations)
            critic_values.append(np.array(crit_vals.float().tolist()).reshape(-1))

    values = [[critic_values[k][i] for k in range(num_actions)] for i in range(mini_batch_size)]
    for idx in range(mini_batch_size):
        values[idx].append(values[idx][-1])

    gamma = params['gamma']
    lam = params['lambda']
    all_advantages, all_returns = [], []
    for idx in range(mini_batch_size):
        d_reward = np.array(rewards[idx])
        d_value = np.array(values[idx])
        deltas = d_reward + gamma * d_value[1:] - d_value[:-1]
        all_advantages.append(discounted_cumulative_sums(deltas, gamma * lam))
        all_returns.append(np.array(discounted_cumulative_sums(d_reward, gamma * lam),
                                    dtype=np.float32))

    flat = np.concatenate(all_advantages)
    adv_mean, adv_std = np.mean(flat), np.std(flat)
    all_advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in all_advantages]

    observation_tensor, action_pos_tensor = [], []
    action_tensor, logprob_tensor, advantage_tensor, return_tensor = [], [], [], []

    for batch_idx in range(mini_batch_size):
        obs = observation[batch_idx]
        for step in range(num_actions):
            fragment = obs[:step + 1 + COND_DIM]
            observation_tensor.append(fragment + [0] * (num_actions + COND_DIM - len(fragment)))
            # the slot the actor read when it chose this action
            action_pos_tensor.append(COND_DIM + step - 1)
            action_tensor.append(actions[batch_idx][step])
            logprob_tensor.append(logprobs[batch_idx][step])
            advantage_tensor.append(all_advantages[batch_idx][step])
            return_tensor.append(all_returns[batch_idx][step])

    observation_tensor = torch.tensor(observation_tensor, dtype=torch.float32).to(device)
    action_pos_tensor = torch.tensor(action_pos_tensor, dtype=torch.long).to(device)
    action_tensor = torch.tensor(action_tensor, dtype=torch.float32).to(device)
    logprob_tensor = torch.tensor(logprob_tensor, dtype=torch.float32).to(device)
    advantage_tensor = torch.tensor(advantage_tensor, dtype=torch.float32).to(device)
    return_tensor = torch.tensor(return_tensor, dtype=torch.float32).to(device)

    final_actor_loss, final_kl = 0, 0
    for _ in range(params['update_iterations']):
        final_actor_loss, final_kl = actor.ppo_update(
            observation_tensor, action_tensor, logprob_tensor, advantage_tensor,
            action_pos_tensor
        )
        if final_kl > params['target_kl']:
            break

    final_critic_loss = 0
    for _ in range(params['update_iterations']):
        final_critic_loss = critic.ppo_update(observation_tensor, return_tensor)

    epoch_data = {
        'des': epoch_des,
        'obj': epoch_obj,
        'constraints': epoch_constraints,
        'constraint_vals': epoch_constraint_vals,
    }
    stats = {
        'actor_loss': final_actor_loss,
        'critic_loss': final_critic_loss,
        'kl': final_kl,
        'coverage': grid.coverage(),
    }

    return actor, critic, NFE, epoch_data, stats
