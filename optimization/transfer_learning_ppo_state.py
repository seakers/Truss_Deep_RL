# optimization/transfer_learning_ppo_state.py

import numpy as np
import torch
import gc
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.indicators.hv import HV

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress

# Reuse directly from existing module [11]
from optimization.ppo_rand_weights_state import run_epoch, get_models


def _estimate_max_values_truss(eval_function, num_samples=500):
    """Quick random search to estimate max values for normalization on a transfer problem."""
    num_actions = eval_function.n_elements
    action_range = len(eval_function.area_map)
    rng = np.random.default_rng(42)

    all_obj = []
    all_cv = []
    for _ in range(num_samples):
        design = rng.integers(0, action_range, num_actions).tolist()
        try:
            obj, is_con, cv = eval_function.evaluate(design)
            all_obj.append(obj)
            all_cv.append(cv)
        except Exception:
            continue

    if len(all_obj) == 0:
        raise RuntimeError("Could not evaluate any designs on transfer problem")

    all_obj = np.array(all_obj)
    all_cv = np.array(all_cv)
    max_values = np.max(np.concatenate((all_obj, all_cv), axis=1), axis=0) * 1.1 + 1e-6
    return max_values


def run_transfer_learning_ppo_state(max_values, params):
    """
    Transfer Learning for PPO Random Weights State (Truss):
    Phase 1: Pre-train actor/critic on truss problems with varied load factors
    Phase 2: Fine-tune on actual truss problem
    
    Reuses run_epoch and get_models from ppo_rand_weights_state [11].
    """
    print("\n\nRunning Transfer Learning (PPO State)...\n\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Actual problem setup
    eval_function = TrussProblem()
    num_objectives = eval_function.num_objectives
    objective_min_max = eval_function.objective_min_max
    max_objectives = max_values[:num_objectives]
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])
    num_total_objs = len(max_values)

    epochs = params['num_epochs']
    date_str = params['date_str']

    # Transfer learning load factors: varied versions of actual problem (1.0)
    transfer_load_factors = params.get('transfer_load_factors', [0.7, 0.85, 1.15, 1.3])

    # Budget split: 30% pre-train, 100% fine-tune. This results in more total NFE but gives a direct comparison of fine-tuning performance at the same epoch counts.
    # May adjust in the future, just pretrain epochs from finetune
    pretrain_fraction = params.get('transfer_pretrain_fraction', 0.3)
    pretrain_epochs_total = int(epochs * pretrain_fraction)
    pretrain_epochs_per_set = max(1, pretrain_epochs_total // len(transfer_load_factors))
    finetune_epochs = epochs

    # ============================================================
    # Phase 1: Pre-training on transfer problems (varied loads)
    # ============================================================
    print(f"PHASE 1: Pre-training ({pretrain_epochs_per_set} epochs x "
          f"{len(transfer_load_factors)} load factors = "
          f"{pretrain_epochs_per_set * len(transfer_load_factors)} total)")

    # Initialize fresh models (bypass checkpoint loading)
    original_model_folder = params.get('model_folder')
    params['model_folder'] = None
    actor, critic = get_models(device, params, num_total_objs)
    params['model_folder'] = original_model_folder

    for lf_idx, load_factor in enumerate(transfer_load_factors):
        print(f"\n--- Pre-training on load factor {load_factor} "
              f"({lf_idx + 1}/{len(transfer_load_factors)}) ---")

        # Create transfer problem with modified load
        transfer_problem = TrussProblem(load_factor=load_factor)

        # Estimate max_values for this transfer problem
        transfer_max_values = _estimate_max_values_truss(transfer_problem, num_samples=500)
        transfer_min_mask = min_mask  # Same objective structure

        pt_NFE = 0
        for epoch in range(pretrain_epochs_per_set):
            if epoch % 50 == 49:
                print(f"  Pre-train epoch {epoch + 1}/{pretrain_epochs_per_set}")

            # Pass transfer_problem as override so run_epoch uses it
            # instead of creating a default TrussProblem()
            actor, critic, pt_NFE, _, _ = run_epoch(
                actor, critic, pt_NFE, transfer_max_values,
                transfer_min_mask, device, params,
                eval_function_override=transfer_problem
            )

        # Free transfer problem memory
        del transfer_problem
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save pre-trained checkpoint
    pretrain_dir = f"results/{date_str}/pretrained_models"
    os.makedirs(pretrain_dir, exist_ok=True)
    torch.save(actor.state_dict(), f'{pretrain_dir}/actor_transfer_state.pth')
    torch.save(critic.state_dict(), f'{pretrain_dir}/critic_transfer_state.pth')
    print(f"Pre-trained models saved to {pretrain_dir}/")

    # ============================================================
    # Phase 2: Fine-tuning on actual problem
    # ============================================================
    print(f"\nPHASE 2: Fine-tuning for {finetune_epochs} epochs on actual problem")

    # Reset optimizers with lower LR for fine-tuning
    finetune_lr = params.get('transfer_finetune_lr', params['learning_rate'] * 0.5)
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=finetune_lr)
    actor.scheduler = torch.optim.lr_scheduler.StepLR(
        actor.optimizer, step_size=1000, gamma=0.9
    )
    critic.optimizer = torch.optim.Adam(critic.parameters(), lr=finetune_lr)
    critic.scheduler = torch.optim.lr_scheduler.StepLR(
        critic.optimizer, step_size=1000, gamma=0.9
    )

    # Standard training loop — identical to ppo_rand_weights_state [11]
    NFE = 0
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    all_actor_loss, all_critic_loss, all_avg_obj, all_kl = [], [], [], []
    ref_point = np.ones(num_objectives)
    hv_ind = HV(ref_point=ref_point)

    for epoch in range(finetune_epochs):
        if epoch % 10 == 9:
            print(f"Fine-tune epoch {epoch + 1}/{finetune_epochs}")

        # No override = uses default TrussProblem (load_factor=1.0)
        actor, critic, NFE, epoch_data, stats = run_epoch(
            actor, critic, NFE, max_values, min_mask, device, params
        )

        all_des.extend(epoch_data['des'])
        all_obj.extend(epoch_data['obj'])
        all_constraints.extend(epoch_data['constraints'])
        all_constraint_vals.extend(epoch_data['constraint_vals'])

        all_actor_loss.append(stats['actor_loss'])
        all_critic_loss.append(stats['critic_loss'])
        all_avg_obj.append(stats['avg_obj'])
        all_kl.append(stats['kl'])

    # ============================================================
    # Post-processing — identical to ppo_rand_weights_state [11]
    # ============================================================
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_constraint_vals = np.array(all_constraint_vals)

    ref_values = max_objectives * ref_point
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values
    print(f"Valid designs: {np.sum(~all_constraints)}")

    norm_obj = all_obj / max_objectives
    hv_obj = deepcopy(norm_obj)
    for i in range(num_objectives):
        if objective_min_max[i].lower() == 'max':
            hv_obj[:, i] = 1 - norm_obj[:, i]

    pareto_front_des, pareto_front_obj, hypervolumes = track_pareto_progress(
        hv_obj, all_des, all_constraints, hv_ind)

    # Save fine-tuned models
    torch.save(actor.state_dict(), f'results/{date_str}/actor_transfer_state_finetuned.pth')
    torch.save(critic.state_dict(), f'results/{date_str}/critic_transfer_state_finetuned.pth')

    # Training plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.plot(all_actor_loss); plt.title('Transfer State - Actor Loss'); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(all_critic_loss); plt.title('Transfer State - Critic Loss'); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(all_kl); plt.title('Transfer State - KL'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{date_str}/transfer_state_training.png')
    plt.close()

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE