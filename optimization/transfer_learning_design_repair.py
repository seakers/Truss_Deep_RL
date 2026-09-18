# optimization/transfer_learning_design_repair.py

import numpy as np
import torch
import gc
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.indicators.hv import HV

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress

# Reuse from existing truss design_repair [3]
from optimization.design_repair import (
    get_models, run_repair_epoch, generate_initial_designs,
    sample_next_batch, plot_training_results
)
from optimization.transfer_learning_ppo_state import _estimate_max_values_truss


def run_transfer_learning_design_repair(max_values, params):
    """
    Transfer Learning for Design Repair PPO (Truss):
    Phase 1: Pre-train repair actor/critic on truss problems with varied loads
    Phase 2: Fine-tune on actual truss problem
    """
    print("\n\nRunning Transfer Learning (Design Repair)...\n\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_function = TrussProblem()
    num_objectives = eval_function.num_objectives
    objective_min_max = eval_function.objective_min_max
    num_actions = eval_function.n_elements
    action_range = len(eval_function.area_map)
    max_objectives = max_values[:num_objectives]
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])
    num_total_objs = len(max_values)

    epochs = params['num_epochs']
    mini_batch_size = params['mini_batch_size']
    date_str = params['date_str']

    transfer_load_factors = params.get('transfer_load_factors', [0.7, 0.85, 1.15, 1.3])

    # Budget split: 30% pre-train, 100% fine-tune. This results in more total NFE but gives a direct comparison of fine-tuning performance at the same epoch counts.
    # May adjust in the future, just pretrain epochs from finetune
    pretrain_fraction = params.get('transfer_pretrain_fraction', 0.3)
    total_nfe_budget = epochs * mini_batch_size
    pretrain_nfe_total = int(total_nfe_budget * pretrain_fraction)
    pretrain_nfe_per_set = max(mini_batch_size, pretrain_nfe_total // len(transfer_load_factors))
    finetune_nfe_budget = total_nfe_budget

    # ============================================================
    # Phase 1: Pre-training
    # ============================================================
    print(f"PHASE 1: Pre-training ({pretrain_nfe_per_set} NFE x "
          f"{len(transfer_load_factors)} load factors)")

    # Fresh model init (bypass checkpoint loading)
    original_model_folder = params.get('model_folder')
    params['model_folder'] = None
    actor, critic = get_models(device, params, num_total_objs)
    params['model_folder'] = original_model_folder

    for lf_idx, load_factor in enumerate(transfer_load_factors):
        print(f"\n--- Pre-training on load factor {load_factor} "
              f"({lf_idx + 1}/{len(transfer_load_factors)}) ---")

        transfer_problem = TrussProblem(load_factor=load_factor)
        transfer_max_values = _estimate_max_values_truss(transfer_problem, num_samples=500)
        transfer_max_objectives = transfer_max_values[:num_objectives]

        pt_NFE = 0
        pt_des, pt_obj, pt_con = [], [], []

        while pt_NFE < pretrain_nfe_per_set:
            if pt_NFE == 0:
                initial_designs, initial_objs, pt_des, pt_obj, pt_con, pt_NFE = \
                    generate_initial_designs(
                        pt_des, pt_obj, pt_con, transfer_problem,
                        mini_batch_size, num_actions, action_range, pt_NFE
                    )
            else:
                initial_designs, initial_objs, pt_des, pt_obj, pt_con, pt_NFE = \
                    sample_next_batch(
                        pt_des, pt_obj, pt_con, transfer_problem,
                        mini_batch_size, num_actions, action_range, pt_NFE
                    )

            epoch_data, stats, pt_des, pt_obj, pt_con, pt_NFE = run_repair_epoch(
                actor, critic, pt_des, pt_obj, pt_con,
                initial_designs, initial_objs, transfer_problem,
                mini_batch_size, pt_NFE,
                transfer_max_objectives, min_mask, device, params
            )

            if pt_NFE % (mini_batch_size * 50) < mini_batch_size:
                print(f"  Pre-train NFE: {pt_NFE}/{pretrain_nfe_per_set}")

        del transfer_problem, pt_des, pt_obj, pt_con
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save pre-trained checkpoint
    pretrain_dir = f"results/{date_str}/pretrained_models"
    os.makedirs(pretrain_dir, exist_ok=True)
    torch.save(actor.state_dict(), f'{pretrain_dir}/actor_transfer_repair.pth')
    torch.save(critic.state_dict(), f'{pretrain_dir}/critic_transfer_repair.pth')
    print(f"Pre-trained repair models saved to {pretrain_dir}/")

    # ============================================================
    # Phase 2: Fine-tuning on actual problem
    # ============================================================
    print(f"\nPHASE 2: Fine-tuning for ~{finetune_nfe_budget} NFE on actual problem")

    # Reset optimizers with lower LR
    finetune_lr = params.get('transfer_finetune_lr', params['learning_rate'] * 0.5)
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=finetune_lr)
    actor.scheduler = torch.optim.lr_scheduler.StepLR(actor.optimizer, step_size=1000, gamma=0.9)
    critic.optimizer = torch.optim.Adam(critic.parameters(), lr=finetune_lr)
    critic.scheduler = torch.optim.lr_scheduler.StepLR(critic.optimizer, step_size=1000, gamma=0.9)

    # Standard design repair loop — identical to run_design_repair_ppo [3]
    actual_problem = TrussProblem()
    NFE = 0
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    all_actor_loss, all_critic_loss, all_avg_reward, all_kl = [], [], [], []

    ref_point = np.ones(num_objectives)
    hv_ind = HV(ref_point=ref_point)

    while NFE < finetune_nfe_budget:
        epoch = NFE // mini_batch_size
        if epoch % 10 == 9:
            print(f"Fine-tune Epoch {epoch + 1}, NFE: {NFE}/{finetune_nfe_budget}")

        if NFE == 0:
            initial_designs, initial_objs, all_des, all_obj, all_constraints, NFE = \
                generate_initial_designs(
                    all_des, all_obj, all_constraints, actual_problem,
                    mini_batch_size, num_actions, action_range, NFE
                )
        else:
            initial_designs, initial_objs, all_des, all_obj, all_constraints, NFE = \
                sample_next_batch(
                    all_des, all_obj, all_constraints, actual_problem,
                    mini_batch_size, num_actions, action_range, NFE
                )

        epoch_data, stats, all_des, all_obj, all_constraints, NFE = run_repair_epoch(
            actor, critic, all_des, all_obj, all_constraints,
            initial_designs, initial_objs, actual_problem,
            mini_batch_size, NFE,
            max_objectives, min_mask, device, params
        )

        all_des.extend(epoch_data['des'])
        all_obj.extend(epoch_data['obj'])
        all_constraints.extend(epoch_data['constraints'])
        all_constraint_vals.extend(epoch_data['constraint_vals'])

        all_actor_loss.append(stats['actor_loss'])
        all_critic_loss.append(stats['critic_loss'])
        all_avg_reward.append(stats['avg_reward'])
        all_kl.append(stats['kl'])

    # ============================================================
    # Post-processing — identical to design_repair [3]
    # ============================================================
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)

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
        hv_obj, all_des, all_constraints, hv_ind
    )

    # Save fine-tuned models
    torch.save(actor.state_dict(), f'results/{date_str}/actor_transfer_repair_finetuned.pth')
    torch.save(critic.state_dict(), f'results/{date_str}/critic_transfer_repair_finetuned.pth')
    plot_training_results(all_actor_loss, all_critic_loss, all_kl, all_avg_reward, params)

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE