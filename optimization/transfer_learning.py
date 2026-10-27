"""Transfer Learning PPO for the 72-bar truss: pretrain across a sweep of load
cases (position x magnitude) carrying actor+critic weights forward, then solve
the FINAL task (== baseline config: node 16, load_factor 1.0) at the full
baseline NFE budget, conditioning on a load context vector.

Mirrors the spacecraft transfer orchestrator but tasks are described purely by
(load_node, load_factor) so no cached problem object is rebuilt externally.
"""
import os
import copy
import torch
import numpy as np

from optimization.design_synthesis import run_ppo_rand_weights_state
from optimization.design_repair import run_design_repair_ppo
from optimization.transfer_tasks import (
    generate_task_sweep, make_task_params, context_for_params, CONTEXT_DIM,
)

SYNTH_CKPTS  = ('actor_model.pth',        'critic_model.pth')          # synthesis defaults [3]
REPAIR_CKPTS = ('actor_repair_model.pth', 'critic_repair_model.pth')   # repair defaults [2]


def _finite_ckpt(folder, names):
    for n in names:
        path = f"{folder}/{n}"
        if not os.path.exists(path):
            return False
        sd = torch.load(path, map_location="cpu")
        if any(not torch.isfinite(t).all() for t in sd.values()):
            return False
    return True


def _run_transfer(kind, max_values, params, base_run_fn, ckpt_names):
    date_str = params['date_str']
    tasks = generate_task_sweep(params)

    pretrain_frac = params.get('transfer_pretrain_fraction', 0.3)
    base_epochs = params['num_epochs']
    n_pretrain = max(1, len(tasks) - 1)
    # Per-pretrain-task budget: spread the pretrain fraction over the sweep.
    pretrain_epochs = max(1, int((pretrain_frac * base_epochs) / n_pretrain))

    carry_folder = None            # None -> fresh models on task 0
    final_result = None

    for i, task in enumerate(tasks):
        tparams = make_task_params(params, task)
        tparams['context_dim']    = CONTEXT_DIM
        tparams['context_vector'] = context_for_params(tparams)
        tparams['model_folder']   = carry_folder        # warm-start from prev task

        if task['is_final']:
            # Final task == baseline config, FULL baseline budget for fair comparison.
            tparams['date_str']   = date_str            # artifacts in standard location
            tparams['num_epochs'] = base_epochs
            print(f"[Transfer {kind}] FINAL task "
                  f"(node {task['load_node']}, factor {task['load_factor']}, "
                  f"{base_epochs} epochs)")
            final_result = base_run_fn(max_values, tparams)
        else:
            tparams['date_str']   = f"{date_str}/transfer_{kind}/task_{i:02d}"
            tparams['num_epochs'] = pretrain_epochs
            os.makedirs(f"results/{tparams['date_str']}", exist_ok=True)
            print(f"[Transfer {kind}] pretrain {i+1}/{len(tasks)} "
                  f"node={task['load_node']} factor={task['load_factor']}")
            base_run_fn(max_values, tparams)

        # Carry weights forward only if the checkpoint is finite.
        if _finite_ckpt(f"results/{tparams['date_str']}", ckpt_names):
            carry_folder = f"results/{tparams['date_str']}"

    return final_result


def run_transfer_learning_ppo_state(max_values, params, run_idx=0, best_hv_so_far=-np.inf):
    print("\n\nRunning Transfer Learning Design Synthesis PPO...\n\n")
    return _run_transfer('synthesis', max_values, params,
                         run_ppo_rand_weights_state, SYNTH_CKPTS)


def run_transfer_learning_design_repair(max_values, params, run_idx=0, best_hv_so_far=-np.inf):
    print("\n\nRunning Transfer Learning Design Repair PPO...\n\n")
    return _run_transfer('repair', max_values, params,
                         run_design_repair_ppo, REPAIR_CKPTS)