"""
Task sweep + context-vector extraction for Truss Transfer Learning PPO.

A "task" = a (load_node, load_factor) pair. load_node is one of the four top
vertices; load_factor scales the fixed downward -5000 point load in
TrussProblem._generate_loads(). The context vector conditions the policy/critic
on the load CASE:
    - load position as a normalized one-hot over the 4 top nodes
    - load magnitude (load_factor), lightly scaled to ~O(1)

Unlike the spacecraft context (orbit + zero-padded payloads), there is exactly
one downward load, so no presence bits / padding are needed.
"""
import copy
import numpy as np

TOP_NODES = [16, 17, 18, 19]          # must match truss_problem.TOP_NODES
N_TOP = len(TOP_NODES)

POS_FEATURES = N_TOP                   # one-hot over the four candidate nodes
MAG_FEATURES = 1                       # load_factor
CONTEXT_DIM = POS_FEATURES + MAG_FEATURES    # = 5

_FACTOR_SCALE = 1.0                    # load_factor already ~O(1); keep stable across tasks


def build_context_vector(load_node, load_factor):
    """Fixed-length context: one-hot node position + scaled magnitude."""
    onehot = np.zeros(N_TOP, dtype=float)
    onehot[TOP_NODES.index(load_node)] = 1.0
    mag = np.array([load_factor / _FACTOR_SCALE], dtype=float)
    ctx = np.concatenate([onehot, mag]).astype(np.float32)
    ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert ctx.shape[0] == CONTEXT_DIM, (ctx.shape[0], CONTEXT_DIM)
    return ctx


def context_for_params(params):
    """Build the context vector for a given params dict."""
    return build_context_vector(params['load_node'], params['load_factor'])


def generate_task_sweep(base_params, seed=0):
    """Pretraining tasks (cross product of top nodes x transfer_load_factors)
    + 1 FINAL task that matches base_params (node 16, factor 1.0) so it is
    identical to what the baseline methods solve.
    """
    factors = base_params.get('transfer_load_factors', [0.7, 0.85, 1.15, 1.3])
    final_node = base_params.get('load_node', 16)
    final_factor = base_params.get('load_factor', 1.0)

    tasks = []
    for node in TOP_NODES:
        for f in factors:
            # skip an exact duplicate of the final/baseline task
            if node == final_node and f == final_factor:
                continue
            tasks.append({'load_node': node, 'load_factor': float(f),
                          'is_final': False})

    tasks.append({'load_node': final_node, 'load_factor': float(final_factor),
                  'is_final': True})
    return tasks


def make_task_params(base_params, task):
    """Clone base params, override load_node + load_factor for this task."""
    p = copy.deepcopy(base_params)
    p['load_node'] = int(task['load_node'])
    p['load_factor'] = float(task['load_factor'])
    return p