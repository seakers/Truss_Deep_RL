# test_overlap.py
import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from evaluation.truss_problem import TrussProblem


def random_design(problem):
    return [int(np.random.choice(v['range'])) for v in problem.design_space]


def test_pairs_and_decode():
    p = TrussProblem(overlap_constraint=True)
    print(f"Detected {len(p.overlap_pairs)} overlapping pairs:")
    for (i, j) in p.overlap_pairs:
        print(f"  elem {i} {tuple(p.elements[i])}  <->  elem {j} {tuple(p.elements[j])}")

    print(f"design_space length: {len(p.design_space)} "
          f"(default has {TrussProblem().n_elements})")

    # Structural check: no pair ever has both members with nonzero area.
    for _ in range(500):
        area_idx = p._decode(random_design(p))
        for (i, j) in p.overlap_pairs:
            assert not (area_idx[i] > 0 and area_idx[j] > 0), \
                f"Both members active in pair ({i}, {j})"
    print("Decode check passed: no overlapping pair ever has both members active.")
    return p


def render_pairs(p, save_name="overlap_check"):
    """Plot undeformed geometry; overlapping-pair members in red, rest gray."""
    pair_elems = {e for pair in p.overlap_pairs for e in pair}
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    for idx, (n1, n2) in enumerate(p.elements):
        x = [p.nodes[n1][0], p.nodes[n2][0]]
        y = [p.nodes[n1][1], p.nodes[n2][1]]
        z = [p.nodes[n1][2], p.nodes[n2][2]]
        if idx in pair_elems:
            ax.plot(x, y, z, color='red', linewidth=2.0, alpha=0.9)
        else:
            ax.plot(x, y, z, color='lightgray', linewidth=1.0, alpha=0.6)
    ax.scatter(p.nodes[:, 0], p.nodes[:, 1], p.nodes[:, 2], c='black', s=15)
    ax.set_title(f'Overlapping-pair members (red): {len(p.overlap_pairs)} pairs')
    plt.tight_layout()
    plt.savefig(save_name + ".png", dpi=200)
    plt.close()
    print(f"Saved {save_name}.png")


def test_actor_heads(p, params):
    import torch
    from optimization.ppo_architectures import Actor, RepairActor
    device = torch.device('cpu')

    actor = Actor(device, params, p.design_space, p.unique_design_space, p.num_objectives)
    assert len(actor.heads) == len(p.unique_design_space)
    for h, var in zip(actor.heads, p.unique_design_space):
        print(f"  synthesis head '{var['name']}' -> {h.out_features} logits")

    w = np.array([0.4, 0.3, 0.3])
    genes, log_prob, _ = actor.sample_action(w, p)
    assert len(genes) == len(p.design_space)
    # every sampled gene must be legal for its own variable
    for i, v in enumerate(p.design_space):
        assert int(genes[i]) in list(v['range']), f"gene {i} illegal: {genes[i]}"

    repair = RepairActor(device, params, p.design_space, p.unique_design_space,
                         p.num_objectives)
    sel_head = repair.value_heads[repair.name_to_head['Sel']]
    assert sel_head.out_features == 2, sel_head.out_features
    print(f"  repair 'Sel' value head -> {sel_head.out_features} logits (expected 2)")

    # pointer landing on a selector gene must yield a 2-way value choice
    enc = [0.0] * len(p.design_space)
    for _ in range(200):
        vi, new_value, stop, lp_vi, lp_val, lp_stop, info = repair.sample_action(
            torch.tensor(enc, dtype=torch.float32))
        assert new_value in list(p.design_space[vi]['range'])
    print("  repair sampling passed: values always legal for the pointed gene.")

if __name__ == "__main__":

    params = {
        'num_epochs': 10,
        'mini_batch_size': 4,
        'gamma': 0.999,
        'lambda': 0.9,
        'learning_rate': 0.0003,
        'clip_ratio': 0.05,
        'update_iterations': 5,
        'target_kl': 0.003,
        'chunk_size': 256,
        'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'model_folder': None,
        # Transfer learning params
        'transfer_pretrain_fraction': 0.3,
        'transfer_finetune_lr': 0.00015,
        'transfer_load_factors': [0.7, 0.85, 1.15, 1.3],
        # Set to None to run everything from scratch.
        # Set to a results folder string to load max_values and actor models from it.
        'pretrained_artifacts_path': 'results/2026-06-15_10-09-18',
        'overlap_constraint': True,  # Truss-specific
    }

    p = test_pairs_and_decode()
    render_pairs(p)
    test_actor_heads(p, params)