"""Round 9, experiment VIII -- does the RL get better when the problem is posed properly?

Experiment VII found the one change so far that helps: posing the 72-bar truss with its
classical 16 symmetry groups instead of 72 independent bars. That was measured on the GA.
This asks whether it rescues the reinforcement learning, which is the part the lab
actually cares about.

There is a specific reason to expect it might. The synthesis PPO builds a design
autoregressively, one variable at a time, and gets a single reward at the end. With 72
variables that is a 72-step episode whose entire return depends on the last token as much
as the first -- a credit-assignment problem that gets exponentially harder with length. In
group space the episode is 16 steps. Same policy, same reward, same budget, one sixth the
horizon.

So the comparison is the same three methods in both parameterisations:

  bars72    -- 72 independent variables, the repo's formulation
  groups16  -- 16 group variables expanded to 72 bars at evaluation time

Objectives, constraints, the solver and the hypervolume reference are all untouched --
only the number of knobs the optimiser turns. A method that improves a lot here was being
held back by the formulation rather than by anything about the method.
"""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
WORK = os.path.join(HERE, "r9gwork")
os.makedirs(os.path.join(WORK, "results"), exist_ok=True)

import evaluation.truss_problem as tp_mod
from evaluation.truss_problem import TrussProblem
from exp_symmetry import GROUP_OF_BAR, N_GROUPS

MV = np.load(os.path.join(HERE, "max_values.npy"))


class GroupedTrussProblem(TrussProblem):
    """The same physics, presented with 16 design variables instead of 72.

    Everything the optimiser sees -- the number of variables, the action space, the
    episode length -- shrinks to the 16 classical groups. Everything the SOLVER sees is
    unchanged: the design is expanded back to 72 bar areas before any analysis, so mass,
    displacement, frequency, the constraints and the hypervolume reference all keep
    exactly the meaning they have everywhere else in the repo. The comparison is between
    two parameterisations of one problem, not between two problems.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.full_n_elements = self.n_elements      # 72, what the solver needs
        self.n_elements = N_GROUPS                  # 16, what the optimiser sees

    def _expand(self, design):
        d = np.asarray(design, dtype=int)
        if d.size == self.full_n_elements:
            return d                                # already per-bar
        return d[GROUP_OF_BAR]

    def solve(self, discrete_actions):
        return super().solve(self._expand(discrete_actions))

    def evaluate(self, design, return_utilisation=False):
        out = super().evaluate(self._expand(design), return_utilisation=return_utilisation)
        if not return_utilisation:
            return out
        objectives, is_constrained, constraints, util = out
        # collapse per-bar utilisation to per-group, so a caller that asked for 16
        # variables is handed 16 numbers rather than 72
        util = np.asarray(util, dtype=float)
        gu = np.array([util[GROUP_OF_BAR == g].mean() for g in range(N_GROUPS)])
        return objectives, is_constrained, constraints, gu


ARMS = [
    ("PPO synthesis, 72 bars",   'state', False),
    ("PPO synthesis, 16 groups", 'state', True),
    ("GA tuned, 72 bars",        'ga',    False),
    ("GA tuned, 16 groups",      'ga',    True),
]
SEEDS = [0, 1, 2]
# Extra seeds run separately and merged. At 3 seeds the grouping effect on the PPO was
# +0.1065, 3/3, but p=0.111 -- the largest positive effect measured this round and the
# least well established. The PPO arms are cheap in group space (~510s) so more seeds are
# affordable, and an effect this size deserves them before it is reported.
EXTRA_SEEDS = [3, 4, 5, 6, 7]
JOBS = [(ai, s) for ai in range(len(ARMS)) for s in SEEDS]

BASE = dict(num_epochs=1000, mini_batch_size=64, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None, nfe_budget=64000,
            keep_elitism=8, parent_selection_type='tournament', K_tournament=3,
            mutation_num_genes=1)
BASE['lambda'] = 0.9


def run(kind, grouped, seed, tag):
    """Swap the problem class the target module builds, then call it unmodified.

    The runners construct TrussProblem() internally rather than taking one, so the
    parameterisation is switched by rebinding the name in the module under test. Nothing
    else about the method changes, which is the point of the experiment.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    p = dict(BASE)
    p['date_str'] = tag
    os.makedirs(os.path.join(WORK, "results", tag), exist_ok=True)

    cls = GroupedTrussProblem if grouped else TrussProblem
    if kind == 'state':
        import optimization.ppo_rand_weights_state as m
        p.update(reward_mode='hv', hv_reset_period=3)
        # only this module builds the problem; transformer_rand_weights_state does not
        # import TrussProblem at all, so there is nothing to rebind there
        old = m.TrussProblem
        m.TrussProblem = cls
        try:
            r = m.run_ppo_rand_weights_state(MV, p, run_idx=seed)
        finally:
            m.TrussProblem = old
        return r[:6]

    import optimization.genetic_algorithm as m
    old = m.TrussProblem
    m.TrussProblem = cls
    try:
        r = m.run_genetic_algorithm(MV, p)
    finally:
        m.TrussProblem = old
    return r[:6]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--extra', action='store_true')
    a = ap.parse_args()
    os.chdir(WORK)
    if a.extra:
        JOBS = [(ai, sd) for ai in range(len(ARMS)) for sd in EXTRA_SEEDS]
    out = os.path.join(HERE, "data", f"r9g{'2' if a.extra else ''}_shard_{a.shard}.jsonl")
    for j, (ai, seed) in enumerate(JOBS):
        if j % a.nshards != a.shard:
            continue
        label, kind, grouped = ARMS[ai]
        tag = f"g_{kind}_{'grp' if grouped else 'bar'}_s{seed}"
        t0 = time.time()
        des, obj, pfd, pfo, hvs, nfe = run(kind, grouped, seed, tag)
        rec = dict(job_id=j, name=label, kind=kind, grouped=grouped, seed=seed,
                   hv=float(hvs[-1]) if len(hvs) else 0.0, nfe=int(nfe),
                   front_size=int(len(pfo[-1])) if len(pfo) and pfo[-1] is not None else 0,
                   secs=time.time() - t0)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{j}] {label} seed={seed} hv={rec['hv']:.4f} nfe={rec['nfe']} "
              f"{rec['secs']:.0f}s", flush=True)
