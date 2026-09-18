"""Round 9, experiment IX -- deploy the SUPERVISED pointer as a mutation operator.

Experiment I showed that every physics-guided operator loses to blind mutation. But every
operator it tested was a FIXED rule -- "always edit the least-utilised bar", "always apply
fully-stressed sizing" -- and experiment II showed exactly why a fixed rule must lose: the
correct choice reverses sign partway through a search, so any fixed rule is wrong for a
large part of every run while random is never systematically wrong.

A learned pointer is not a fixed rule. It is a function of the design, so it can in
principle say "high utilisation" early and "low utilisation" late. Experiment I therefore
does not cover it, and the question is genuinely open.

Experiment VI produced such a pointer, and it is not a weak one: trained supervised on
brute-forced labels it picks the truly best bar 25.9% of the time against 1.4% for chance
and 7.8% for the best textbook heuristic, and sits at KL 0.64 from uniform where every
RL-trained pointer sits below 0.002.

So: put it in the GA and measure. Arms, all at 64,000 evaluations with the utilisation
solve counted as an NFE so the blind control pays the same price:

  random      -- blind control, uniform bar, uniform value
  supervised  -- bar from the supervised pointer (sampled from its distribution), uniform value
  sup_greedy  -- bar from the supervised pointer (argmax), uniform value

The greedy variant is included because experiment I measured that greedy determinism alone
costs 0.195 hypervolume by collapsing population diversity, independently of whether the
rule is right. Sampling is the fair comparison; argmax shows what insisting on the single
best bar costs.

If `supervised` beats `random`, the intelligent-mutation direction is alive and the fix is
supervised pretraining rather than reward engineering. If it does not, then even a
genuinely good state-dependent bar-chooser cannot pay for itself here, and the direction
is closed on stronger evidence than experiment I alone provides.
"""
import argparse, json, os, sys, time
from copy import deepcopy

import numpy as np
import pygad
import torch
from pymoo.indicators.hv import HV

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))

from evaluation.truss_problem import TrussProblem
from utils.hypervolume_utils import track_pareto_progress
from optimization.transformer_design_repair import Actor

MV = np.load(os.path.join(HERE, "max_values.npy"))
MIN_MASK = np.array([True, True, False, True, True])
NFE_BUDGET = 64000
POP = 64
CKPT = os.path.join(HERE, "data", "r9_supervised_pointer.pth")


def load_pointer():
    prob = TrussProblem()
    NV, AL = prob.n_elements, len(prob.area_map)
    dev = torch.device("cpu")
    P = dict(num_epochs=1, mini_batch_size=32, gamma=0.999, learning_rate=3e-4,
             clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
             model_folder=None, use_stress=True, normalise_design=True)
    P['lambda'] = 0.9
    a = Actor(device=dev, params=P, num_variables=NV, action_length=AL, num_objectives=5)
    a.to(dev)
    a(torch.zeros((1, NV, 2)))
    a.load_state_dict(torch.load(CKPT, map_location=dev))
    a.eval()
    return a


def run_arm(arm, seed):
    prob = TrussProblem()
    NV, AL = prob.n_elements, len(prob.area_map)
    MO = MV[:prob.num_objectives]
    ref = np.ones(prob.num_objectives)
    hv_ind = HV(ref_point=ref)
    actor = None if arm == 'random' else load_pointer()

    all_des, all_obj, all_con = [], [], []
    state = {'nfe': 0}

    def fit(ga_, sol, idx):
        d = [int(x) for x in sol]
        ov, isc, cv = prob.evaluate(d)
        all_des.append(d); all_obj.append(ov); all_con.append(isc)
        state['nfe'] += 1
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float)
            a[~MIN_MASK] = 0.0
            a[-2:] = cv
            a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]
            return a
        a = np.array(ov + cv, dtype=float)
        a[MIN_MASK] = -a[MIN_MASK]
        return a

    rng = np.random.default_rng(seed + 4242)

    def mutate(offspring, ga_):
        for i, d in enumerate(offspring):
            d = np.asarray(d, dtype=int).copy()
            # every arm pays the utilisation solve, so the comparison is of the RULE
            _, _, _, u = prob.evaluate([int(x) for x in d], return_utilisation=True)
            state['nfe'] += 1
            if arm == 'random':
                b = int(rng.integers(0, NV))
            else:
                st = torch.tensor(np.stack([d.astype(float),
                                            np.abs(np.asarray(u, dtype=float))], -1),
                                  dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _, _ = actor(st)
                p = torch.softmax(logits.float(), -1)[0].numpy()
                p = np.clip(p, 1e-12, None); p = p / p.sum()
                b = int(p.argmax()) if arm == 'sup_greedy' else int(rng.choice(NV, p=p))
            d[b] = int(rng.integers(0, AL))
            offspring[i] = d
        return offspring

    def on_gen(ga_):
        if state['nfe'] >= NFE_BUDGET:
            return "stop"

    ga = pygad.GA(num_generations=100000, num_parents_mating=POP // 2, fitness_func=fit,
                  sol_per_pop=POP, num_genes=NV, gene_space=[range(AL)] * NV,
                  parent_selection_type="tournament", K_tournament=3,
                  crossover_type="single_point", mutation_type=mutate,
                  keep_elitism=8, keep_parents=0, save_solutions=False,
                  on_generation=on_gen, random_seed=seed)
    ga.run()

    all_obj = np.array(all_obj); all_des = np.array(all_des); all_con = np.array(all_con)
    ref_values = MO * ref
    ref_values[~MIN_MASK[:prob.num_objectives]] = 0.0
    all_obj[all_con] = ref_values
    norm = all_obj / MO
    hv_obj = deepcopy(norm)
    for k in range(prob.num_objectives):
        if prob.objective_min_max[k].lower() == 'max':
            hv_obj[:, k] = 1 - norm[:, k]
    pfd, pfo, hvs = track_pareto_progress(hv_obj, all_des, all_con, hv_ind)
    final = pfo[-1] if len(pfo) and pfo[-1] is not None else []
    return dict(hv=float(hvs[-1]) if len(hvs) else 0.0, nfe=state['nfe'],
                front_size=len(final), valid_frac=float(np.mean(~all_con)))


ARMS = ['random', 'supervised', 'sup_greedy']
SEEDS = [0, 1, 2, 3, 4, 5]
JOBS = [(a, s) for a in ARMS for s in SEEDS]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"r9d_shard_{a.shard}.jsonl")
    for j, (arm, seed) in enumerate(JOBS):
        if j % a.nshards != a.shard:
            continue
        t0 = time.time()
        rec = run_arm(arm, seed)
        rec.update(job_id=j, name=arm, seed=seed, secs=time.time() - t0)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{j}] {arm} seed={seed} hv={rec['hv']:.4f} nfe={rec['nfe']} "
              f"{rec['secs']:.0f}s", flush=True)
