"""Round 9, experiment VI -- CAN this architecture learn bar selection at all?

Experiment IV killed the comfortable explanation. Choosing WHICH bar to edit accounts for
24.1% of the variance in mutation outcome, MORE than choosing what value to set it to
(16.2%). So the pointer head is not being asked to learn a distinction that does not
exist. Its collapse to uniform is a genuine failure, and two explanations remain:

  (a) REPRESENTATION -- the network, its inputs, or its capacity cannot express the
      mapping from design to best-bar, so no training signal would help.
  (b) CREDIT ASSIGNMENT -- the mapping is learnable, but PPO's single scalar advantage,
      shared between two heads and averaged over a state distribution where the right
      answer REVERSES SIGN with search stage (experiment II: corr(|util|, dPhi) goes
      +0.52 on random designs to -0.27 on converged ones), never delivers it.

Supervised learning separates them cleanly. Brute-force the label -- for each design,
score every one of the 72 bars by the average improvement obtained from editing it -- and
train the SAME pointer architecture on the SAME inputs with cross-entropy, where credit
assignment is trivial and the state distribution is fixed. If it learns, (a) is dead and
the problem is the RL. If it cannot, no amount of reward engineering was ever going to
work and the fix has to be architectural.

Two reference points come free: how often the textbook heuristics (edit the least- or
most-utilised bar) pick the truly best bar, and how often chance does (1/72 = 1.4%).
"""
import argparse, os, sys, json, time
import numpy as np, pygad, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))
from evaluation.truss_problem import TrussProblem
from optimization.design_repair import design_potential
from optimization.transformer_design_repair import Actor

MV = np.load(os.path.join(HERE, "max_values.npy"))
MO = MV[:3]
MIN_MASK = np.array([True, True, False, True, True])
prob = TrussProblem()
NV, AL = prob.n_elements, len(prob.area_map)
VALS = np.linspace(0, AL - 1, 5).astype(int)     # values probed per bar to score it


def evolve(seed, gens):
    def fit(g_, sol, idx):
        ov, isc, cv = prob.evaluate([int(x) for x in sol])
        if isc:
            a = -np.array(np.append(MO, cv), dtype=float); a[~MIN_MASK] = 0.0
            a[-2:] = cv; a[-2:][MIN_MASK[-2:]] = -a[-2:][MIN_MASK[-2:]]; return a
        a = np.array(ov + cv, dtype=float); a[MIN_MASK] = -a[MIN_MASK]; return a
    ga = pygad.GA(num_generations=gens, num_parents_mating=32, fitness_func=fit,
                  sol_per_pop=64, num_genes=NV, gene_space=[range(AL)] * NV,
                  parent_selection_type="tournament", K_tournament=3,
                  crossover_type="single_point", mutation_type="random",
                  mutation_num_genes=1, keep_elitism=8, keep_parents=0,
                  save_solutions=False, random_seed=seed)
    ga.run()
    return [[int(x) for x in s] for s in ga.population]


def build_data(shard, nshards):
    rng = np.random.default_rng(1000 + shard)
    out = []
    jobs = [(sd, g) for sd in range(4) for g in (0, 100, 400)]
    for j, (sd, gens) in enumerate(jobs):
        if j % nshards != shard:
            continue
        pop = evolve(sd, gens) if gens else [list(rng.integers(0, AL, NV)) for _ in range(64)]
        for d in pop[:48]:
            o, c, cv, util = prob.evaluate(d, return_utilisation=True)
            phi0 = design_potential(o, cv, MO, MIN_MASK)
            score = np.zeros(NV)
            for b in range(NV):
                acc = []
                for v in VALS:
                    d2 = list(d); d2[b] = int(v)
                    o2, _, cv2 = prob.evaluate(d2)
                    acc.append(design_potential(o2, cv2, MO, MIN_MASK) - phi0)
                score[b] = float(np.mean(acc))
            out.append(dict(stage=gens, seed=sd, design=[int(x) for x in d],
                            util=[float(x) for x in util], score=score.tolist()))
        print(f"stage={gens} seed={sd} done ({len(out)} designs)", flush=True)
    with open(os.path.join(HERE, "data", f"r9_super_data_{shard}.json"), "w") as f:
        json.dump(out, f)


def train():
    data = []
    for f in sorted(os.listdir(HERE)):
        if f.startswith("r9_super_data_") and f.endswith(".json"):
            data += json.load(open(os.path.join(HERE, "data", f)))
    print(f"{len(data)} labelled designs")

    X_d = np.array([r['design'] for r in data], dtype=np.float32)
    X_u = np.abs(np.array([r['util'] for r in data], dtype=np.float32))
    S = np.array([r['score'] for r in data], dtype=np.float32)
    Y = S.argmax(1)
    stage = np.array([r['stage'] for r in data])

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(data))
    ntr = int(0.8 * len(data))
    tr, te = perm[:ntr], perm[ntr:]

    # --- reference points -------------------------------------------------------
    print("\nhow often does each rule pick the truly best bar? (test split)")
    print(f"  chance (1/72)                  {1/NV:>7.1%}")
    for nm, pick in [("edit the LEAST utilised bar", lambda u: int(np.argmin(u))),
                     ("edit the MOST utilised bar",  lambda u: int(np.argmax(u)))]:
        acc = float(np.mean([pick(X_u[i]) == Y[i] for i in te]))
        int5 = float(np.mean([pick(X_u[i]) in np.argsort(-S[i])[:5] for i in te]))
        print(f"  {nm:<30} {acc:>7.1%}   (pick lands in true top-5: {int5:>5.1%})")

    # --- supervised pointer -----------------------------------------------------
    dev = torch.device("cpu")
    P = dict(num_epochs=1, mini_batch_size=32, gamma=0.999, learning_rate=3e-4,
             clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
             model_folder=None, use_stress=True, normalise_design=True)
    P['lambda'] = 0.9
    torch.manual_seed(0)
    actor = Actor(device=dev, params=P, num_variables=NV, action_length=AL, num_objectives=5)
    actor.to(dev)
    actor(torch.zeros((1, NV, 2)))

    Xt = torch.tensor(np.stack([X_d, X_u], -1))
    Yt = torch.tensor(Y, dtype=torch.long)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    lossf = torch.nn.CrossEntropyLoss()

    def evaluate(idx):
        actor.eval()
        with torch.no_grad():
            lg, _, _ = actor(Xt[idx])
            lg = lg.float()
            pred = lg.argmax(1).numpy()
            top1 = float(np.mean(pred == Y[idx]))
            top5 = float(np.mean([Y[i] in np.argsort(-lg[k].numpy())[:5]
                                  for k, i in enumerate(idx)]))
            # is the bar it picks actually a good one, even when not THE best?
            pct = float(np.mean([(S[i] < S[i][pred[k]]).mean() for k, i in enumerate(idx)]))
            p = torch.softmax(lg, -1).numpy()
            kl = float(np.mean(np.log(NV) + (p * np.log(p + 1e-12)).sum(1)))
        actor.train()
        return top1, top5, pct, kl

    print(f"\n{'epoch':>6} {'train loss':>11} {'test top-1':>11} {'test top-5':>11} "
          f"{'pct-rank of pick':>17} {'KL(ptr||unif)':>14}")
    print('-' * 76)
    bs = 64
    for ep in range(31):
        idx = rng.permutation(tr)
        tot = 0.0
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            lg, _, _ = actor(Xt[b])
            loss = lossf(lg.float(), Yt[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0:
            t1, t5, pct, kl = evaluate(te)
            print(f"{ep:>6} {tot/len(idx):>11.4f} {t1:>11.1%} {t5:>11.1%} "
                  f"{pct:>17.1%} {kl:>14.4f}", flush=True)

    print("\nper-stage test accuracy of the supervised pointer:")
    for st, lbl in [(0, 'random designs'), (100, 'after 100 gens'), (400, 'after 400 gens')]:
        sel = np.array([i for i in te if stage[i] == st])
        if len(sel):
            t1, t5, pct, kl = evaluate(sel)
            print(f"  {lbl:>16}: n={len(sel):>4}  top-1 {t1:>6.1%}  top-5 {t5:>6.1%}  "
                  f"pct-rank {pct:>6.1%}  KL {kl:.4f}")
    print("\nFor scale: the RL-trained pointers sit at KL 0.000-0.011 from uniform.")

    # Saved so the learned pointer can be DEPLOYED as a mutation operator. Experiment I
    # showed every FIXED physics rule loses to blind mutation, and experiment II explained
    # why: the right rule reverses partway through a search. A supervised pointer is not a
    # fixed rule -- it is a function of the design -- so it is not covered by that result
    # and has to be measured on its own (experiment IX).
    torch.save(actor.state_dict(), os.path.join(HERE, "data", "r9_supervised_pointer.pth"))
    print("saved supervised pointer -> r9_supervised_pointer.pth", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['data', 'train'], required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    if a.stage == 'data':
        build_data(a.shard, a.nshards)
    else:
        train()
