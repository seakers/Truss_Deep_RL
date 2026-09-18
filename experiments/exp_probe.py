"""Round 9, experiment III -- one consistent read of what every trained operator learned.

Rounds 5-8 each measured pointer statistics with a slightly different probe. This
re-measures all 18 trained actors (round 7's 2x2 and round 8's fixed rerun) plus matched
untrained controls, on the same designs, with the same code, using exactly the state the
GA deployment path builds. Four numbers per actor:

  KL(pointer || uniform)  -- did it learn WHICH bar to edit at all? 0 = no preference.
  corr(|util|, p)         -- is whatever preference it has aligned with the physics?
  P(stop)                 -- has the stop head collapsed to "change nothing"?
  KL(value || uniform)    -- did the other head learn anything, as a comparison?

Measured on evolved designs, because that is where the operator is deployed. Untrained
controls give the null: a randomly initialised network is not exactly uniform either, so
"trained KL is small" only means something next to "untrained KL is this small too".
"""
import os, sys, json
import numpy as np, torch, pygad
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))
from evaluation.truss_problem import TrussProblem
from optimization.transformer_design_repair import Actor

MV = np.load(os.path.join(HERE, "max_values.npy"))
MO = MV[:3]
MIN_MASK = np.array([True, True, False, True, True])
prob = TrussProblem()
NV, AL = prob.n_elements, len(prob.area_map)
DEV = torch.device("cpu")

BASE = dict(num_epochs=1, mini_batch_size=32, gamma=0.999, learning_rate=3e-4,
            clip_ratio=0.05, update_iterations=5, target_kl=0.003, chunk_size=256,
            model_folder=None, max_edits=50)
BASE['lambda'] = 0.9

# (tag prefix, work dir, actor params) -- params must match how the arm was TRAINED
ARMS = [
    ("A_baseline",       "r7work", {}),
    ("B_stress",         "r7work", dict(use_stress=True)),
    ("C_shaping",        "r7work", {}),
    ("D_stress_shaping", "r7work", dict(use_stress=True)),
    ("E_normdesign",     "r8work", dict(normalise_design=True)),
    ("F_stress",         "r8work", dict(use_stress=True)),
]


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
    return [[int(x) for x in s] for s in ga.population[:32]]


# TWO design distributions, because round 8 and round 9 disagreed about the sign of
# corr(|util|, p) and the disagreement turns out to be the distribution, not a bug.
# Round 8's probe drew UNIFORMLY RANDOM designs; the operator is deployed on EVOLVED
# populations. Experiment II measured that the best bar to edit reverses sign between
# those two regimes, so a pointer can legitimately look physics-aligned on one and
# anti-aligned on the other. Measuring both settles it instead of picking one.
print("building probe designs ...", flush=True)
_rng = np.random.default_rng(0)
SETS = {
    'random designs':  [list(_rng.integers(0, AL, NV)) for _ in range(32)],
    'evolved designs': evolve(0, 200),
}
DESIGNS, UTILS = None, None


def use_set(name):
    global DESIGNS, UTILS
    DESIGNS = SETS[name]
    UTILS = []
    for d in DESIGNS:
        _, _, _, u = prob.evaluate(d, return_utilisation=True)
        UTILS.append(np.abs(np.asarray(u, dtype=float)))


def build(params, ckpt):
    p = dict(BASE); p.update(params)
    a = Actor(device=DEV, params=p, num_variables=NV, action_length=AL, num_objectives=5)
    a.to(DEV)
    dummy = (torch.zeros((1, NV, 2)) if p.get('use_stress') else torch.zeros((1, NV)))
    a(dummy)
    if ckpt:
        a.load_state_dict(torch.load(ckpt, map_location=DEV))
    a.eval()
    return a, p


def probe(actor, p):
    kls, corrs, pstops, vkls = [], [], [], []
    with torch.no_grad():
        for d, u in zip(DESIGNS, UTILS):
            if p.get('use_stress'):
                st = torch.tensor(np.stack([np.asarray(d, float), u], -1), dtype=torch.float32)
            else:
                st = torch.tensor(np.asarray(d, float), dtype=torch.float32)
            pl, enc, sl = actor(st.unsqueeze(0))
            pr = torch.softmax(pl.float(), -1).squeeze(0).numpy()
            kls.append(float(np.log(NV) + (pr * np.log(pr + 1e-12)).sum()))
            if np.std(u) > 1e-12 and np.std(pr) > 1e-12:
                corrs.append(float(np.corrcoef(u, pr)[0, 1]))
            pstops.append(float(torch.softmax(sl.float(), -1)[0, 1]))
            vi = torch.tensor([int(np.argmax(pr))])
            vl = actor.get_value_logits(enc, vi)
            vp = torch.softmax(vl.float(), -1).squeeze(0).numpy()
            vkls.append(float(np.log(AL) + (vp * np.log(vp + 1e-12)).sum()))
    return (np.mean(kls), np.mean(corrs) if corrs else np.nan,
            np.mean(pstops), np.mean(vkls))


rows = []
for SETNAME in SETS:
    use_set(SETNAME)
    print(f"\n\n########## measured on {SETNAME.upper()} ##########")
    print(f"\n{'arm':>20} {'seed':>5} {'KL(ptr||unif)':>14} {'corr(|util|,p)':>15} "
          f"{'P(stop)':>9} {'KL(val||unif)':>14}")
    print('-' * 84, flush=True)
    for name, work, params in ARMS:
        # matched untrained control, same architecture, no checkpoint
        torch.manual_seed(0)
        a, p = build(params, None)
        k, c, s, vk = probe(a, p)
        rows.append(dict(arm=name, seed=-1, trained=False, kl=k, corr=c, pstop=s,
                         vkl=vk, dset=SETNAME))
        print(f"{name + ' (untrained)':>20} {'--':>5} {k:>14.4f} {c:>+15.4f} "
              f"{s:>9.3f} {vk:>14.4f}")
        for t in range(3):
            ck = os.path.join(HERE, work, "results", f"{name}_t{t}",
                              "actor_repair_model.pth")
            if not os.path.isfile(ck):
                continue
            a, p = build(params, ck)
            k, c, s, vk = probe(a, p)
            rows.append(dict(arm=name, seed=t, trained=True, kl=k, corr=c, pstop=s,
                             vkl=vk, dset=SETNAME))
            print(f"{name:>20} {t:>5} {k:>14.4f} {c:>+15.4f} {s:>9.3f} {vk:>14.4f}")
        print(flush=True)

for SETNAME in SETS:
    print(f"\npaired trained-vs-untrained on {SETNAME} (3 seeds each):")
    for name, _, _ in ARMS:
        tr = [r for r in rows if r['arm'] == name and r['trained']
              and r['dset'] == SETNAME]
        unl = [r for r in rows if r['arm'] == name and not r['trained']
               and r['dset'] == SETNAME]
        if not tr or not unl:
            continue
        un = unl[0]
        signs = ' '.join(f"{r['corr']:+.2f}" for r in tr)
        print(f"  {name:>18}: KL {un['kl']:.4f} -> {np.mean([r['kl'] for r in tr]):.4f}"
              f"   corr {un['corr']:+.3f} -> {np.mean([r['corr'] for r in tr]):+.3f}"
              f"  [per seed: {signs}]"
              f"   P(stop) {un['pstop']:.3f} -> {np.mean([r['pstop'] for r in tr]):.3f}")

json.dump(rows, open(os.path.join(HERE, "data", "r9_probe.json"), "w"))
