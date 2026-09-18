# Round 2 — confirmation at n=8 and the real evaluation budget

Follow-up to `RESULTS.md`. 148 runs total (90 synthesis/repair + 40 hybrid ablation +
18 training-quality control), 4 parallel workers, ~5 h wall.

Everything here is **n=8 seeds** unless stated, with Welch t-tests. Round 1 showed that
n=3 comparisons reverse at n=8, so nothing below is reported at n<6.

---

## Headline

1. **The hybrid gain is real and large, but it is not coming from the reinforcement
   learning.** A randomly-initialised repair actor used as a GA mutation operator beats
   plain GA by **+33.3% (p<0.0001)**. Training that actor with the current pipeline adds
   **nothing** (−5.3%, p=0.06 — i.e. trending *negative*).
2. **Warm Start GA does nothing at all** — trained or untrained, it is indistinguishable
   from plain GA.
3. **The entropy bonus does not help**, even at 64,000 evaluations where the collapse it
   targets actually happens. My last surviving hypothesis from round 1 is dead.
4. **The repair-reward gain did not survive.** +4.0% at n=3 became +2.7% (p=0.11) at n=8,
   and vanished entirely at a larger budget (−0.0%, p=0.99).
5. **One thing did work**: an actor trained with the hypervolume reward *and* 3.3× more
   training is the first configuration to beat a random operator (+6.2%, p=0.068).
6. **Fixing the two implementation bugs lowers hypervolume** (0.374 vs 0.430) while cutting
   variance by 4×. They are still real bugs; see the recommendation.

---

## 1. Hybrid ablation — the project's core claim (n=8, 16,000 NFE, identical NFE per arm)

| Arm | mean HV | sd | min | max |
|---|---|---|---|---|
| plain GA | 0.5065 | 0.0231 | 0.4858 | 0.5540 |
| Warm Start GA, untrained actor | 0.5052 | 0.0146 | 0.4852 | 0.5321 |
| Warm Start GA, trained actor | 0.4990 | 0.0097 | 0.4853 | 0.5135 |
| Intelligent Mutation GA, **untrained** actor | **0.6750** | 0.0395 | 0.5991 | 0.7245 |
| Intelligent Mutation GA, trained actor | 0.6394 | 0.0285 | 0.5898 | 0.6898 |

Decomposition:

| Effect | Delta | % | p | |
|---|---|---|---|---|
| Operator structure alone (IMGA untrained vs plain GA) | +0.1685 | **+33.3%** | **<0.0001** | **significant** |
| What the learning adds (IMGA trained vs untrained) | −0.0355 | **−5.3%** | 0.060 | ns, trending negative |
| Warm-start structure alone (untrained vs plain GA) | −0.0013 | −0.2% | 0.898 | ns |
| What the learning adds (Warm Start trained vs untrained) | −0.0062 | −1.2% | 0.334 | ns |
| Total hybrid effect (IMGA trained vs plain GA) | +0.1330 | +26.3% | <0.0001 | significant |

**Reading:** the entire, large, statistically solid benefit of Intelligent Mutation GA comes
from *replacing PyGAD's random mutation with a loop of single-variable edits terminated by a
stop decision* — not from anything the policy learned. A network with random weights does
the job just as well, in fact slightly better.

This directly overturns the +11% "training helps" result I reported earlier from a 3-seed,
640-evaluation run. That number was noise.

Warm Start GA is a null result at every level: the initial population it produces is worth
nothing over PyGAD's random initialisation.

---

## 2. Is the trained actor bad because training is *broken*, or because it was *undertrained*?

Control: retrain the repair actor at 32,000 NFE (3.3× more than before) and with the
hypervolume reward, then reuse each as the IMGA mutation operator. (n=8, 16,000 NFE.)

| Repair actor used as mutation operator | mean HV | sd | vs random init | p |
|---|---|---|---|---|
| plain GA (no learned operator) | 0.5065 | 0.0231 | — | — |
| random init (untrained) | 0.6750 | 0.0395 | — | — |
| trained 9,600 NFE, domination reward | 0.6394 | 0.0285 | −5.3% | 0.060 |
| trained 32,000 NFE, domination reward | 0.6811 | 0.0240 | +0.9% | 0.716 |
| **trained 32,000 NFE, hypervolume reward** | **0.7169** | 0.0449 | **+6.2%** | **0.068** |

**Reading:** undertraining explains part of it — 3.3× more training moves a *harmful* actor
(−5.3%) back to parity (+0.9%). But parity is all the current reward buys. Only when the
reward is also changed to hypervolume contribution does the learned operator finally beat a
random one, and even then p=0.068 is borderline.

This is the single most promising result in either round, and it needs both changes at once:
**reward and training budget**. Note the asymmetry — the hypervolume reward does *not* help
the repair PPO as a standalone optimiser (section 4), but it does produce a better mutation
operator. That makes sense: an operator's job is to propose front-advancing edits, which is
exactly what the hypervolume reward trains for and what the domination-vs-parent reward
does not.

---

## 3. Synthesis PPO at the real 64,000-evaluation budget (n=8)

| Config | mean HV | sd | final entropy | collapse epoch | critic EV | p vs baseline |
|---|---|---|---|---|---|---|
| baseline (repo as-is) | 0.4303 | 0.0850 | 0.838 | 389 | −0.54 | — |
| entropy bonus 0.01 | 0.4189 | 0.0807 | 2.568 | 196 | −0.27 | 0.787 |
| entropy bonus 0.01, decayed | 0.4271 | 0.0876 | 1.243 | 357 | −0.45 | 0.942 |
| all ratio/PE fixes | 0.3742 | 0.0219 | 1.161 | 397 | −1.55 | 0.109 |
| all fixes + entropy decay | 0.3615 | 0.0165 | 1.478 | 292 | −1.55 | 0.057 |
| random search | 0.4059 | 0.0101 | — | — | — | 0.445 |

**The entropy bonus is dead.** It does exactly what it is supposed to mechanically — final
entropy rises from 0.84 to 2.57 — and hypervolume does not move (p=0.79). Collapse is not
what was limiting the policy. Pooling all 40 synthesis runs, collapsed runs average
HV=0.4071 and non-collapsed 0.3945 (p=0.53): whether the policy collapses is unrelated to
how well it does.

**PPO is still not distinguishable from random search** at the real budget (0.4303 vs
0.4059, p=0.45), with 8× the variance.

**The bug fixes lower the mean and cut the variance 4×** (sd 0.085 → 0.022). The buggy
version's advantage comes entirely from its upper tail (max 0.5665 vs 0.4058): the broken
importance ratio injects noise into the policy gradient, which occasionally acts as lucky
extra exploration. The corrected implementation converges reliably to a mediocre policy.
That is a more honest optimiser that reports a worse number.

---

## 4. Repair PPO reward target — did not replicate

| Reward | @4,800 NFE (n=8) | p | @12,800 NFE (n=6) | p |
|---|---|---|---|---|
| `domination` (current) | 0.2883 ± 0.0086 | — | 0.3025 ± 0.0122 | — |
| `front` | 0.2922 ± 0.0080 | 0.366 | 0.3059 ± 0.0124 | 0.639 |
| `hv` | 0.2962 ± 0.0101 | 0.115 | 0.3024 ± 0.0056 | 0.986 |

The +4.0% from round 1 (n=3) shrank to +2.7% (ns) at n=8 and disappeared at the larger
budget. **As a standalone search method, the reward target does not matter.** Its value is
only in the operator it produces (section 2).

---

## 5. What this means for the project

The paper's claim is that reinforcement learning improves evolutionary search on this
problem. The measurements say something more specific and, as it stands, less favourable:

- The **mechanism** that helps is architectural, not learned. Single-variable edits with a
  learned stop criterion are a much better mutation operator for a 72-gene discrete problem
  than PyGAD's `mutation_percent_genes=10` random mutation. That is a real and publishable
  finding — it is just not a reinforcement-learning finding, and the ablation is required to
  state it honestly.
- The **learning** currently contributes nothing, and with the shipped reward and training
  budget it is slightly harmful. The one configuration that beats random init needs both a
  hypervolume-based reward and 3.3× the training, and even then sits at p=0.068.
- **Warm Start GA should probably be dropped** or re-examined; it is a null result on every
  axis measured.

### Recommended next steps

1. **Run the ablation as a standard arm in every experiment.** `params['random_init_actor']`
   is now wired into `warm_start_ga`, `intelligent_mutation_ga` and
   `warm_start_intelligent_ga`, and the ablation methods are listed (commented out) in
   `main.py:48-51`. Without it, a reviewer cannot tell the +26% hybrid gain from a +33%
   operator gain that the RL is diluting.
2. **Push on section 2**: hypervolume reward + longer repair training, n≥16 to resolve
   p=0.068. This is the only live path to "the RL adds value".
3. **Fix the two bugs, and re-baseline.** They are genuine defects (`RESULTS.md` §5), but
   fixing them lowers reported hypervolume from 0.430 to 0.374. I have deliberately **not**
   applied them to the repo, because doing so silently would change your headline numbers.
   The honest framing is that the current numbers partly reflect noise-driven exploration
   from a broken importance ratio.
4. **Raise the seed count.** sd is 0.085 at 64k evaluations for the synthesis PPO — 6 runs
   cannot resolve anything smaller than ~0.09 HV. The hybrid arms are tighter (sd ~0.02-0.04)
   and 8 seeds are adequate there.

---

## Reproducing

```
# 90 synthesis/repair jobs, 4 workers
for i in 0 1 2 3; do python experiments/runner.py --shard $i --nshards 4 & done
python experiments/analyze.py

# 40 hybrid-ablation jobs (run with cwd containing results/ABL/)
python experiments/train_repair_actor.py
for i in 0 1; do python experiments/runner2.py --shard $i --nshards 2 & done
python experiments/analyze2.py

# 18 training-quality control jobs
for i in 0 1; do python experiments/runner3.py --shard $i --nshards 2 --mode train & done
for i in 0 1; do python experiments/runner3.py --shard $i --nshards 2 --mode imga & done
python experiments/analyze3.py
```

Raw per-run records: `shard_*.jsonl` (synthesis/repair), `abl_shard_*.jsonl` (ablation),
`ctl_shard_*.jsonl` (control). Rendered summaries: `round2_*.txt`.
