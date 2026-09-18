# Rounds 7–9: does the learned mutation operator have anything to learn?

> **Round 10 revises Experiment I.** The "oracle ceiling" measured below used operators that
> each optimise a single scalar while being scored on hypervolume. An operator that targets
> hypervolume gain directly does beat blind mutation when its lookahead is free, and loses
> when the lookahead is charged as evaluations. See [`RESULTS_ROUND10.md`](RESULTS_ROUND10.md).

**Status: complete.** Every experiment described here has finished. Numbers are
regenerated from the raw shard files by `python experiments/analyze_round9.py`.

All hypervolumes are on the lab's original `max_values.npy`. Numbers in
`RESULTS_ROUND6.md` are on a rebuilt reference that was later reverted and are **not**
comparable to anything here.

### Audit pass

Every table below was recomputed from the raw `r9*_shard_*.jsonl` files by an independent
script that does not import `analyze_round9.py`. Means, standard deviations, seed counts,
paired deltas, win counts and p-values reproduce to four decimal places. Supporting checks:
`max_values.npy` is byte-identical to `max_values_random16k.npy`, confirming the round-6
reference was genuinely reverted; all 198 shard records carry a usable `hv` with none
dropped; the non-empty `*_err_*.txt` logs contain only pygad's `keep_parents` warning, not
failures; and seed sets are identical across arms within each experiment, so `ttest_rel` is
the correct test.

Two defects were found and are corrected in place, each flagged at the point it occurs:

1. **Experiment VIII did not run both families at one budget** — GA arms spent ~56,060 NFE
   against the PPO's 64,000. The within-family contrasts that carry the conclusions are
   unaffected; the cross-family remark is conservative in the direction stated.
2. **The oracle table's first row was stale** (0.8010, actually 0.8053), so the header's
   former claim that "nothing is hand-copied" was itself false and has been withdrawn.

Neither changes a conclusion. **The weakest evidence in this document is sample size**:
Track C and experiments X and XI rest on n=3, which is the sample size round 1 established
is unreliable here and which experiment VIIb saw point the wrong way at n=4. Experiment X's
−0.1873 and the "leave the flat zero alone" recommendation should be re-run at n=8 before
either is relied on.

Not covered by this pass: rounds 1–6 raw data, and the correctness of `TrussProblem`'s
finite-element solver. Every arm is scored relative to that solver, so an error in it would
preserve rankings while invalidating absolute values.

---

## The short version, without jargon

Rounds 1–6 ended on an awkward note. The headline method — a neural network that learns
*how to mutate* a truss design inside a genetic algorithm — turned out to be winning for a
boring reason. It wasn't making smarter changes; it was making **fewer** changes, and
fewer changes happened to suit this problem. Once the plain genetic algorithm was allowed
to make one change at a time too, it beat every learned version by a wide margin.

That left an obvious question: *fine, but could a smarter operator work, if we fixed the
things that were broken?* Rounds 7 and 8 fixed them — the network can now see the physics,
and its reward no longer secretly punishes it for doing anything. Round 9 asked the
question that should have come first:

> **Is there anything to learn here at all?**

The answer is no, and it is not close. I hand-built mutation operators that are given
*perfect* physics information for free — the exact stress in every bar, and the textbook
structural-engineering rule for what to do about it. Every single one of them performs
**worse** than picking a bar at random and changing it to a random size.

So the network was not failing to learn a useful rule. There is no useful rule of that
shape to learn on this problem. That is a fact about the problem, not about the network,
and it means no amount of better reinforcement learning would have rescued this approach.

Round 9 also found *why*: the engineering rule the operator is meant to learn **reverses**
partway through every search, so no fixed rule can be right, and a population search loses
more from concentrating its mutations than it gains from aiming them well.

And it found one change that does help, which has nothing to do with the operator: this
benchmark is normally posed with **16** design variables rather than 72. Posing it that way
lets the genetic algorithm reach a given quality with roughly **8x fewer** evaluations, and
raises the reinforcement-learning method's score by **+0.100** (8 of 8 seeds, p=0.0003) --
about four and a half times what it gives the GA, because the RL is the method that runs
out of budget first.

Several claims from earlier rounds did not survive re-measurement, including two of my own
from this round. They are listed in *Corrections* below rather than quietly dropped.

---

## Why there is nothing to learn: the rule reverses

The intuition behind the whole approach is a real engineering principle: a bar carrying
almost no load is wasted weight and should be shrunk; a bar at its stress limit needs
thickening. So a good operator should look at each bar's **utilisation** (how close it is
to its stress limit) and act accordingly.

I measured whether that is actually true here, by taking real designs at three points in a
search, changing one bar at a time across the whole range of utilisations, and recording
whether the design got better or worse (`exp_landscape.py`, 4,320 measured mutations).

| designs measured on | corr(&#124;utilisation&#124;, improvement) | p |
|---|---|---|
| random designs | **+0.52** | 3.6e-99 |
| after 100 generations | **+0.30** | 1.2e-31 |
| after 400 generations | **−0.27** | 1.0e-24 |

**The sign flips.** Early in a search, the bars worth touching are the *highly* stressed
ones. Late in a search, they are the *lightly* stressed ones. A single fixed rule is
therefore wrong for a large part of every run — while picking at random is never
*systematically* wrong.

Looking closer says what is really going on:

| stage | quartile of &#124;util&#124; | P(mutation improves) | mean change | mean gain *when* it improves |
|---|---|---|---|---|
| random | lowest | 58.3% | +0.00103 | +0.00427 |
| random | highest | 39.4% | +0.00062 | **+0.01839** |
| 400 gens | lowest | 39.2% | −0.00004 | +0.01029 |
| 400 gens | highest | **16.1%** | **−0.02432** | +0.00672 |

Editing a heavily loaded bar is **high variance** — the biggest wins and the biggest
losses both live there. Editing an unloaded bar is **low variance** — very little happens
either way. The operator in rounds 5–8 learned to prefer the unloaded bars, which is
learning to *do nothing safely*. That is exactly the behaviour measured: ~1–2 tiny edits
per design and a near-uniform choice of which bar to touch.

---

## Experiment I — the oracle ceiling

If a perfect operator can't beat random, the direction is closed regardless of how good
the learning is. So I replaced the network with hand-coded operators built from textbook
structural engineering, gave them the per-bar stress for free, and ran them in the same
GA at the same 64,000-evaluation budget (`exp_oracle.py`, 6 seeds each).

Reading the stress costs a solve, and that solve is **counted as an evaluation** — so the
blind control `random` pays exactly the same cost, and the comparison isolates the *rule*
rather than its price.

| operator | which bar | what value | hypervolume |
|---|---|---|---|
| tuned GA (no physics at all, no solve cost) | uniform | uniform | **0.8053** |
| `random` (blind control, pays the solve) | uniform | uniform | 0.7673 |
| `low_util` | least-utilised (argmin) | shrink, random | 0.7338 |
| `hi_soft` | sampled ∝ &#124;util&#124; | uniform | 0.6621 |
| `lo_fsd` | least-utilised (argmin) | fully-stressed sizing | 0.5704 |
| `fsd_one` | uniform | fully-stressed sizing | 0.5598 |
| `high_util` | most-utilised (argmax) | thicken, random | 0.4672 |
| `hi_fsd` | most-utilised (argmax) | fully-stressed sizing | 0.3723 |
| `fsd_all` | every bar at once | fully-stressed sizing | 0.2457 |

> **Correction — the first row.** It previously read 0.8010. That number was stale: the row
> is not produced by `exp_oracle.py` at all, it is imported from Track C by
> `analyze_round9.py:83`, and the true value of those records is **0.8053 ± 0.0145**. It is
> also **n=3 on a different seed set**, and it does not pay the utilisation solve, so it is
> context rather than a paired comparison — every paired contrast below is against `random`,
> which does pay the solve and is measured on the same six seeds. No ordering changes; every
> physics rule sits far below both values.

Paired against the blind control, on the same seeds:

| operator | Δ vs `random` | seeds better | p |
|---|---|---|---|
| `low_util` | −0.0335 | 1/6 | 0.13 |
| `hi_soft` | −0.1053 | 0/6 | 2.2e-05 |
| `lo_fsd` | −0.1970 | 0/6 | 3.3e-06 |
| `fsd_one` | −0.2076 | 0/6 | 9.3e-06 |
| `high_util` | −0.3001 | 0/6 | 1.6e-06 |
| `hi_fsd` | −0.3950 | 0/6 | 8.3e-07 |
| `fsd_all` | −0.5217 | 0/6 | 4.1e-09 |

**Nothing beats blind mutation.** The best physics rule is statistically
indistinguishable from random; every other one is decisively worse.

### Separating "greedy" from "wrong"

The `high_util` and `hi_fsd` arms always edit the *same* bar (the argmax), so every
offspring gets the same gene touched and population diversity collapses. That is a
different failure from "the rule is wrong", so `hi_soft` samples the bar in proportion to
utilisation — same physics preference, no determinism:

| contrast | Δ | seeds | p | reading |
|---|---|---|---|---|
| `hi_soft` − `high_util` | **+0.1948** | 6/6 | 0.0001 | greedy determinism costs this much |
| `hi_soft` − `random` | **−0.1053** | 0/6 | 2.2e-05 | the preference *still* costs this much on top |

Both effects are real and separable. Being greedy is expensive, and preferring loaded bars
is expensive even when you aren't greedy.

---

## Experiment VI — the architecture *can* learn bar selection; PPO just doesn't teach it

Experiment IV (below) shows the pointer is being asked to learn something real. That
leaves two explanations for its collapse: the network cannot represent the mapping, or the
mapping is representable but PPO never delivers the signal. Supervised learning separates
them — same architecture, same inputs, but the label is brute-forced (every one of the 72
bars scored by the average improvement from editing it) and credit assignment is trivial.

| | picks the truly best bar | in true top-5 | KL(pointer ‖ uniform) |
|---|---|---|---|
| chance | 1.4% | — | 0 |
| textbook rule: edit least-utilised bar | 7.8% | 14.7% | — |
| textbook rule: edit most-utilised bar | 6.9% | 11.2% | — |
| **supervised pointer, same architecture** | **25.9%** | **53.4%** | **0.6359** |
| every RL-trained pointer | — | — | **0.0000–0.0017** |

The same network reaches 26% top-1 — 18× chance, 3.3× the best textbook heuristic — and
ends up **370× further from uniform** than any RL-trained pointer, having been given the
identical inputs. It was still improving when training stopped at epoch 30.

**So the pointer collapse is credit assignment, not representation.** No architectural
change was needed; the reinforcement learning simply never delivered a usable signal, and
experiment II says why — the correct answer reverses sign partway through every search, so
the gradient averaged over the state distribution points nowhere in particular.

Accuracy also degrades as designs converge (33.3% on random designs, 30.0% at 100
generations, 16.3% at 400), which matches experiment II: late in a search the useful signal
is mostly "don't break the loaded bars", and the magnitudes are small.

---

## Experiment IX — even a *good* learned pointer loses to random

Experiment I ruled out every **fixed** physics rule, and experiment II explained why a
fixed rule must lose: the correct choice reverses partway through a search. That left one
shape of operator standing — a **state-dependent** one, which can say "high utilisation"
early and "low utilisation" late. Experiment VI produced exactly that, so it was deployed
as the GA's mutation operator (`exp_deploy_supervised.py`), with the utilisation solve
counted as an NFE for every arm including the blind control.

| operator | how the bar is chosen | hypervolume | Δ vs blind | seeds better | p |
|---|---|---|---|---|---|
| `random` | uniform | **0.7755** ± 0.0184 | — | — | — |
| `supervised` | sampled from the supervised pointer | **0.7395** ± 0.0157 | **−0.0360** | **0/6** | **0.0098** |
| `sup_greedy` | the pointer's single best bar (argmax) | **0.4803** ± 0.0098 | **−0.2952** | **0/6** | **<0.0001** |

**It loses.** A pointer that picks the truly best bar 26% of the time — 18× chance, 3.3×
the best textbook heuristic — is worse as a mutation operator than picking uniformly at
random.

The greedy arm also **independently replicates the determinism result from experiment I**.
Taking the pointer's argmax instead of sampling from it costs a further **−0.2592** (0/6,
p<0.0001); experiment I measured the same effect on a completely different operator
(`high_util` vs `hi_soft`) at **−0.195**. Two unrelated operators, one mechanism:
committing every offspring to a single "best" gene collapses population diversity, and that
costs roughly 0.20–0.25 hypervolume — an order of magnitude more than choosing the right
bar could ever return.

This is the result that closes the direction, because it removes every remaining excuse.
The failure is not that the network cannot learn the rule (experiment VI: it can), not that
the physics never reached it (the stress channel demonstrably changes its behaviour,
&#124;corr&#124; 0.44 vs 0.14 without), and not that the reward punished acting (shaping fixed
that, P(stop) 0.96 → 0.55). Choosing bars well simply does not help this search.

The reconciliation with experiment IV is the interesting part. Bar choice really does drive
24% of the variance in a **single** mutation's outcome. But a genetic algorithm does not
make single mutations — it makes thousands and lets selection keep the good ones. Random
mutation spreads those trials across all 72 bars; a pointer concentrates them on the bars
it believes are good, and the diversity lost costs more than the better choices return.
An operator can be *right* and still be *harmful* in a population-based search.

---

## Experiment IV — the flat pointer is a genuine failure, not a rational one

My working hypothesis was comfortable: maybe *which* bar you edit barely matters, so a
network that refuses to express a preference is simply being correct, and the collapse to
uniform was never a bug.

That hypothesis is **wrong**. A two-way variance decomposition over a full bar × value
grid (`exp_variance.py`, 108 designs × 12 bars × 10 values) says:

| stage | bar effect | value effect | interaction |
|---|---|---|---|
| random designs | 24.8% | 16.5% | 58.7% |
| after 100 gens | 22.9% | 14.8% | 62.3% |
| after 400 gens | 24.6% | 17.2% | 58.2% |
| **all** | **24.1%** | **16.2%** | **59.7%** |

Choosing *which bar* matters **more** than choosing *what value* (24.1% vs 16.2%). So the
pointer head is being asked to learn something that genuinely carries signal, and its
collapse is a real learning failure.

The 60% interaction term is the other half of the story: the right value **depends on**
which bar you picked, so neither head can be right on its own.

---

## Experiment VII — the one change that helps: pose the problem properly

Every method in this repository optimises **72 independent bar areas**. That is not how
this benchmark is posed in the structural optimisation literature. The 72-bar space truss
is a standard test problem solved with **16 design variables**: the structure has four
identical levels, and within each level the 4 legs, 8 diagonals, 4 top chords and 2 cross
braces are interchangeable. I verified the geometry satisfies this exactly — bar lengths
are identical inside every group.

The repo is therefore searching a space of 50⁷² instead of 50¹⁶.

### At the full budget: a small but real gain

`exp_budget.py`, 12 matched seeds per formulation:

| formulation | hypervolume at 64,000 evals | delta vs 72 bars | seeds better | p |
|---|---|---|---|---|
| `groups16` - 16 group variables | **0.8248** +- 0.0170 | **+0.0198** | **9/12** | **0.027** |
| `bars72` - the repo's formulation | 0.8050 +- 0.0149 | - | - | - |

This took twelve seeds to establish, and the first four pointed the wrong way. On seeds
0-3 alone the difference was **-0.0007** (2/4, p=0.97), and an earlier draft of this
document concluded from it that there was "no final-quality gain". On seeds 4-11 it is
**+0.0300**. The four-seed result was the unlucky draw, and the twelve-seed answer agrees
with an independent measurement through the repo's own `run_genetic_algorithm`
(+0.0334, 3/3, p=0.009). Two harnesses, one conclusion.

The lesson belongs next to the result: a ~0.02 effect against a ~0.015 seed-to-seed spread
is not resolvable at four seeds, and when two measurements disagree the temptation is to
keep the significant one. Adding seeds is what settles it.

### Where the effect is largest: budget

| evaluations | `bars72` | `groups16` |
|---|---|---|
| 2,000 | 0.5163 | **0.7131** |
| 4,000 | 0.5907 | **0.7530** |
| 8,000 | 0.6564 | **0.7826** |
| 16,000 | 0.7154 | **0.7962** |
| 32,000 | 0.7674 | **0.8141** |
| 64,000 | 0.8050 | **0.8248** |

The 16-group formulation reaches **0.713 after 2,000 evaluations**; the 72-variable
formulation needs about **16,000** to match that. An **8x reduction in evaluations** for
the same quality, sustained across the whole curve.

So the classical formulation wins on both counts, by very different margins: a modest
**+0.02** at the endpoint, and roughly **8x cheaper** to reach any given quality along the
way. A genetic algorithm with 64,000 evaluations to spend collects the endpoint gain. A
method nowhere near converged at that budget collects the budget effect instead - which is
what experiment VIII found.

(A third arm, `sym_mut` - 72 variables with symmetry-aware mutation moves - was measured at
0.8193 +- 0.0289 on 4 seeds. It was not extended to 12 seeds, so it is reported here
without a significance claim.)


---

## Experiment VIII — the formulation is worth far more to the RL than to the GA

The synthesis PPO builds a design autoregressively, one variable at a time, and receives a
single scalar reward at the end. With 72 variables that is a 72-step credit-assignment
problem. With 16 it is a 16-step one. Nothing else changes within a family: same policy,
same reward, same evaluation budget, same hypervolume reference, same solver — the design
is expanded back to 72 bar areas before any physics is computed (`runner25.py`, verified to
give bit-identical objectives to the expanded 72-bar design).

> **Correction — the two families did not spend the same budget.** An earlier version of
> this section claimed "same 64,000-evaluation budget" across all four arms. That is wrong.
> The PPO arms spend exactly 64,000; the **GA arms spend ~56,060**. `runner25.py` does not
> set `num_generations`, so the GA is stopped by its generation count before `nfe_budget`
> binds — `keep_elitism=8` carries eight individuals over unevaluated, so a generation costs
> 56 evaluations, not 64. (`runner23.py` sets generations generously and does reach 64,015,
> which is why Track C is unaffected.)
>
> The two paired contrasts below are **within-family**, and both arms of each share a
> budget, so `+0.1000` and `+0.0223` stand as measured. The cross-family remark that the PPO
> remains behind the GA involves a GA that won on **12% fewer evaluations**, so it is
> conservative in the direction stated. The defect is in the sentence, not the finding.

All four arms are run **inside the same experiment on the same metric**, which matters: the
round-6 re-baseline reported the synthesis PPO at 0.3000, but every number in
`RESULTS_ROUND6.md` was measured on a *rebuilt* `max_values.npy` that has since been
reverted (see that file's own header). Round-6 numbers are therefore not comparable to
anything here, and the comparison below deliberately does not use them.

| method | formulation | hypervolume (8 seeds) | Pareto points | wall time |
|---|---|---|---|---|
| tuned GA | **16 symmetry groups** | **0.8154** +- 0.0202 | 501 | 152 s |
| tuned GA | 72 free bars | 0.7931 +- 0.0207 | 588 | 149 s |
| synthesis PPO | **16 symmetry groups** | **0.7036** +- 0.0313 | 81 | 370 s |
| synthesis PPO | 72 free bars | 0.6036 +- 0.0198 | 77 | 1567 s |

| paired contrast | delta | seeds | p |
|---|---|---|---|
| grouping, on the **GA** | +0.0223 | 6/8 | 0.042 |
| grouping, on the **PPO** | **+0.1000** | **8/8** | **0.0003** |

The reformulation is worth about **4.5x more to the PPO than to the GA**, and it makes the
PPO **4x faster in wall-clock** (370 s vs 1567 s) because the episode is a sixth as long.
It narrows the PPO's gap to the GA from -0.190 to -0.112.

It does not close it. The synthesis PPO in its best formulation is still clearly behind a
plain genetic algorithm, and the reformulation helps the GA too, so grouping does not turn
the RL into the better method. What it does is remove a handicap that was costing the RL
five times what it was costing the GA -- which is exactly what the budget argument predicts,
since the GA is near convergence at 64,000 evaluations and the PPO is nowhere near it.

This was 3 seeds when first reported, at +0.1065 with p=0.111. It was extended to 8 because
it was the largest positive effect of the round and the least well established; at 8 seeds
it is 8/8 and p=0.0003.

### A correction to an earlier draft of this section

An earlier version claimed the grouped PPO went from 0.3000 to 0.7583, "from below random
search to competitive with the tuned GA", a gain of +0.46. That compared this experiment's
grouped PPO against the **round-6** 72-bar PPO — and every number in `RESULTS_ROUND6.md`
is on a *rebuilt* `max_values.npy` that has since been reverted, as that file's own header
states. It was a cross-metric comparison of exactly the kind this document criticises
elsewhere. The matched, same-metric gain is **+0.100** (8 seeds), and the grouped PPO does
**not** reach the GA.

---

## Corrections to earlier claims

Two things I reported in earlier rounds do not survive a careful re-measurement. Both are
recorded here rather than quietly dropped.

### 1. "The stress channel makes the pointer physics-aligned (corr −0.73, 3/3 seeds)"

**This does not hold, and the −0.73 is not reproducible.** Re-measuring all 18 trained
actors with one consistent probe (`exp_probe.py`), on both uniformly random designs and
evolved designs:

| arm | seed | corr on random designs | corr on evolved designs |
|---|---|---|---|
| `B_stress` | 0 | +0.624 | +0.677 |
| `B_stress` | 1 | −0.335 | −0.460 |
| `B_stress` | 2 | +0.621 | +0.585 |
| `D_stress_shaping` | 0 | +0.624 | +0.706 |
| `D_stress_shaping` | 1 | −0.478 | −0.548 |
| `D_stress_shaping` | 2 | +0.506 | +0.458 |
| `F_stress` | 0 | +0.319 | +0.274 |
| `F_stress` | 1 | −0.098 | −0.217 |
| `F_stress` | 2 | +0.221 | +0.207 |

The **sign is set by the training seed, not by anything about the physics**. Seed 1 goes
one way and seeds 0 and 2 the other, in all three stress arms.

I first guessed that the discrepancy was the design distribution — the round-8 probe drew
uniformly random designs (`probe8.py:18`) while the operator is deployed on evolved ones,
and the landscape sign flip made that a tempting explanation. **The data refutes it.** The
two distributions agree per actor at **r = +0.995**; the distribution is not what varies.

What survives, and what does not:

| claim | verdict |
|---|---|
| the stress channel makes the pointer's preference **stronger** | **holds** — mean &#124;corr&#124; 0.442 with the channel vs 0.141 without, p < 1e-5 |
| that preference is **aligned with the physics** | **does not hold** — signed mean +0.205, not distinguishable from zero (p=0.067), and *positive* if anything, the opposite of the reported −0.73 |

So the stress channel does reach the network and does change its behaviour. It simply does
not push the behaviour in a consistent direction — which is exactly what experiment II
predicts should happen when the correct direction reverses partway through every search.

### 2. "Random search (0.4008) beats the synthesis PPO (0.3000)"

**This does not hold either, for two compounding reasons.**

First, `optimization/random_search.py:48` builds its hypervolume reference **from its own
sample** (`max_values = max(all_obj) * 1.1`), while every other method is normalised by the
shared `max_values.npy`. Random search is graded on a curve fitted to its own results.
Scoring one identical 64,000-design sample both ways (`exp_rs_metric.py`, 6 seeds):

| random search, scored on | hypervolume |
|---|---|
| its **own** reference, as shipped | **0.4025** +- 0.0164 |
| the **shared** reference, as everything else | **0.3624** +- 0.0088 |
| inflation | **+0.0402**, 6/6 seeds, p=0.0014 |

Second, the 0.3000 for the PPO came from `RESULTS_ROUND6.md`, which is on the *rebuilt*
reference that has since been reverted. Neither number in the original comparison was on
the metric the repository currently uses.

On the lab's own reference, measured this round at the same 64,000 evaluations:

| method | hypervolume |
|---|---|
| tuned GA | **0.8050** |
| synthesis PPO | **0.6071** |
| random search | **0.3624** |

**The synthesis PPO beats random search comfortably.** The reported ordering was an
artifact of the two problems above, not a property of the method. (Note that the shipped
metric's 0.4025 is essentially the 0.4008 round 6 reported — the inflation is the whole
gap.)

**The random-search normalisation is a bug in the repository, not just in my analysis**,
and it affects any comparison that has ever involved that baseline.

### 3. Where the synthesis PPO's budget goes -- and why that is not the problem

The runs record what fraction of generated designs satisfy the constraints, and the answer
looked damning:

| method | feasible designs generated | distinct designs |
|---|---|---|
| tuned GA | 96-99% | - |
| random search | 65.5% | - |
| synthesis PPO | **10.3%** | 100.0% |

The PPO spends about **90% of its 64,000 evaluations on infeasible designs**. It is not
mode collapse -- every design it generates is distinct. And there is an obvious culprit in
the reward. Under `reward_mode='hv'`, the setting round 6 selected and every round-9 run
uses, `optimization/ppo_rand_weights_state.py:246` reads:

```python
step_reward = 0.0 if constraints else archive.offer(to_hv(obj_values))[1] * hv_scale
```

**Every infeasible design receives exactly 0.0.** Not a graded penalty: zero, identically,
for all of them, so a design a hair over its stress limit is indistinguishable from one
wildly over. The repository's other reward mode does not behave this way -- under
`reward_mode='scalarized'` an infeasible design still scores through the two constraint
dimensions, which carry ~58% of the weight mass.

### The obvious fix makes it worse

That was a hypothesis, so it was measured rather than asserted (`runner26.py`, 3 matched
seeds, 64,000 evaluations):

| reward mode | hypervolume | feasible designs | Pareto points |
|---|---|---|---|
| `hv` (current default) | **0.6071** +- 0.0325 | **10.3%** | 58 |
| `scalarized` | 0.4197 +- 0.0536 | **75.4%** | 80 |
| difference | **-0.1873** (0/3, p=0.032) | **+65.1 points** (3/3, p=0.047) | +22 |

The scalarized reward **fixes the feasibility collapse completely** and **loses 0.19
hypervolume doing it.**

So the first half of the diagnosis holds: the zero-reward line is what produces the 10%
feasible fraction. The second half -- that this is why the PPO underperforms -- is wrong,
and the experiment says so unambiguously.

The reason inverts the intuition. In a constrained multi-objective problem the best
trade-offs sit **on** the constraint boundary. A policy spending most of its budget just
past that boundary is probing where the good designs are, and paying for it in wasted
evaluations; a policy that stays safely inside generates plenty of feasible, mediocre
designs and never reaches the front. The 10% feasible fraction is not a defect. It is the
better of the two behaviours, and the alarming-looking number belongs to the arm that
scores higher.

An earlier draft of this document called the low feasible fraction "a concrete, actionable
defect in the synthesis formulation and a better lead than anything in the
mutation-operator thread". That was wrong, and the experiment proposed to confirm it is
what disproved it.

### Experiment XI - and the middle option does not exist either

The flat zero still has a real cost: ~90% of episodes return the same constant, so the
policy cannot tell a near-miss from a disaster. Switching reward modes removes that
blindness but throws away the hypervolume objective at the same time. A new option
`infeasible_penalty` (added to `ppo_rand_weights_state.py`, default 0.0 so existing
behaviour is bit-identical) separates them: keep the hypervolume reward for feasible
designs, and give infeasible ones a graded penalty proportional to constraint violation
rather than a flat zero.

| `infeasible_penalty` | hypervolume | feasible | vs default (hv) | vs default (feasible) |
|---|---|---|---|---|
| **0.0** (current) | **0.6071** +- 0.0325 | 10.3% | - | - |
| 0.5 | 0.6108 +- 0.0211 | 17.6% | +0.0038 (1/3, p=0.85) | +7.4 pts (3/3, p=0.085) |
| 2.0 | 0.5761 +- 0.0407 | 29.9% | -0.0309 (0/3, p=0.18) | +19.7 pts (3/3, p=0.027) |

The graded penalty does not get the best of both. It slides along the **same trade-off
curve** the reward-mode comparison traced: more feasibility, less hypervolume,
monotonically. Nothing beats the flat zero.

**The existing design is therefore vindicated rather than merely inherited.** Giving every
infeasible design exactly 0.0 looks like an oversight in a diagnostics table; it sits at
the good end of a real trade-off, and both attempts to improve on it -- switching reward
modes and grading the penalty -- made things worse. That is worth recording explicitly,
because the opposite conclusion was the obvious one to draw and it would have been wrong.

(`infeasible_penalty` is left in the code at its default of 0.0. It changes nothing unless
set, and it is the knob anyone re-opening this question will want.)

---

## Experiment V - is the multi-objective weight conditioning doing anything?

The synthesis PPO is scalarised multi-objective RL: each episode draws a random weight
vector over the five objective/constraint dimensions and the policy is supposed to produce
a *different* design depending on the weights it is shown. The repo ships a `state` variant
that puts the weights in the observation and an `uninformed` variant that does not.

The naive comparison is misleading, and it is worth showing why. `ppo_rand_weights_uninformed.py`
has **no hv reward branch at all** -- it always uses the scalarized reward. So comparing the
two shipped variants confounds weight-visibility with reward mode, and experiment X measured
that reward mode alone is worth 0.19. Using experiment X's `state + scalarized` arm as the
missing control separates them:

| configuration | n | hypervolume | feasible |
|---|---|---|---|
| state + `hv` reward | 6 | **0.6033** +- 0.0229 | 12.7% |
| state + `scalarized` | 3 | 0.4197 +- 0.0536 | 75.4% |
| uninformed + `scalarized` | 6 | 0.3678 +- 0.0334 | 73.2% |
| random search | 6 | 0.3624 +- 0.0088 | 65.5% |

| contrast | what it isolates | delta | p |
|---|---|---|---|
| state+`hv` - state+`scalarized` | **the reward mode** | **+0.1873** (3/3) | 0.032 |
| state+`scalarized` - uninformed+`scalarized` | **seeing the weights** | +0.0446 (1/3) | 0.44 |
| uninformed+`scalarized` - random search | **the learning itself** | +0.0054 (3/6) | 0.85 |

Three readings, in descending order of confidence.

**The reward mode is doing nearly all of the work** -- four times more than the weight
conditioning, and it is the only contrast that reaches significance.

**The weight conditioning is worth little and is not established.** +0.045 on 3 paired
seeds at p=0.44 is not distinguishable from zero. The mechanism the method's
multi-objective story depends on is, at best, a small effect that this experiment cannot
confirm.

**Stripped of both, the PPO is random search.** With no weights and the scalarized reward,
it scores 0.3678 against random sampling's 0.3624 -- a gap of 0.005 on 6 seeds (p=0.85).
Whatever the synthesis PPO is contributing over random sampling, it is contributed by the
hypervolume reward, not by the policy learning to condition on trade-offs.

---

## What the probe does confirm

| measurement | untrained | after training | reading |
|---|---|---|---|
| P(stop) — baseline arm | 0.508 | **0.962** | the unshaped reward collapses the policy into "change nothing" |
| P(stop) — all four shaped arms | 0.40–0.54 | **0.51–0.59** | potential-based shaping fixes it, exactly as intended |
| KL(pointer ‖ uniform) | 0.0065–0.0100 | **0.0000–0.0017** | training pushes the pointer *toward* uniform in every arm |
| KL(value ‖ uniform) | 0.015–0.028 | **0.10–0.74** | the value head learns normally |

So the two heads behave completely differently under the same gradient: one learns, the
other is actively flattened.

---

## Track C - deploying the actual trained operators

Experiments I and IX ruled out hand-built and supervised operators. This is the direct
test on the operators rounds 5-8 actually produced, at 64,000 evaluations with the NFE cap
enforcing an equal budget for every arm (`runner23.py`, 3 seeds, 21 runs).

| arm | hypervolume | vs plain GA | seeds better | p |
|---|---|---|---|---|
| **tuned GA, no operator** | **0.8053** +- 0.0145 | - | - | - |
| IMGA trained `A_baseline`, tuned GA | 0.7965 +- 0.0064 | -0.0088 | 1/3 | 0.29 |
| IMGA untrained, tuned GA | 0.7780 +- 0.0097 | -0.0274 | 0/3 | 0.059 |
| IMGA trained `E_normdesign`, tuned GA | 0.7711 +- 0.0052 | -0.0342 | 0/3 | 0.094 |
| IMGA trained `F_stress`, tuned GA | 0.7472 +- 0.0242 | -0.0582 | 0/3 | 0.030 |
| IMGA trained `F_stress`, **old** GA | 0.6728 +- 0.0517 | -0.1325 | 0/3 | 0.066 |
| IMGA untrained, **old** GA | 0.5756 +- 0.0139 | -0.2298 | 0/3 | 0.0005 |

Three findings that need to be kept apart, because collapsing them is how rounds 1-6 went
wrong.

**Training helps - but only the plain operator.** Holding the GA fixed at its tuned
settings:

| operator | vs the untrained network | seeds better | p |
|---|---|---|---|
| `A_baseline` (no stress channel, no shaping) | **+0.0186** | 3/3 | **0.014** |
| `E_normdesign` (normalised input, shaping) | -0.0068 | 1/3 | 0.49 |
| `F_stress` (stress channel, shaping) | **-0.0308** | 0/3 | 0.067 |

**The round 7-8 "fixes" made deployment worse.** The stress channel and potential-based
shaping were added to repair defects that were real -- the physics was numerically
invisible, and the unshaped reward collapsed the policy into never acting. Both repairs
worked on their own terms (P(stop) 0.96 -> 0.55; the stress channel demonstrably reaches
the network). They also raised the training-time hypervolume slightly, 0.3163 against
0.3061 in round 8. And at deployment they cost 0.05 hypervolume relative to the plain
baseline operator, and put `F_stress` below the untrained network. The changes improved the
number they were tuned against and degraded the one that matters.

**But no configuration beats not having an operator at all.** The best is -0.0088 against a
GA that mutates one gene uniformly at random.

The remaining gap is the baseline, not the method. The *same untrained operator* scores
0.5756 under the historical GA settings (`keep_elitism=0`, `sss` selection) and 0.7780
under the tuned ones - a **0.20** difference produced entirely by the genetic algorithm it
is bolted onto, with no learning involved. Every hybrid result in rounds 1-6 was measured
against the crippled version, and that is where the reported advantage came from.

---

## Rounds 7–8 for reference (32,000 evaluations, 3 seeds)

| arm | hypervolume | sd |
|---|---|---|
| `D_stress_shaping` | 0.3285 | 0.0172 |
| `B_stress` | 0.3263 | 0.0203 |
| `F_stress` | 0.3163 | 0.0169 |
| `C_shaping` | 0.3117 | 0.0093 |
| `A_baseline` | 0.3061 | 0.0080 |
| `E_normdesign` | 0.3048 | 0.0108 |

No contrast reaches significance at 3 seeds (best: `D` − `A` = +0.0224, 3/3, p=0.064).
Given experiment I, chasing significance here would have been chasing an effect with no
headroom behind it.

---

## What to do next

Ordered by what the evidence actually supports.

**1. Adopt the 16-group formulation.** It is the standard way this benchmark is posed, it
costs nothing to implement, and it is the only change measured this round that helps every
method. Worth **+0.02** hypervolume to the GA at full budget (9/12 seeds, p=0.027) and about **8x
fewer evaluations** to reach a given quality; worth **+0.100 to the synthesis PPO** (8/8
seeds, p=0.0003), which is budget-limited and therefore collects an effect 4.5x larger, and
runs 4x faster besides.

**2. Fix `random_search.py:48`.** It normalises hypervolume against maxima from its own
sample, inflating its score by +0.040 (6/6 seeds, p=0.0014). Any published comparison
involving that baseline is affected.

**3. Stop work on the learned mutation operator.** Four independent measurements say the
direction has no headroom: no fixed physics rule beats blind mutation (experiment I), a
supervised pointer that is genuinely good also loses (experiment IX), the underlying rule
reverses sign mid-search so no stationary rule can be right (experiment II), and the
trained operators do not beat an operator-free GA at deployment (track C). The apparent
gains in rounds 1-6 came from a crippled GA baseline worth 0.20 hypervolume.

**4. Treat the synthesis PPO's weight conditioning as unproven.** Holding the reward fixed
it is worth +0.045 at p=0.44, and without it the method matches random search. If the
multi-objective story matters to the project, that mechanism needs to be established
before more is built on it.

**5. Leave the hv reward's flat zero alone.** It looks wrong and both attempts to improve
it made things worse.

## Questions for the lab, not changes to make

**The hypervolume reference.** `max_values.npy` is the *nadir* on the minimised axes but
the *ideal* on the maximised one, and designs routinely beat it on frequency, which is why
`hypervolume_utils.py` has to clip. Round 6 rebuilt it, that rebuilt reference produced the
numbers in `RESULTS_ROUND6.md`, and it was then reverted -- changing the measurement
invalidates every previously reported result, which is not a call to make from outside.
The clip is a workaround, not a fix, and someone who owns the benchmark should decide.

**Whether 72 free variables was deliberate.** The literature formulation is 16 groups. If
the 72-variable version is intentional -- to make the problem harder for the RL, say --
that is a fine reason, but it should be stated, because it costs every method roughly an
order of magnitude of budget.

---

## Reproducing

```
# diagnostics of the problem itself
python experiments/exp_landscape.py                                  # II  mutation landscape
python experiments/exp_variance.py                                   # IV  variance decomposition

# operators
python experiments/exp_oracle.py     --shard 0 --nshards 4           # I   oracle ceiling
python experiments/exp_oracle.py     --shard 0 --nshards 3 --set 2   # I   determinism control
python experiments/exp_supervised.py --stage data --shard 0 --nshards 4
python experiments/exp_supervised.py --stage train                   # VI  learnability
python experiments/exp_deploy_supervised.py --shard 0 --nshards 3    # IX  deploy it
python experiments/exp_probe.py                                      #     pointer statistics
python experiments/runner23.py       --shard 0 --nshards 6           # C   deployment

# formulation
python experiments/exp_symmetry.py   --shard 0 --nshards 4           # VII  16 groups
python experiments/exp_budget.py     --shard 0 --nshards 3           # VIIb budget curves
python experiments/exp_budget.py     --shard 0 --nshards 4 --extra   # VIIb seeds 4-11
python experiments/runner25.py       --shard 0 --nshards 4           # VIII grouped RL

# synthesis PPO
python experiments/exp_rs_metric.py                                  # 2    random-search metric
python experiments/runner24.py       --shard 0 --nshards 4           # V    weight conditioning
python experiments/runner26.py       --shard 0 --nshards 6           # X    reward mode
python experiments/runner27.py       --shard 0 --nshards 6           # XI   graded penalty

python experiments/analyze_round9.py                                 # regenerate the tables
```

Set `OMP_NUM_THREADS=1` when running several shards at once: each torch process otherwise
claims every core, and on a 16-core machine ~20 concurrent runs produced ~800 threads and
starved everything.

## Code changed in this round

| file | change |
|---|---|
| `evaluation/truss_problem.py` | `evaluate(..., return_utilisation=True)` returns the per-bar utilisation the solver already computes and used to discard |
| `optimization/ppo_rand_weights_state.py` | `infeasible_penalty` option (default 0.0, behaviour unchanged) |
| `utils/hypervolume_utils.py` | corrected a stale comment claiming the reference had been rebuilt; it was reverted, so the clip is load-bearing |
| `optimization/genetic_algorithm.py`, `intelligent_mutation_ga.py` | `nfe_budget` cap so every arm spends equal evaluations (elitism carries individuals over unevaluated, and a stress-aware operator costs extra solves) |
