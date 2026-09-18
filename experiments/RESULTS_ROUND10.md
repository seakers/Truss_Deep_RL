# Round 10 — the delta-hypervolume oracle, and why the truss oracle is closed-form

**Status: complete.** This round had no writeup when it was run; this file was assembled
afterwards from the two scripts' docstrings and from their regenerated output. Every table
is printed by `python experiments/analyze_r10.py` from `data/r10d1_shard_*.jsonl` and
`data/r10d2_shard_*.jsonl`, and by `python experiments/probe_rank1.py`. Nothing here is
hand-copied from memory.

All hypervolumes are on the lab's original `max_values.npy`, the same reference as rounds
7–9, at 64,000 evaluations, with the round-6 tuned GA (elitism 8, 1-bar mutation,
tournament selection) as the host. Six seeds, shared across arms, paired t-tests.

---

## Why round 9's "ceiling" is withdrawn

`RESULTS_ROUND7-9.md` Experiment I concluded that utilisation-guided mutation has a
ceiling: eight hand-built physics operators, given perfect per-bar stress for free, all
lost to blind mutation, and Experiment IX found a supervised pointer lost too.

Every one of those operators optimises a **single scalar**. Fully-stressed design is a
minimum-mass sizing rule with no displacement or frequency term; the supervised pointer's
labels come from `design_potential`, an equal-weight sum of the three normalised
objectives. Both are single directions through objective space. Both were scored by
**hypervolume**, which pays for spread across the front. An operator that is excellent at
climbing one direction is expected to underperform on a spread metric, so its losing says
nothing about what a well-aimed operator could do. (`fsd_all` at 0.2457 is largely a
64-design population collapsing onto a fixed point, after which a cumulative-archive
hypervolume simply stops growing; see the spread diagnostic below.)

The replacement experiment builds an operator whose target **is** the metric.

---

## Experiment I — the delta-HV oracle (set 1)

For each offspring the operator proposes K = 8 candidate single-bar edits, evaluates each,
and keeps the one that adds the most hypervolume to the running archive. That target is
direction-agnostic: a move that opens a new lightweight extreme and a move that opens a
new stiff extreme both score, so it cannot collapse the front the way a fixed rule does.

| arm | what it does |
|---|---|
| `random` | blind control: one uniform (bar, value) edit, charged 1 evaluation. Identical to the tuned GA's own operator |
| `dhv_free` | keep the argmax of true delta-HV. The lookahead is **free**: rejected candidates are neither charged nor entered into the scoring archive, so the arm gets information, not extra samples |
| `dhv_free_soft` | same, but sample among the candidates with positive delta in proportion to it. Controls for the determinism penalty round 9 measured at 0.20–0.25 on two unrelated operators |
| `dhv_charged` | every lookahead evaluation is charged as an evaluation and kept in the scoring archive. What the oracle actually costs if computed directly |

`front` is the number of points on the final Pareto front; `spread` is the product of
the front's per-objective ranges; `info` is the fraction of mutations where at least one
candidate advanced the archive, so the oracle had any signal to act on.

| arm | HV | sd | delta vs `random` | seeds better | p | front | spread | info |
|---|---|---|---|---|---|---|---|---|
| `random` | 0.7994 | 0.0131 | — | — | — | 658 | 0.1515 | — |
| **`dhv_free`** | **0.8448** | 0.0106 | **+0.0453** | **6/6** | **0.0017** | 834 | 0.1765 | 0.059 |
| `dhv_free_soft` | 0.8393 | 0.0145 | +0.0399 | 6/6 | 0.0042 | 874 | 0.1789 | 0.070 |
| `dhv_charged` | 0.7566 | 0.0165 | −0.0428 | 0/6 | 0.0041 | 834 | 0.0805 | 0.233 |

Three readings.

**The information is useful.** With the lookahead free, aiming edits at hypervolume gain
beats blind mutation on every seed. 0.8448 is the highest hypervolume measured on the
72-bar formulation in any round (round 9's tuned GA: 0.8053; its 16-group GA: 0.8248).
The round-9 ceiling claim is refuted on stronger evidence than it was made on: an operator
aimed at the metric wins, so what was wrong in round 9 was the target, not the idea. Every
learned operator in this repository trains on domination rewards or on `design_potential`
while being scored on hypervolume.

**It does not pay the determinism penalty.** Sampling among the positive candidates
instead of taking the argmax lands within 0.006 of the argmax version. Round 9 measured a
0.20–0.25 cost for committing every offspring to a single "best" gene under fixed rules;
an operator whose target rewards spread does not show it.

**Computed directly, it costs more than it returns.** Charged for its 8 lookahead solves
the same operator loses on every seed. Its front is as large as `dhv_free`'s but its spread
is half the control's, because nine evaluations per accepted move leave far fewer moves.
The operator is worth having only if the oracle can be had for much less than 8 solves,
which is the premise of amortising it into a network. The next section says that on the
truss no network is needed.

---

## Set 2 — the spread diagnostic on round 9's rules

Round 9 stored only hypervolume. Re-running its eight operators with the front cardinality
and per-objective ranges recorded:

| arm | HV | sd | delta vs `random_solve` | seeds better | p | front | spread |
|---|---|---|---|---|---|---|---|
| `random_solve` (blind, pays the solve) | 0.7647 | 0.0173 | — | — | — | 582 | 0.1154 |
| `low_util` | 0.7473 | 0.0183 | −0.0174 | 2/6 | 0.18 | 968 | 0.0724 |
| `hi_soft` | 0.6530 | 0.0159 | −0.1117 | 0/6 | 6.3e-05 | 459 | 0.0833 |
| `lo_fsd` | 0.5704 | 0.0258 | −0.1943 | 0/6 | 1.2e-05 | 708 | 0.0435 |
| `fsd_one` | 0.5656 | 0.0372 | −0.1991 | 0/6 | 2.8e-05 | 453 | 0.0608 |
| `high_util` | 0.4828 | 0.0381 | −0.2819 | 0/6 | 1.2e-05 | 505 | 0.0005 |
| `hi_fsd` | 0.3723 | 0.0300 | −0.3924 | 0/6 | 9.9e-07 | 88 | 0.0110 |
| `fsd_all` | 0.2457 | 0.0126 | −0.5190 | 0/6 | 8.8e-09 | 9 | 0.0047 |

The ordering and magnitudes reproduce round 9 (its `random` arm scored 0.7673). The new
columns show how the rules lost: the three worst arms have fronts whose spread is one to
two orders of magnitude below the blind control's, and `fsd_all` ends with nine points on
its front. These operators did not make bad edits so much as make the same kind of edit
until the population had nowhere left to go, which is exactly what scoring a single-scalar
rule on a spread metric predicts. `low_util`, the only rule that was not significantly
worse than random in either round, is also the only one whose front stays large.

---

## A single-bar edit is a rank-1 stiffness update

`probe_rank1.py` checks an identity that makes the oracle above cheap on this problem.

The element stiffness is T^T k_local T with k_local = [[1, −1], [−1, 1]] = v v^T, so each
element contributes (EA/L) b b^T with b = [d, −d]. The global stiffness K is therefore
**linear in the areas**, and changing one bar's area is a rank-1 update. From one LU
factorisation of the parent's K, the Sherman–Morrison formula gives the displacements of
every candidate single-bar edit exactly, mass is exactly linear, and the axial stresses
follow from the displacements. Only the fundamental frequency needs an approximation: a
first-order Rayleigh estimate using the eigenvector `truss_problem.py` already computes
and discards.

Checked against full re-solves on 40 designs × 25 single-bar edits (1,000 comparisons),
relative error of the closed-form value against the re-solved one:

| quantity | median | 90th percentile | max |
|---|---|---|---|
| mass | 0 | 0 | 0 |
| max displacement | 7.4e-16 | 2.7e-15 | 9.7e-15 |
| stress (max over bars) | 4.7e-16 | 1.6e-15 | 5.7e-15 |
| frequency (first-order Rayleigh) | 3.2e-04 | 1.7e-02 | 6.9e-01 |

Mass, displacement and both constraints are exact to machine precision. Frequency is
right to 0.03% on the median edit and to 1.7% on nine edits in ten, but has a heavy tail
(69% at worst), so a closed-form oracle would mis-rank some edits on that axis alone.

**What this means.** On the truss, the quantity a learned operator would be trained to
approximate, the delta-HV of every candidate single-bar edit, can simply be computed: one
factorisation per parent instead of 8 extra solves, and for all 72 × 50 candidates rather
than a sample of 8. The truss is therefore a poor testbed for the learned-operator claim.
A learned operator earns its keep where move evaluation is expensive **and** structurally
opaque, and on this problem it is neither. The closed-form oracle has **not** been deployed
as a GA operator here; the probe only establishes that it is available.

---

## What follows from this

1. **Round 9's recommendation 3, "stop work on the learned mutation operator", is
   narrowed.** The direction is closed *on the truss*, because the oracle is closed-form
   there, not because well-aimed operators cannot help. The predictive criterion is
   whether move evaluation is expensive and opaque; the spacecraft problem meets it and
   the truss does not.
2. **Anything trained on the truss should target hypervolume gain, not domination or a
   scalar potential.** That is the one target that beat blind mutation.
3. **GA-side improvements stay on the truss** and are cheap to test: rank-1 screening of
   candidate edits, SMS-EMOA-style survival selection, an ordinal mutation step, and
   archive-based selection are all unmeasured.
4. **Do not add more reward-shaping variants or bigger networks.** Nine rounds went there.

---

## Reproducing

```
python experiments/exp_dhv_oracle.py --shard 0 --nshards 6 --set 1     # set 1, 24 jobs
python experiments/exp_dhv_oracle.py --shard 0 --nshards 6 --set 2     # set 2, 48 jobs
python experiments/probe_rank1.py                                      # the identity
python experiments/analyze_r10.py                                      # the tables
```

Raw records: `data/r10d1_shard_*.jsonl` (set 1), `data/r10d2_shard_*.jsonl` (set 2).
Shards are resumable by `job_id`. Set `OMP_NUM_THREADS=1` when running several at once.
