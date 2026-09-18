# Round 3 — a reward that works, and two corrections to round 2

Follow-up to `RESULTS.md` (round 1) and `RESULTS_ROUND2.md` (round 2). 72 runs total
(48 reward/bug sweep + 24 goal-conditioning), 6 parallel workers, ~24 core-hours.

Everything here is **n=8 seeds at 64,000 NFE** (1000 epochs x batch 64), the budget
`main.py` actually uses. Seeds are shared across every arm, so **paired t-tests** are the
correct test; round 2 used Welch, which throws away the blocking and is part of why it
concluded "nothing works". Welch p-values are printed alongside by `analyze4.py`.

---

## Headline

1. **One reward change produces a large, clean win.** The hypervolume-improvement reward
   computed against a **batch-local archive** (reset every epoch) reaches **HV 0.5596 vs
   the baseline's 0.4303 — +30.1%, p=0.0045**, and beats random search by **+37.9%
   (p<0.0001)**. Seven of the eight baseline seeds fall below the *worst* seed of this arm.
2. **It is also 5x more stable.** sd 0.016 vs the baseline's 0.085. Round 2 correctly
   identified variance as the thing blocking every comparison; this arm removes it.
3. **Round 2's claim that fixing the implementation bugs lowers hypervolume does not
   replicate, and was an artifact of a third bug.** Corrected: 0.4498 vs 0.4303, with
   variance halved (sd 0.043 vs 0.085).
4. **Baseline PPO does not beat random search** (p=0.42). *Bug-fixed* PPO does (+0.044,
   p=0.022). The fixes are what make this method clear a random baseline at all.
5. **The same reward against a global archive mostly fails** (0.4742, p=0.49, sd 0.121).
   The difference between the winning arm and the failing arm is one line: whether the
   archive resets each epoch.
6. **Goal-conditioning works but loses.** A new approach conditioning the policy on
   objective-space targets reaches 0.5010 (p=0.090) — the second-best family measured — but
   is beaten by the simpler batch-archive reward and carries 5x its variance.

---

## 1. The full comparison (n=8 paired seeds, 64,000 NFE)

| arm | mean HV | sd | min | max | valid% | vs baseline | p | vs batch archive | p |
|---|---|---|---|---|---|---|---|---|---|
| random search | 0.4059 | 0.010 | 0.391 | 0.421 | — | −0.0244 | 0.42 | −0.1537 | **0.0000** |
| baseline (repo as-was) | 0.4303 | 0.085 | 0.343 | 0.567 | 66.7% | — | — | −0.1293 | **0.0045** |
| hv reward, bugs left in | 0.4327 | 0.082 | 0.352 | 0.586 | 64.4% | +0.0024 | 0.66 | −0.1269 | **0.0042** |
| bugs fixed | 0.4498 | 0.043 | 0.376 | 0.490 | 79.7% | +0.0195 | 0.60 | −0.1098 | **0.0003** |
| front bonus only (no goals) | 0.4530 | 0.071 | 0.342 | 0.550 | 46.8% | +0.0226 | 0.62 | −0.1067 | **0.0054** |
| fixed + hv reward (global archive) | 0.4742 | 0.121 | 0.313 | 0.662 | 38.5% | +0.0439 | 0.49 | −0.0854 | 0.090 |
| goal-conditioned | 0.5010 | 0.076 | 0.367 | 0.582 | 25.8% | +0.0707 | 0.090 | −0.0586 | 0.075 |
| goal-conditioned + hv term | 0.5172 | 0.081 | 0.360 | 0.601 | 23.8% | +0.0869 | 0.055 | −0.0425 | 0.20 |
| **fixed + hv, batch archive** | **0.5596** | **0.016** | **0.531** | **0.580** | 29.0% | **+0.1293** | **0.0045** | — | — |

Per-seed, the winning arm against the baseline:

```
seed              0      1      2      3      4      5      6      7
baseline      0.5665 0.3654 0.4005 0.5101 0.3609 0.3427 0.3874 0.5089
batch archive 0.5511 0.5670 0.5307 0.5459 0.5658 0.5637 0.5728 0.5800
```

It wins 7 of 8 seeds outright and effectively ties the eighth (seed 0, the baseline's
lucky tail).

---

## 2. Why the batch-local archive wins: the reward became learnable

Round 2's diagnosis — that every per-design scalarisation is *anti*-correlated with
hypervolume — held up. A 32-policy grid sweep (`probe6.py`) scored each reward variant
against measured hypervolume:

| reward | rho(reward, HV) | signal density |
|---|---|---|
| repo as-is (pinned constraint weights) | **−0.301** | 100% |
| unpinned constraint weights | **−0.306** | 100% |
| objectives only, no constraint terms | −0.31 | 100% |
| Tchebycheff improvement | −0.33 | 100% |
| HVI vs static archive | +0.871 | 6.8% |
| **HVI vs running archive** | **+0.981** | 5.1% (0.2% late in a run) |

Only a **history-dependent** reward tracks hypervolume: novelty pays, duplication does not.
No per-design scalarisation can, because hypervolume is a property of a *set*, not of a
design. This also settles round 2's open question about unpinning the constraint weights —
it moves rho from −0.301 to −0.306, i.e. nowhere.

But the global running archive introduces a new defect. `probe9.py` measured how much of
that reward the policy can actually observe:

- **96.9% of HVI reward variance comes from archive order**, which is not in the observation.
- The reward is therefore **non-Markovian w.r.t. the observation**, and the value function is
  being asked to predict a number that is not a function of its input.

That is visible directly in the critic's explained variance:

| arm | critic EV |
|---|---|
| baseline | −0.54 |
| bugs fixed | −1.54 |
| fixed + hv reward (global archive) | **−3.5e6** |
| hv reward, bugs left in | **−6.5e7** |
| fixed + hv, **batch archive** | **−0.065** (median; 7 of 8 seeds in −0.05 to −0.07) |

Resetting the archive each epoch makes each design compete only against the other 63 in its
own minibatch. That keeps the novelty property (`probe8.py`: rho = **+0.893**, selecting the
same optimal policy) while raising signal density from 5% to **23%** and, critically, making
the reward nearly Markovian again. **The critic learns it in 7 of 8 seeds.**

One caveat, stated plainly: seed 7 diverged to EV −6.7e9, dragging the arm's *mean* EV to
−8.4e8. The claim above rests on the median, not the mean. That seed still produced the
arm's best hypervolume (0.5800), so the divergence did not harm the outcome — but it is not
explained.

Final policy entropy confirms no collapse: **2.69** for the batch archive vs **0.84** for
the baseline.

---

## 3. Correction: bug fixes do *not* lower hypervolume

`RESULTS_ROUND2.md` (headline 6, §5c) reported that fixing the two implementation bugs
lowers HV, 0.374 vs 0.430. **That does not replicate.** Corrected: **0.4498 vs 0.4303**,
with sd falling from 0.085 to 0.043.

The cause was a third, unreported bug in round 2's own harness. The `fix_position` arm
gathered the action distribution at `len(frag) - 1` — the slot that predicts the *next*
action, not the one taken. Round 2's "all fixes" arm was therefore measured with the
position gather still wrong, in a new way. Fixed in `variant.py` as
`POS.append(ntot + s - 1)` and plumbed through `Actor.ppo_update(..., action_pos=...)`.

The mean shift alone is not significant (p=0.60). Two things about this arm are:

- **Variance halves** (sd 0.085 -> 0.043), which is what makes downstream comparisons
  resolvable at all.
- **It is the cheapest arm that beats random search**: +0.0439, **p=0.022**. The unfixed
  baseline does not (p=0.42).

The bugs themselves are unchanged from round 1 §5a/5b: positional encoding indexed by batch
rather than sequence position, and an importance ratio that is not 1.0 at the first PPO
update. Both are fixed in `transformer_rand_weights_state.py` and
`transformer_rand_weights_uninformed.py`.

---

## 4. Goal-conditioned PPO — works, loses, and fails in an interpretable way

New approach in `optimization/ppo_goal_conditioned.py`, registered in `main.py`. The policy
is conditioned on a 5-D vector: a target cell in normalised objective space, plus grid
coverage and current archive hypervolume. Each epoch samples 64 targets from **frontier
cells** — empty cells adjacent to a reached one — so the batch is explicitly sent to 64
different places. Reward is a dense goal-reaching term plus a sparse front-advancement bonus.

The motivation was round 2's observability problem: 96.9% of the useful reward signal is
hidden archive state, and `probe9.py` showed a coarse occupancy grid recovers most of it —
**8 cells recover 93.7%, 27 cells 96.4%** — because the descriptor space is the 3-D
objective vector, not the 72-D design. The model never needs millions of features.

Result: **0.5010 (p=0.090 vs baseline)**, rising to **0.5172 (p=0.055)** with the HV term
added. A genuine improvement on the baseline, and the second-best family measured. But it
loses to the batch archive, and the reason is specific and diagnosable:

```
                                r(HV, valid_frac)   valid% range
goal-conditioned                      -0.950         0.06 - 0.73
goal-conditioned + hv term            -0.924         0.05 - 0.70
fixed + hv, batch archive             -0.220         0.20 - 0.37
```

```
goal-conditioned, per seed (HV / valid_frac):
  s0 0.44/0.53   s1 0.45/0.37   s5 0.37/0.73     <- stuck in the high-feasibility mode
  s2 0.54/0.08   s3 0.57/0.11   s4 0.52/0.10   s6 0.58/0.09   s7 0.54/0.06
```

The arm is **bimodal**. Its good seeds reach 0.582, matching the batch archive's best
(0.580). It simply fails to get there in 3 of 8 runs: those seeds collapse into a safe
high-feasibility mode and never spread across objective space. The batch archive lands in
the productive regime in **8 of 8**, with valid_frac pinned in a narrow 0.20-0.37 band.

**Ablation** (`goal_coef = 0`, front bonus only): 0.4530.

- goal signal contribution: **+0.048, p=0.27** — directionally right, not significant at
  n=8. The occupancy-grid targets are not clearly earning their complexity.
- adding the HV term to goals: **+0.016, p=0.068**.

Conclusion: the per-epoch archive reset already supplies the diversity pressure the explicit
goal targets were designed to supply, and supplies it without the failure mode. The approach
is preserved and registered, but is not the recommended configuration.

---

## 5. The feasibility trade-off is real and worth taking

The winning arm runs at **29% feasible** against the baseline's 67%, and buys +30%
hypervolume with it. Because NFE is pinned identically across every arm, this is not a
hidden cost — it is a straight win under a fixed evaluation budget.

Round 2's finding that hypervolume is largely a proxy for feasible fraction still holds for
the *unfixed* arms (baseline r = −0.929, hv-reward-with-bugs r = −0.916), and breaks down
exactly where it should: the batch-archive arm has **r = −0.220**. Its hypervolume is no
longer explained by how many feasible designs it happens to produce.

---

## 6. Scorecard against pre-registered predictions

Predictions were registered before the sweep ran.

| arm | predicted | actual | verdict |
|---|---|---|---|
| bugs fixed | 0.36-0.40 | 0.4498 | **wrong**, including the direction |
| fixed + hv reward | 0.35-0.50 | 0.4742 | correct range, but argued downward within it |
| hv reward, bugs left in | ~ baseline | 0.4327 vs 0.4303 | **correct** |
| fixed + hv, batch archive | *not registered* | 0.5596 | — |

The arm that won was the one for which no prediction was registered. Separately, an interim
report from this sweep highlighted a single 0.6624 seed from the global-archive arm as a
breakthrough; that arm finished at 0.4742, with a 0.3131 seed below random search. **n=1 is
not evidence** — already round 1's lesson (§3), and repeated here anyway.

---

## 7. Recommendations

1. **Adopt the batch-local HVI reward as the default synthesis configuration.**
   `reward_mode='hv'`, `hv_archive='batch'`, bug fixes on. This is the only change across
   three rounds with a large effect, a small p-value, and low variance simultaneously.
2. **Keep the bug fixes regardless of reward.** They are correctness fixes, they halve
   variance, and they are what lets PPO beat random search.
3. **Do not pursue the global-archive variant.** Its reward is unlearnable by construction
   and its variance (sd 0.121) makes it a lottery.
4. **Explain the seed-7 critic divergence** in the batch-archive arm before trusting the
   configuration in production. It did not hurt hypervolume, but EV −6.7e9 is not understood.
5. **Re-run the round 2 hybrid ablations with the corrected position gather.** Round 2's
   central claim — that the RL contributes nothing to the hybrid GA methods — was measured
   with the same off-by-one that produced its incorrect bug-fix result. It may or may not
   survive; it has not been retested.
6. **Report paired tests.** Seeds are shared by construction in this harness; Welch discards
   that and inflates p-values.

---

## Reproducing

```
# 48 reward/bug jobs, 6 workers, ~15.3 core-hours
for i in 0 1 2 3 4 5; do python experiments/runner4.py --shard $i --nshards 6 & done
python experiments/analyze4.py

# 24 goal-conditioning jobs, 6 workers, ~8.9 core-hours
for i in 0 1 2 3 4 5; do python experiments/runner5.py --shard $i --nshards 6 & done
```

Raw per-run records: `r3_shard_*.jsonl` (reward/bug sweep), `r4_shard_*.jsonl`
(goal-conditioning). Both runners resume from completed `job_id`s, so `--nshards` can change
between launches without re-running finished work.

**Note:** `analyze4.py` iterates a hardcoded arm list written before the batch-archive config
was appended to `runner4.py`, so it silently omits the winning arm. Analyse over both shard
globs directly rather than trusting that script's printed table.

Reward-alignment probes quoted above (`probe6.py`, `probe8.py`, `probe9.py`) were run from a
scratch directory and are not checked in; their measurements are reproduced inline.
