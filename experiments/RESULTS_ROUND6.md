# Round 6 — the metric is repaired, the GA is tuned, and the ranking changed

Executes steps 1–3 of `PLAN_ROUND6.md`. Step 4 (the learned operator's state and reward)
was **not** started.

> **Every number in this file is on a rebuilt `max_values.npy` and is NOT comparable to
> rounds 1–5.** The old reference is preserved at `experiments/max_values_random16k.npy`.
> The frequency axis moved from 0.879 to 1.75, so all scores shift and — as the plan
> warned was possible — some rankings invert.

---

## Headline

> **Reference-dependence warning.** Hypervolume rankings depend on the normalisation
> constant, and this round demonstrably changed one (see §5). Claims 1-3 below are stated
> against the 1.75 reference and have **not** been shown to hold at other plausible
> values. A reference sweep and a reference-free dominance comparison are the outstanding
> work; neither has been run. Claim 5 rests on mechanism, not volume, and is unaffected.

1. **The synthesis PPO scores below random search on this reference.** 0.3000 vs 0.4008,
   0/8 seeds, p<0.00001. On the broken metric it looked competitive (0.55 vs 0.41). The
   gap is large, but its sign has not been tested against other reference values.
2. **The original GA also scores below random search here** (0.2620 vs 0.4008) -- and this
   is precisely the pair whose order inverted when the reference changed (§5), so it is
   the least trustworthy comparison in the table, not evidence for the first claim.
3. **The tuned GA wins by a wide margin**: 0.5997, beating every other arm by +0.19 to
   +0.34, **8/8 seeds on all five comparisons**, p<0.00001 throughout.
4. **The repaired reference holds.** Across 132 sweep runs the escape fraction is
   **0.0000%** — the clipping safety net never fired.
5. **Round 5's conclusion survives the metric repair.** A plain 1-bar GA (0.4101) still
   matches the trained transformer operator (0.4074), and IMGA-trained still beats
   IMGA-untrained by exactly the amount its edit count predicts.
6. **The re-normalisation inverted a ranking**, which is the risk `PLAN_ROUND6.md` flagged
   as "a real possibility, not a formality". It happened. See §5.

---

## Step 1 — the repaired metric

**The defect** (diagnosed in `RESULTS_ROUND5.md` §4): `max_values.npy` held the
per-objective best of a 16,000-sample random search. All ten optimizer files divide by
it, and `track_pareto_progress` did not clip. For the maximisation objective the code
computes `1 - obj/max_value`, so any design with frequency above the reference produced a
**negative** coordinate and pymoo integrated volume outside the unit cube.

**Bounds established** by pooling every feasible design from 12 deliberately diverse runs
(random search, 8-bar GA, 1-bar GA, 1-bar + elitism 8, 1-bar + elitism 32, IMGA trained;
2 seeds each — 637,203 feasible designs):

| objective | observed range | old reference | |
|---|---|---|---|
| mass (min) | 470.4 – 3980.7 | 3476.5 | unchanged |
| max displacement (min) | 0.0623 – 0.2000 | 1.00449 | unchanged |
| **frequency (max)** | up to **1.5850** | **0.8793** | **→ 1.75** |

Only index 2 changed, so the rescale is confined to the axis that was actually invalid.

**Why the old value was wrong.** `random_search.py:48` computes the whole array with one
line -- `np.max(...) * 1.1` over the sampled designs -- for three objectives that need two
different meanings. Mass and displacement are minimised, so `max_values` is the *nadir*
and the max over random designs is a sound, conservative estimate. Frequency is
maximised, so `max_values[2]` is the *ideal*, and the max over random designs asks random
search to estimate the optimum -- the one thing it is worst at. The reference is therefore
derived from the weakest method in the suite, and any method that genuinely beats random
search on the maximised objective escapes the box. This was not an accident waiting to
happen; it was guaranteed.

**Provenance of 1.75, stated accurately.** The bound was set from the pooled maximum of
the 12-run sweep above (1.5158), which was *not* the highest value already on record:
`r5i_shard_*.jsonl` contains 1.5850 from a 1-bar + nsga2 + elitism 8 run. The bounds
sweep also omitted nsga2 selection and the tournament + elitism 8 configuration that
became the round-6 default, so the reference was set from a sample that excluded the
best-performing config. Real headroom is therefore **~10% over 1.5850, not ~15%**.
Escape was 0.0000% across all 132 subsequent runs including the tuned config, so 1.75
does bound what has actually been produced -- but the margin is thin against a quantity
that has risen at every step (random 0.75, 8-bar GA 0.91, 1-bar 1.21, +elitism 1.59).

**Clipping** was added to `track_pareto_progress` as a safety net — one shared chokepoint
rather than ten call sites. A design better than the reference now stops earning credit
on that axis instead of earning unbounded credit. Across every run in this round it never
triggered.

### A separate distortion, left alone deliberately

Every feasible design has displacement ≤ 0.2000 — that is the feasibility limit — but the
reference is 1.00449, so all designs occupy the bottom 20% of that axis while mass spans
88% of its own. **The hypervolume therefore weights mass roughly six times more heavily
than displacement, arbitrarily.** Fixing it means dividing displacement by ~0.2, which
changes the relative importance of your objectives. That is a modelling decision, not a
bug, so it is documented here rather than silently applied.

---

## Step 2 — the tuned GA

### Stage A: elitism × mutation size, seeds 0–3 (screening, n=4)

Mean HV; these two settings were expected to interact, since round 5's 1-bar mutation
may have been compensating for the fact that nothing protected a good design.

| | 1 bar | 2 bars | 4 bars |
|---|---|---|---|
| elitism 0 | 0.4061 | 0.3716 | 0.3231 |
| elitism 2 | 0.4773 | 0.4410 | 0.3655 |
| elitism 4 | 0.5454 | 0.5278 | 0.4186 |
| **elitism 8** | **0.5935** | 0.5355 | 0.4643 |
| elitism 16 | 0.5714 | 0.5396 | 0.4669 |
| elitism 32 | 0.5690 | 0.5007 | 0.4708 |

1 bar is best at **every** elitism level, so round 5's mutation finding was not merely
compensating for missing elitism — the two effects are additive. Elitism rises
monotonically to 8 and flattens after.

### Stage B: everything else, on Stage A's winner, seeds 0–3 (screening, n=4)

| arm | mean HV | vs Stage A winner | p |
|---|---|---|---|
| **select tournament** | **0.6192** | **+0.0256** | **0.036** |
| pop 32 | 0.6050 | +0.0115 | 0.59 |
| select nsga2 | 0.6031 | +0.0096 | 0.76 |
| select sss (Stage A winner) | 0.5935 | — | — |
| keep_parents 8 | 0.5935 | +0.0000 | — |
| pop 128 | 0.5722 | −0.0213 | 0.50 |
| crossover uniform | 0.5142 | −0.0793 | 0.09 |
| crossover two_points | 0.5103 | −0.0832 | 0.038 |
| select rank | 0.1959 | −0.3976 | 0.0002 |

`keep_parents` is inert — pygad ignores it whenever `keep_elitism > 0`, which is why the
delta is exactly zero. Rank selection is catastrophic. Tournament is the only gain, and
at n=4 with p=0.036 it is screening, not proof.

### Stage C: confirmation on held-out seeds 4–11 (n=8)

| arm | n | mean HV | sd |
|---|---|---|---|
| **tuned** (elitism 8, 1 bar, tournament) | 8 | **0.6112** | 0.036 |
| round-5 default (1 bar, no elitism) | 8 | 0.4101 | 0.029 |
| original default (8 bars, no elitism) | 8 | 0.2617 | 0.007 |

```
tuned  - round-5 default    +0.2011   p=0.00001   8/8
tuned  - original default   +0.3495   p<0.00001   8/8
round5 - original default   +0.1484   p<0.00001   8/8
```

Confirmed on seeds never used for selection. Applied as the new default in
`genetic_algorithm.py`.

---

## Step 3 — every method on the repaired metric

64,000 NFE, seeds 0–7, each method in its current repo configuration.

| method | n | mean HV | sd | min | max |
|---|---|---|---|---|---|
| **GA tuned (round-6 default)** | 8 | **0.5997** | 0.020 | 0.5772 | 0.6324 |
| IMGA, trained actor | 8 | 0.4074 | 0.019 | 0.3835 | 0.4431 |
| random search | 8 | 0.4008 | 0.014 | 0.3835 | 0.4236 |
| IMGA, untrained actor | 8 | 0.3549 | 0.020 | 0.3285 | 0.3859 |
| synthesis PPO (hv reward, reset 3) | 8 | 0.3000 | 0.010 | 0.2843 | 0.3129 |
| GA original (8 bars, no elitism) | 8 | 0.2620 | 0.007 | 0.2490 | 0.2717 |

```
GA tuned - IMGA trained      +0.1923  p<0.00001  8/8
GA tuned - random search     +0.1989  p<0.00001  8/8
GA tuned - IMGA untrained    +0.2448  p<0.00001  8/8
GA tuned - synthesis PPO     +0.2997  p<0.00001  8/8
GA tuned - GA original       +0.3377  p<0.00001  8/8

synthesis PPO - random search  -0.1008  p<0.00001  0/8
IMGA trained  - IMGA untrained +0.0525  p=0.00134  7/8
IMGA trained  - GA tuned       -0.1923  p<0.00001  0/8
```

**The synthesis PPO loses to random search on every one of eight seeds.** So does the
original GA. Both of the project's hand-built methods were below the trivial baseline,
and the broken metric concealed it.

**Round 5's finding survives.** Stage C's 1-bar no-elitism GA (0.4101, seeds 4–11) sits
essentially on top of IMGA-trained (0.4074, seeds 0–7): a plain GA with no network
anywhere in it still matches the trained transformer. These are different seed sets, so
this is consistent rather than paired — the paired version is `RESULTS_ROUND5.md` §3
(p=0.69) on the old metric.

---

## §5 — the re-normalisation changed a ranking

On the old reference, GA original (0.5311) beat random search (0.4059). On the repaired
one, random search (0.4008) beats GA original (0.2620). The ordering inverted.

This is not a bug in either measurement; it is what re-weighting an objective does.
Doubling the frequency axis' span changes how much a broad, diverse front is worth
relative to a converged one, and random search keeps a far more diverse front than a GA
that has collapsed onto mass and displacement. `PLAN_ROUND6.md` flagged this as a real
risk before the run, and it materialised.

The practical consequence: **any round 1–5 conclusion that depended on the size or sign
of a gap needs re-checking against this reference, not assumed to carry over.** The
conclusions that survive are the ones resting on mechanism rather than on a hypervolume
delta — the operator probe in `RESULTS_ROUND5.md` §1 (edit counts, uniform pointer head)
is metric-independent and stands.

---

## Changes applied to the repo

| file | change |
|---|---|
| `experiments/max_values.npy` | frequency axis 0.879 → 1.75; old file kept as `max_values_random16k.npy` |
| `utils/hypervolume_utils.py` | clip normalised coordinates at 0 in `track_pareto_progress` |
| `optimization/genetic_algorithm.py` | `keep_elitism=8`, `parent_selection_type='tournament'`, `K_tournament=3` (all overridable via params) |
| `optimization/warm_start_ga.py` | `keep_elitism=8` — **extrapolated**, flagged in-file, not measured |

---

## Reproducing

```
# step 1a -- bounds (CPU, 8 workers, ~25 min)
for i in 0..7: python experiments/runner15.py --shard $i --nshards 8
# step 2 -- stages A, B, C in order; B and C read the previous stage's winner
for i in 0..7: python experiments/runner16.py --stage a --shard $i --nshards 8
for i in 0..7: python experiments/runner16.py --stage b --shard $i --nshards 8
for i in 0..7: python experiments/runner16.py --stage c --shard $i --nshards 8
# step 3 -- re-baseline (CPU 12 workers + GPU 4 workers, ~2 h)
for i in 0..11: python experiments/runner17.py --device cpu --shard $i --nshards 12
for i in 0..3:  python experiments/runner17.py --device gpu --shard $i --nshards 4
```

Raw records: `r6a_*` (bounds), `r6b0_*` / `r6b1_*` / `r6b2_*` (stages A/B/C),
`r6c_cpu_*` / `r6c_gpu_*` (re-baseline).

---

## Limitations

- Stages A and B are n=4 screening. Only the Stage C winner was confirmed at n=8, on
  held-out seeds. Any Stage A/B ordering other than the winner is unconfirmed.
- The GA sweep covered elitism, mutation size, selection, crossover and population. It
  did not cover adaptive mutation, crossover probability, or gene-space restriction.
- IMGA was run in its historical configuration (elitism 0, sss selection). It inherits
  the same defects the GA had, so its number here understates what the hybrid could do
  with the tuned settings. That comparison was deliberately not run: changing it would
  make the arm something other than the method rounds 1–5 measured.
- The displacement-axis compression (§1) is unaddressed and still weights mass ~6x more
  heavily than displacement.
- The synthesis PPO was run at one configuration (`reward_mode='hv'`,
  `hv_reset_period=3`), the best known from round 4.

---

## What step 4 would now have to clear

The bar for any learned-operator work is **0.5997**, from a genetic algorithm with three
settings changed. The two structural problems identified in `RESULTS_ROUND5.md` §7.4 are
unchanged: the operator never sees the objectives, so it cannot know which bar to edit;
and its reward makes stopping immediately the optimal policy, which is what it learned.
Not started, per the plan.
