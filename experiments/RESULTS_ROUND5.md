# Round 5 — the learned operator is a mutation rate, and the metric has been lying

Follow-up to `RESULTS.md` (round 1), `RESULTS_ROUND2.md`, `RESULTS_ROUND3.md` and
`RESULTS_ROUND4.md`. ~120 runs plus five offline probes of the trained checkpoints.

Two findings, one of which invalidates part of every previous round.

---

## Headline

1. **The hybrid's advantage is its mutation rate, not its network.** `genetic_algorithm.py`
   and `intelligent_mutation_ga.py` are identical apart from one argument. The plain GA
   redrew ~8 of 72 bars per offspring; the repair actor redraws 1. Setting
   `mutation_num_genes=1` on the plain GA **matches the trained operator exactly**:
   0.8175 vs 0.8089, **p=0.69**, paired on 8 seeds through the repo's own code.
2. **The trained actor's pointer head is exactly uniform.** Its KL from uniform is
   0.0000 to four decimals across five independently trained checkpoints. Training
   taught the *stop* head to fire immediately (P(stop) 0.43 → 0.90–0.99, mean edits
   1.69 → 1.05) and taught the bar-selection head nothing at all.
3. **Round 4's "the RL earns its keep" reproduces and is still fully explained by (1).**
   Trained beats untrained by +0.0837 (p=0.016, 7/8). A GA with no network anywhere in
   it beats the untrained operator by +0.0923 (p<0.0001, 8/8) and ties the trained one.
4. **`max_values.npy` is 80% too low on frequency, and hypervolume is not clipped.**
   Any design better than the reference gets a *negative* normalised coordinate, and
   pymoo then integrates volume outside the unit cube. The old 8-bar GA escaped the box
   on 1 design in 61,857; a 1-bar GA escapes on **46%** of its feasible designs. Reported
   scores above 1.0 are routine once elitism is on.
5. **On a metric that is not broken, elitism matters more than everything else.**
   `keep_elitism=8` is worth **+0.166** on top of the mutation fix — more than every
   method change in rounds 1–4 combined.
6. **The honest baseline is now ~0.79, above every RL number ever measured here.** Four
   rounds of reinforcement learning were compared against a GA crippled by two settings.

---

## 1. What the mutation operator actually does

Probing the five round-4 `DOM_DES` checkpoints directly, on 300–400 designs each:

| operator | P(stop) | mean edits | median | pointer KL from uniform | pointer logit sd |
|---|---|---|---|---|---|
| random init | 0.431 | 1.69 | 1 | 0.0003 | 0.119 |
| trained t0 | 0.902 | 1.12 | 1 | **0.0000** | 0.006 |
| trained t1 | 0.962 | 1.07 | 1 | **0.0000** | 0.003 |
| trained t2 | 0.993 | 1.01 | 1 | **0.0000** | 0.031 |
| trained t3 | 0.968 | 1.03 | 1 | **0.0000** | 0.003 |
| trained t4 | 0.982 | 1.02 | 1 | **0.0000** | 0.003 |

For reference the plain GA redrew 8 bars, always, and a uniform bar choice would give
KL 0 and max probability 1/72 = 0.0139 — which is what the trained actors output.

The pointer-query weight norm barely moves during training (2.33 → 2.20–2.38), so this
is not weight decay. The logits themselves have gone flat: the head is functionally
constant with respect to its input.

### Why it collapses

`calculate_step_reward` (`design_repair.py:305`) pays

```
-1.01  infeasible
-0.51  parent dominates child
-0.01  mutually non-dominated
+0.99  child dominates parent      <- rare, for a random single-bar edit
```

Expected reward per edit is firmly negative. The rollout loop always edits *before* it
can stop, so one edit is the floor, and the return-maximising policy is to take that
floor and quit. That is precisely what training produces. Meanwhile every bar is about
equally bad to touch, so the pointer head receives no consistent gradient and drifts to
uniform. **The reward makes inaction optimal, and the actor learned to do nothing as
fast as possible.** Deployed as a GA mutation operator, doing nothing quickly *is* a
small mutation rate, which is why it looked like a win.

---

## 2. Mutation size explains every hybrid number in rounds 1–4

Same GA, same population, generations, fitness, selection, crossover, elitism —
`mutation_num_genes` swept. 64,000 NFE, 4 seeds (`runner11.py`, raw metric, see §4):

| arm | mean HV | sd |
|---|---|---|
| 8 bars (repo default) | 0.5274 | 0.012 |
| 4 bars | 0.6430 | 0.010 |
| 2 bars | 0.7395 | 0.014 |
| geometric ~1.7 edits (mimics untrained actor) | 0.7670 | 0.022 |
| 1 bar | 0.8082 | 0.029 |
| uniform 1 edit | 0.8409 | 0.062 |

Round 4's numbers at the same budget: plain GA 0.5311, IMGA untrained 0.7301, IMGA
trained 0.7906. The 8-bar arm reproduces the plain-GA number (0.5274 vs 0.5311), which
validates the harness. **Every hybrid result lands where its measured edit count
predicts, and nowhere else.**

---

## 3. The paired confirmation, through the repo's own functions

`runner12.py` calls `run_genetic_algorithm` and `run_intelligent_mutation_ga` directly,
same 8 seeds, cycling the round-4 checkpoints:

| arm | n | mean HV | sd |
|---|---|---|---|
| **GA, 1 bar (new default)** | 8 | **0.8175** | 0.021 |
| IMGA, trained actor | 8 | 0.8089 | 0.051 |
| IMGA, untrained actor | 8 | 0.7252 | 0.033 |
| GA, 8 bars (old default) | 8 | 0.5195 | 0.016 |

```
GA 1 bar   - IMGA trained     +0.0086   p=0.6940   5/8    <- indistinguishable
GA 1 bar   - IMGA untrained   +0.0923   p<0.0001   8/8
IMGA trained - IMGA untrained +0.0837   p=0.0157   7/8    <- round 4's effect, reproduced
GA 1 bar   - GA 8 bars        +0.2980   p<0.0001   8/8
```

Round 4's effect is real and reproduces. It is also entirely reproduced by an integer.

---

## 4. The hypervolume metric is broken above the reference point

`max_values.npy` holds the per-objective best of a 16,000-sample random search. Every
hypervolume in rounds 1–5 divides by it. `track_pareto_progress` does not clip, so a
design better than the reference produces a negative coordinate and pymoo integrates
volume *outside* the unit cube.

Best values found in this round's sweep against that reference:

| objective | best found | reference | |
|---|---|---|---|
| mass (min) | 444.59 | 3476.55 | — |
| max displacement (min) | 0.0609 | 1.0045 | — |
| **frequency (max)** | **1.5850** | **0.8793** | **reference is 80% too low** |

Re-measuring with coordinates clipped at 0 (`runner14.py`, 4 seeds):

| arm | HV raw | **HV clipped** | sd | designs outside box |
|---|---|---|---|---|
| 8 bars, no elitism (old default) | 0.5274 | **0.5223** | 0.014 | 0.0% |
| 4 bars, no elitism | 0.6430 | 0.5719 | 0.006 | 1.2% |
| 2 bars, no elitism | 0.7395 | 0.6041 | 0.009 | 19.4% |
| 1 bar, no elitism (new default) | 0.8082 | 0.6200 | 0.019 | 46.3% |
| **1 bar + elitism 8** | 1.1812 | **0.7856** | 0.013 | 35.4% |
| 1 bar + nsga2 + elitism 8 | 1.2002 | 0.7748 | 0.008 | 34.0% |
| 1 bar + elitism 8, pop 128 | 1.1388 | 0.7614 | 0.017 | 36.7% |

Paired against the old default on the clipped metric, all 4/4 seeds:

```
4 bars            +0.0496  p=0.0037
2 bars            +0.0818  p=0.0021
1 bar             +0.0977  p=0.0032
1 bar + elitism 8 +0.2633  p=0.0001
```

Two consequences:

- **The mutation-rate effect is real but ~3x smaller than the raw metric says**
  (+0.098 clipped against +0.281 raw). The direction and significance survive; the
  magnitude does not.
- **A hypervolume comparison is only valid between arms with similar escape fractions.**
  The old GA (0.0%) against the synthesis PPO (near 0%) was fine. The old GA against
  IMGA (~20–46%) was not, and that is round 4's headline comparison.

Clipping is the conservative repair — a design better than the reference simply stops
earning credit on that axis. The correct repair is to rebuild `max_values.npy` from
designs that actually exist and re-measure. That has not been done.

Note that `to_hv` in `ppo_rand_weights_state.py` *does* clip, for the reward only. So
the synthesis PPO has been trained with a reward blind to any improvement past the
reference, while being scored with a metric that pays for it.

---

## 5. Elitism, not the mutation rate, is the biggest single lever

With `keep_elitism=0` and `keep_parents=0`, nothing protects a good design: every
offspring replaces a parent, so an unlucky mutation destroys accumulated progress. That
is why the GA was so sensitive to mutation size in the first place — the mutation fix
was compensating for missing elitism.

On the clipped metric, `keep_elitism=8` adds **+0.166** on top of the 1-bar mutation
(0.6200 → 0.7856). pygad already warns that `parent_selection_type="sss"` with
`keep_parents=0` is incoherent; switching to `nsga2` selection did not help further
(0.7748, within noise), and neither did doubling the population (0.7614).

Only three settings were tried. This is the least-explored, highest-return area in the
project and it costs 70 seconds a run.

---

## 6. Corrections to earlier rounds

| round | claim | status |
|---|---|---|
| 4 headline 2 | Trained repair operator beats a random one, +8.6% p=0.021 | **Reproduces, wrong cause.** The gain is the edit count falling 1.69 → 1.05; a GA with no network ties it (p=0.69) |
| 4 headline 6 | IMGA + trained operator, 0.7906, is the best configuration measured | **Superseded.** Inflated by an unclipped metric; 0.6200 clipped, and a plain GA with elitism reaches 0.7856 |
| 4 §6 rec 1 | Make the hybrid the headline method, report the operator ablation with it | **Withdrawn.** The hybrid is a 1-bar mutation GA with a transformer that outputs a uniform distribution |
| 4 §6 rec 3 | Keep the domination reward for the operator | **Wrong lever.** That reward is why the actor learned to stop immediately and why the pointer head has no gradient |
| 4 §1 | Hybrid beats synthesis PPO at equal NFE | **Direction holds, comparison invalid.** The two arms escape the reference box at very different rates |
| 1–5 | All cross-method hypervolume comparisons | **Valid only between arms with similar escape fractions** (§4) |

Round 4's §2a observation-mismatch bug and §2a positional-encoding bug are genuine and
are now fixed in the repo. Neither is worth measurable hypervolume on its own.

Round 4's own headline was additionally selected as the best of four operator arms
sharing one baseline, and did not replicate on actors 5–9 (pooled +0.019, p=0.37, 6/10);
its 64,000-NFE confirmation reused only the five selected actors (`runner10.py:58`).

---

## 7. Recommendations

1. **Rebuild `max_values.npy` and re-measure everything.** Until the reference bounds
   the designs being produced, the headline metric rewards escaping it and no
   cross-method number is trustworthy. Clip as an interim measure.
2. **Sweep the GA properly.** Elitism level, selection operator, population/generation
   split, crossover, adaptive mutation. Three settings bought +0.26; the space is
   essentially untouched and runs cost 70 seconds.
3. **Re-baseline before any further RL work.** Every RL claim in rounds 1–5 was measured
   against a GA with two bad settings. The honest baseline is ~0.79 clipped, above every
   RL result this project has produced.
4. **If the learned operator is kept, fix the two things that make learning impossible:**
   - It never sees the objectives. `intelligent_mutation_ga.py` feeds it the 72-bar
     design alone, so it cannot know whether the truss is too heavy or too floppy and
     therefore cannot choose which bar to change. The parent's objectives are available
     free at mutation time.
   - Its reward makes inaction optimal (§1). Score the design the episode ends on, or its
     contribution to the archive — not per-step domination against the parent.
   Without both, the pointer head has no gradient, which is what §1 measures.
5. **Report the synthesis PPO as a baseline.** It is not competitive, and `reward_mode`
   / `hv_reset_period` tuning does not change that.

---

## Reproducing

```
# track F -- mutation-size sweep, standalone harness      (CPU, 8 workers, ~5 min)
for i in 0..7: python experiments/runner11.py --shard $i --nshards 8
# track G -- paired confirmation through the repo's own functions (CPU, 8 workers, ~1 h)
for i in 0..7: python experiments/runner12.py --shard $i --nshards 8
# track H -- GA settings on top of the 1-bar default      (CPU, 8 workers, ~10 min)
for i in 0..7: python experiments/runner13.py --shard $i --nshards 8
# track I -- the same sweep with raw + clipped metrics    (CPU, 8 workers, ~10 min)
for i in 0..7: python experiments/runner14.py --shard $i --nshards 8
```

Raw records: `r5f_shard_*.jsonl`, `r5g_*`, `r5h_*`, `r5i_*`. The offline checkpoint
probes (edit counts, pointer uniformity, weight norms) are one-off scripts, not runners.

**Limitations.** Tracks F, G and H report the raw metric and so overstate every arm that
escapes the reference box; track I is the corrected measurement and is n=4. Track G's
headline comparison (GA 1 bar vs IMGA trained) is between two arms with similar escape
rates, so it is approximately fair, but it has not been re-run clipped. Track I tried
three GA settings, not a sweep.

**Changes applied to the repo this round.** `mutation_num_genes=1` in
`genetic_algorithm.py` and `warm_start_ga.py`; `hv_reset_period` in
`ppo_rand_weights_state.py` (default 3); `obs_mode='design'` in `design_repair.py`; the
positional-encoding transpose in the `transformer_design_repair.py` Critic. Elitism has
**not** been changed — §5 is three data points, not a tuned setting.
