# Round 4 — the hybrid wins, the RL earns its keep, and round 3's mechanism was wrong

Follow-up to `RESULTS.md` (round 1), `RESULTS_ROUND2.md` (round 2) and
`RESULTS_ROUND3.md` (round 3). ~190 runs, 13 parallel workers, ~55 core-hours.

Paired t-tests throughout: seeds are shared across arms by construction, so pairing is the
correct test. Hypervolume is measured against the same fixed `max_values.npy` used in every
previous round, so all four rounds' numbers are directly comparable.

---

## Headline

1. **The synthesis PPO was never the right method.** At an identical 64,000-evaluation
   budget, Intelligent Mutation GA with an *untrained* operator reaches **0.7301**, against
   **0.5526** for round 3's celebrated batch-archive PPO and **0.4059** for random search.
   No previous round compared the two families at equal NFE; three rounds of tuning went
   into the arm that loses by 24%.
2. **The reinforcement learning does add value — round 2's central claim is overturned.**
   A trained repair operator beats a random one by **+8.6% (p=0.021)** at 16,000 NFE and by
   **+0.0474 (p=0.019, 6/6 seeds)** at 64,000 NFE. Round 2 concluded the opposite
   (−5.3%, p=0.06) from **one** trained actor reused across 8 GA seeds; this round trains
   ten independent actors and treats the actor as the random variable it is.
3. **A third implementation bug explains the reversal.** `design_repair.py` rolls the policy
   out on a 72-token design but runs `ppo_update` on a 77-token design+objectives+constraint
   state, so the PPO ratio is evaluated at states the policy never visited — and
   `intelligent_mutation_ga.py` then deploys it on 72 tokens again. Making the three agree
   is worth +0.0211 on its own.
4. **Round 3's explanation of its own result was wrong.** It attributed the batch-archive
   win to the reset making the reward learnable by the critic (EV −0.065 vs −3.5e6). But
   resetting every **5** epochs beats resetting every epoch (**0.6023 vs 0.5526**, p=0.0015,
   8/8 seeds) while its critic diverges to EV ≈ −6e6. Critic learnability is a correlate,
   not the cause.
5. **Round 3's headline does confirm at n=16**: batch archive 0.5526 ± 0.023 vs baseline
   0.4156 ± 0.085, **+0.1369, p<0.0001, winning 14 of 16 seeds**. It is a real effect. It is
   just an improvement to the wrong family.
6. **The best configuration measured across all four rounds** is IMGA with a repair operator
   trained under the domination reward and the observation fix: **0.7906 ± 0.031** at 64,000
   NFE, and **0.7438** at 16,000 NFE — the latter beating every synthesis-PPO arm using a
   quarter of the budget.

---

## 1. The comparison nobody had run: both families at 64,000 NFE

Every hybrid number in rounds 1–3 was measured at 16,000 NFE and every synthesis number at
64,000 NFE, so the two were never comparable. They are here.

| method @ 64,000 NFE | n | mean HV | sd |
|---|---|---|---|
| **IMGA, trained operator (DOM_DES)** | 12 | **0.7906** | 0.031 |
| IMGA, untrained operator | 6 | 0.7301 | 0.025 |
| synthesis PPO, reset=5 (round 4 best) | 8 | 0.6023 | 0.021 |
| synthesis PPO, batch archive (round 3 best) | 16 | 0.5526 | 0.023 |
| plain GA | 6 | 0.5311 | 0.009 |
| synthesis PPO, baseline (repo as-was) | 16 | 0.4156 | 0.085 |
| random search | 8 | 0.4059 | 0.010 |

The gap is not close. The worst hybrid seed (0.6988) beats the best synthesis-PPO seed under
round 3's configuration (0.5800). Round 3's recommendation — "adopt the batch-local HVI
reward as the default synthesis configuration" — optimises a method that loses to a plain GA
with a better mutation operator.

**Caveat on the accounting.** NFE counts calls to `TrussProblem.evaluate`. IMGA's mutation
operator makes several single-variable edits per offspring without evaluating any of them,
so it spends more actor forward passes and more wall-clock per NFE than the synthesis PPO
does. If the budget that matters is finite-element evaluations — which is the metric the
project uses throughout — the comparison above is the right one. If it is wall-clock, the
hybrid's advantage shrinks but does not close.

---

## 2. The reinforcement learning does contribute — with the observation bug fixed

### 2a. A third bug, in the repair pipeline

`design_repair.py:191` samples actions from a **72-token** state (the design alone):

```python
var_log_prob, var_idx, ... = actor.sample_action(
    torch.tensor(current_design, dtype=torch.float32).to(device))
```

but `design_repair.py:225` stores a **77-token** state for the update:

```python
new_obs = new_design + new_obj + list(new_constraint_vals)
```

so `ppo_update` recomputes log-probabilities at states the policy never acted in, and the
importance ratio is meaningless. `intelligent_mutation_ga.py` then calls the trained actor
with 72 tokens again, matching the rollout and not the training. `obs_mode='design'` in
`repair_variant.py` makes rollout, update and deployment agree.

The repair *critic* has the round-1 §5a positional-encoding bug too
(`transformer_design_repair.py:333` applies `PositionalEncoding` without the transpose the
Actor uses), so it is also indexed by batch rather than sequence position. Not tested here.

### 2b. Trained vs untrained operator, with the actor treated as a random variable

Round 2 trained **one** actor per configuration and ran 8 GA seeds against it, which measures
GA noise, not training. Here: 5 independently trained actors per configuration × 2 GA seeds,
paired on the GA seed.

| operator (IMGA @ 16,000 NFE) | n | mean HV | vs random init | p | actor-level p | actors winning |
|---|---|---|---|---|---|---|
| **DOM_DES** — domination reward, obs fixed | 10 | **0.7438** | **+0.0588** | **0.021** | **0.029** | **5/5** |
| DOM_FULL — domination, repo's mismatched state | 10 | 0.7227 | +0.0376 | 0.097 | 0.148 | 4/5 |
| HVG_DES — global HVI reward, obs fixed | 10 | 0.7153 | +0.0303 | 0.096 | 0.119 | 4/5 |
| HVB_DES — batch-local HVI reward, obs fixed | 10 | 0.7025 | +0.0174 | 0.49 | 0.61 | 3/5 |
| random init (untrained) | 10 | 0.6850 | — | — | — | — |

And at the full budget, paired on GA seed:

```
IMGA @ 64,000 NFE, trained (DOM_DES) vs untrained:  +0.0474   p=0.0188   6/6 seeds
```

**Round 2's headline 1 does not survive.** "Training that actor with the current pipeline
adds nothing (−5.3%, p=0.06 — i.e. trending negative)" becomes +8.6%, p=0.021, with every
actor beating a random one.

### 2c. Round 2's "only live path" also fails to replicate

Round 2 §2 reported that the hypervolume reward was the one thing that made a learned
operator beat a random one (+6.2%, p=0.068), and recommended pushing on it. It is the
*worst* of the three rewards tested here:

```
HVG_DES - DOM_DES:  -0.0285  p=0.21
HVB_DES - DOM_DES:  -0.0414  p=0.043     (significantly worse)
```

The plain domination-vs-parent reward — the one the repo already ships — is the best
operator reward. What was broken was the observation, not the reward.

The observation fix alone, holding the reward fixed: `DOM_DES − DOM_FULL = +0.0211,
p=0.19` — the right direction, not resolvable at n=10.

---

## 3. Round 3's result confirms; round 3's explanation does not

### 3a. The headline replicates at n=16

Eight fresh seeds (8–15) were run for both the batch-archive arm and the baseline.

| arm | n | mean HV | sd | vs baseline | p | seeds won |
|---|---|---|---|---|---|---|
| baseline (repo as-was) | 16 | 0.4156 | 0.085 | — | — | — |
| **fixed + hv, batch archive** | 16 | **0.5526** | **0.023** | **+0.1369** | **<0.0001** | **14/16** |

Per-seed, the new seeds behave like the old ones (0.5453, 0.5147, 0.5599, 0.5711, 0.5742,
0.5671, 0.5322, 0.4998). The 5× variance reduction holds. This is a solid result.

Round 3's recommendation 4 — explain the seed-7 critic divergence — is partly answered:
across the eight new seeds the critic's explained variance stayed in −0.05 to −0.08 with **no
divergence at all**, so seed 7 was an isolated event rather than a property of the
configuration. It remains unexplained, but it is not systematic.

### 3b. The reset period has an interior optimum, and it is not 1

RESET_CURVE_TABLE

Resetting every 5 epochs beats round 3's every-epoch configuration by **+0.0427, p=0.0015,
winning 8 of 8 seeds**, and is the best synthesis-PPO number measured in any round.

### 3c. Why the mechanism claim fails

Round 3 argued the reset works because it makes the reward Markovian with respect to the
observation, which the critic's explained variance appeared to confirm (−0.065 for reset=1
against −3.5e6 for the global archive). But the *better* arm has a *worse* critic:

| reset period | mean HV | median critic EV |
|---|---|---|
| 1 | 0.5526 | **−0.056** |
| 5 | **0.6023** | −6.2e6 |

If critic learnability were the mechanism, reset=5 could not win. A reading that fits all
four points on the curve better: the reset period sets how demanding the novelty bar is.
Reset every epoch and a design only has to beat 63 others, so the bar is trivial and the
reward is dense but weak; never reset and the bar becomes unreachable, so the reward is
sparse and its variance explodes (sd 0.121 at "never", 0.112 at 25, against 0.021–0.023 at
1 and 5). Somewhere around 5 epochs — 320 competing designs — the bar is demanding enough to
be informative and loose enough to be hit. The final policy entropy and feasible fraction
move monotonically with the period, consistent with an exploration-pressure knob rather than
a value-function-quality one.

This is a hypothesis fitted after the fact to four points. It is not tested.

---

## 4. What did not work

- **Front membership instead of HVI magnitude** (reset=1, reward = 1 if the design joins the
  batch-local front, else 0) reaches **0.5757 +- 0.024** against 0.5526 for the HVI
  magnitude: **+0.0161, p=0.074, winning 7 of 8 seeds**. Not significant, but it is the
  wrong direction for the magnitude to be carrying information -- a one-bit reward does at
  least as well as the real-valued one. It still loses to reset=5 (-0.0266, p=0.0018, 0/8),
  so the reset period matters far more than what the reward pays out. This is further
  evidence for the novelty-bar reading in section 3c and against the value-learnability one:
  a binary reward is *harder* for a critic to fit precisely, not easier.
- **The batch-local HVI reward as a repair-operator reward**: worse than plain domination
  (−0.0414, p=0.043). The reward that transformed the synthesis PPO is actively unhelpful
  in the hybrid, which is a useful reminder that these two uses of "reward" are not the same
  problem.

---

## 5. Corrections to earlier rounds

| round | claim | status |
|---|---|---|
| 2 §1 | "The hybrid gain is not coming from the RL"; training adds −5.3% | **Overturned.** +8.6%, p=0.021, 5/5 actors, once the rollout/update states agree |
| 2 §2 | HV reward + longer training is the only path to a useful operator | **Does not replicate.** The HV rewards are the worst of the three; domination is best |
| 3 §2 | The batch archive wins because the reward becomes learnable by the critic | **Wrong mechanism.** reset=5 wins with a diverged critic |
| 3 §7.1 | Adopt the batch-local HVI reward as the default synthesis configuration | **Superseded.** reset=5 is better, and the whole family loses to the hybrid at equal NFE |
| 3 §1 | Batch archive +0.1369 over baseline | **Confirmed at n=16**, p<0.0001, 14/16 seeds |

Round 1's two bug analyses (§5a positional encoding, §5b importance ratio) and round 2's
variance warnings all still stand.

---

## 6. Recommendations

1. **Make the hybrid the headline method, and report the operator ablation with it.** The
   defensible claim the measurements support is: *a learned single-variable-edit mutation
   operator beats both a random-weight version of the same operator and every synthesis
   method tested, at equal evaluation budget.* Both halves need the ablation to be credible.
2. **Fix the rollout/update observation mismatch in `design_repair.py`.** It is what stands
   between round 2's "the RL contributes nothing" and this round's "+8.6%, p=0.021". Until
   it is fixed the trained actor is being updated at states it never visits.
3. **Keep the domination reward for the operator.** Two rounds of effort went into replacing
   it with a hypervolume reward on the strength of a p=0.068 result from a single actor; at
   n=10 actors it is worse.
4. **If the synthesis PPO is kept at all, use `hv_reset_period=5`, not 1.** But it is not
   competitive with the hybrid, so the honest framing is that it is a baseline.
5. **Fix the repair critic's positional encoding** (§2a) — same defect as round 1 §5a, in a
   file that was assumed clean because its Actor applies the encoding correctly.
6. **Treat the trained actor as a random variable in every future comparison.** One actor
   across N GA seeds measures GA noise. That single methodological choice is what produced
   round 2's incorrect headline.

---

## Reproducing

```
# track A -- reset period 1/5/25, front membership, n=16 confirmation (GPU, 8 workers)
for i in 0..7: python experiments/runner6.py --shard $i --nshards 8

# track B -- repair-operator study and both families at 64k NFE (CPU, 5 workers)
for i in 0..4: python experiments/runner7.py --shard $i --nshards 5

# track D -- bracketing the reset optimum (GPU, 8 workers)
for i in 0..7: python experiments/runner9.py --shard $i --nshards 8

# track E -- learned operator at 64k NFE, actors 5-9 (CPU, 5 workers)
for i in 0..4: python experiments/runner10.py --shard $i --nshards 5

python experiments/analyze5.py
```

Raw records: `r4a_shard_*.jsonl` (track A), `r4b_shard_*.jsonl` (B), `r4d_shard_*.jsonl` (D),
`r4e_shard_*.jsonl` (E). All runners resume from completed `job_id`s and skip only
*successful* rows, so a failed job is retried on the next launch.

**Hardware note.** The synthesis PPO runs on the GPU (4× faster — it batches 64 designs per
forward). The repair PPO and the GA runs on CPU (2.5× faster than the GPU — they roll out one
design at a time, so the GPU sees nothing but kernel-launch overhead). Track B's first launch
died with `CUDA error: out of memory` when 13 workers shared one card; the CPU rerun is what
is reported. Device is a per-track constant, so no comparison here spans both.
