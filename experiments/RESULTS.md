# Measurements for the proposed policy / training changes

> **Round 1.** Several conclusions here were later overturned at n=8 and at the real
> 64,000-evaluation budget. Read [`RESULTS_ROUND2.md`](RESULTS_ROUND2.md) for the
> confirmed results; this file is kept for the diagnostics and the two bug analyses,
> which still stand.

All experiments run on the synthesis PPO (`ppo_rand_weights_state.py` +
`transformer_rand_weights_state.py`) and the repair PPO, using a parameterised copy of the
repo code (`variant.py`, `repair_variant.py`) whose defaults reproduce the current
implementation exactly. Hypervolume is measured against a fixed `max_values.npy`
(16,000-sample random search) so every number is directly comparable.

**Bottom line: of the five changes proposed, none is supported by the evidence as a
performance improvement. One change not originally proposed (the repair reward target)
shows a modest gain. Two genuine implementation bugs were found, but fixing them does not
improve hypervolume. The most important finding is methodological: seed-to-seed variance is
large enough that the project's current 6-run comparisons cannot resolve the effects being
claimed.**

---

## 1. Baseline diagnostics (n=3, 300 epochs x batch 32 = 9,600 evaluations)

| Diagnostic | Result | Reading |
|---|---|---|
| Final hypervolume | 0.3366 | — |
| Policy entropy, final 10 epochs | **3.384** of max 3.912 (ln 50) | 87% of maximum: the policy stays near-uniform |
| Critic explained variance, final 50 epochs | **-0.35** | Worse than predicting the mean; the value baseline is anti-informative |
| GradScaler steps skipped | **0 / 300** | fp16 is *not* silently dropping updates |
| Actor weight drift after 150 epochs | 7.2% of initial norm | It is learning, but slowly |
| Actor KL per epoch (target 0.003) | mean **0.051**, exceeded target in **60/60** epochs | `target_kl` silently caps the actor at *one* update per epoch while the critic gets five |

**Weight conditioning** — designs generated under opposite preference vectors:

| Asked for | mean mass | mean frequency |
|---|---|---|
| light (`[1,0,0,1,1]`) | 2801.8 | 0.4609 |
| stiff (`[0,0,1,1,1]`) | 2770.2 | 0.4509 |
| difference | **-1.1%** | **-2.2% (wrong direction)** |

The policy essentially ignores the preference weights it is conditioned on.

---

## 2. Single-change sweep (n=3, 9,600 evaluations)

| Config | HV | entropy | critic EV | mass resp. | freq resp. |
|---|---|---|---|---|---|
| baseline | 0.3366 +- 0.0021 | 3.384 | -0.35 | 1.6% | 2.9% |
| scaled indices (/49) | 0.3076 +- 0.0139 | 3.023 | -0.59 | 1.1% | 3.0% |
| split encoders | 0.3600 +- 0.0486 | 2.849 | -0.86 | 0.5% | 2.8% |
| embedding | 0.3218 +- 0.0142 | 3.749 | -0.64 | 0.9% | 1.4% |
| entropy bonus 0.01 | 0.3357 +- 0.0041 | 3.517 | -0.34 | 1.6% | 3.3% |
| gamma=1, lambda=1 | 0.3380 +- 0.0028 | 3.093 | -1.00 | 1.6% | 1.0% |
| critic d=64 | 0.3496 +- 0.0090 | 3.421 | **-0.22** | 1.1% | 0.8% |
| critic d=128 x2 | 0.3529 +- 0.0220 | 3.420 | -0.29 | 2.2% | 5.2% |
| actor d=64 | 0.3274 +- 0.0382 | **0.432** | -1.33 | 0.3% | 0.6% |
| actor d=128 x2 | 0.2964 +- 0.0428 | 1.403 | -0.44 | 3.3% | 3.9% |
| fp32 (no amp) | 0.3453 +- 0.0020 | 3.464 | -0.36 | 1.2% | 4.3% |
| looser clip 0.2 / kl 0.03 | 0.3342 +- 0.0070 | 2.739 | -0.46 | 0.7% | 3.5% |

Notable: **enlarging the actor collapses entropy** (3.38 to 0.43 at d=64) and makes results
both worse and far more variable. Enlarging the *critic* is the only change that improves
explained variance.

---

## 3. The variance trap — why section 2 cannot be trusted

Re-running the same configs with **8 seeds instead of 3**:

| Config | n=3 | n=8 | sd (n=8) | p vs baseline |
|---|---|---|---|---|
| baseline | 0.3366 +- 0.0021 | **0.3333** | 0.0237 | — |
| fp32 | 0.3453 +- 0.0020 | **0.3336** | 0.0247 | 0.98 |
| fp32 + critic128x2 | 0.3515 +- 0.0147 | **0.3338** | 0.0236 | 0.97 |
| split encoders | 0.3600 +- 0.0486 | **0.3369** | 0.0563 | 0.87 |

Seeds 0-2 happen to cluster tightly, which made the 3-seed standard deviations (0.002)
understate the true spread (0.024) by more than 10x. **Every apparent winner in section 2
disappears.** The fp32 result of "+0.0087 at 4 sigma" becomes "+0.0003, p=0.98".

fp32 remains worth taking for a different reason: it is **18% faster** (93s vs 114s per
run) at identical quality.

---

## 4. Does it learn at all? (fixed preference vector, n=3)

To remove the random-weight scalarisation as a variable, the policy was trained against a
single fixed preference vector — the easiest possible version of the problem.

| Fixed preference | reward, first 20 epochs | last 20 epochs | change | final entropy |
|---|---|---|---|---|
| mass-only | -0.3134 | -0.3261 | **-0.0126 (worse)** | 3.310 |
| frequency-only | +0.0759 | +0.0918 | +0.0159 | 3.282 |

Over 9,600 evaluations the policy barely improves the single scalar objective it is being
trained on, and in one case gets worse. The random-weight scalarisation is **not** the cause.

---

## 5. Two real implementation bugs

### 5a. Positional encoding is indexed by batch, not by sequence position

`transformer_rand_weights_state.py:82`

```python
def forward(self, x):
    x = x + self.pe[:x.size(0), :]
```

`self.pe` has shape `(max_len, 1, d_model)` — written for `(seq, batch, d)` layout — but
`x` arrives as `(batch, seq, d)`. So `x.size(0)` is the **batch size**. Measured:

- `pe[:8]` has shape `(8, 1, 16)` and broadcasts across the sequence
- all 40 sequence positions receive an **identical** encoding, so *no positional
  information reaches the model*
- two different batch rows receive **different** encodings, so the "position" actually
  encoded is the row index
- the same 30-token sequence placed at batch row 0 vs row 4 yields action distributions
  differing by **0.126 total variation**

`transformer_design_repair.py:142` applies the same class correctly
(`self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)`), so **only the synthesis
models are affected** (`transformer_rand_weights_state.py`,
`transformer_rand_weights_uninformed.py`).

### 5b. The PPO importance ratio is not 1 at the first update

At the first update iteration, before any gradient step, `exp(new_logp - old_logp)` must be
exactly 1.0. Measured over a full rollout:

| Variant | ratio sd | within 1% of 1.0 | outside clip band (0.95-1.05) |
|---|---|---|---|
| repo as-is | 0.448 | 3.6% | **85.9%** |
| + positional encoding fixed | 0.341 | 2.8% | 84.8% |
| + gather at true action position | 0.294 | 3.5% | 80.5% |
| + no dropout during rollout | 0.167 | 11.8% | 57.9% |
| + all of the above (incl. PE dropout) | **0.043** | **83.0%** | **1.3%** |

Three compounding causes:

1. Dropout is active during rollout — in the decoder *and* in `PositionalEncoding`, whose
   `dropout=0.1` default is never overridden anywhere in the repo.
2. `ppo_update` reads `forward(obs)[:, -1, :]` — the last slot of the zero-padded
   77-length sequence, on average **35.5 tokens past** the position where the action was
   actually taken.
3. The PE bug (5a) makes the output depend on batch row, and rollout batching differs from
   update batching.

**Consequence:** with `clip_ratio=0.05`, 86% of samples are outside the clip band before
learning begins, so clipping is triggered by numerical noise rather than by policy change.
This also explains the KL result in section 1 — the measured KL of 0.051 is mostly noise,
which is why the actor's update loop stops after one iteration every single epoch.

### 5c. But fixing them does not improve hypervolume (n=8, 9,600 evals)

| Config | HV | entropy | mass resp. | freq resp. | p vs baseline |
|---|---|---|---|---|---|
| baseline (repo as-is) | 0.3333 +- 0.0237 | 3.443 | 1.5% | 2.8% | — |
| PE fix only | 0.3540 +- 0.0564 | 3.349 | 1.2% | 3.3% | 0.36 |
| all ratio fixes | 0.3258 +- 0.0203 | 3.450 | 1.7% | 5.7% | 0.51 |
| all fixes + 5 update iters | 0.3148 +- 0.0147 | 1.878 | **5.2%** | **13.9%** | 0.09 |

The last row is the one qualitative success: with a valid ratio *and* the update loop no
longer truncated, the policy finally responds to its preference weights (frequency response
2.8% to 13.9%). That does not translate into hypervolume.

---

## 6. Budget sensitivity — sections 1-5 do not transfer to the real configuration

Everything above uses 9,600 evaluations. `main.py` uses **64,000**. Re-run at the real
budget (n=3):

| Config | HV | sd | individual runs | final entropy |
|---|---|---|---|---|
| baseline (repo as-is) | **0.4442** | 0.1074 | 0.5665, 0.3654, 0.4005 | 1.225 |
| all fixes + 5 update iters | 0.3664 | 0.0173 | 0.3809, 0.3472, 0.3712 | 0.701 |
| random search | 0.4037 | 0.0070 | 0.4064, 0.3958, 0.4090 | — |

Two things change completely at the real budget:

1. **Entropy does collapse** — to 0.052, 0.068 and 3.556 across the three baseline seeds.
   The behaviour is *bimodal*: two seeds converge to a near-deterministic policy, one never
   collapses at all. At 9,600 evaluations entropy was still 3.4 in every seed, so the
   collapse was invisible there.
2. **PPO's relationship to random search flips.** At 9,600 evals random search
   significantly beat every PPO variant (0.3795 vs 0.3333, p=0.0007). At 64,000 evals PPO
   is nominally ahead (0.4442 vs 0.4037), but with sd=0.107 the difference is not
   significant (p=0.58).

---

## 7. Repair reward target (n=3, 150 epochs x batch 32, compared at common NFE)

The one change with a positive result. Reward for each single-variable edit:

| Reward | HV | sd | vs current |
|---|---|---|---|
| `domination` — beat the parent (current) | 0.2888 | 0.0108 | — |
| `front` — join the global Pareto front | 0.2928 | 0.0054 | +1.4% |
| `hv` — hypervolume contribution | **0.3004** | 0.0041 | **+4.0%** |

Direction matches the hypothesis and the ordering is monotone, but the margin (+0.0116) is
only about 1-2 standard deviations at n=3. Worth re-running at n=8 before adopting.

---

## 8. Status of each proposed change

| # | Proposal | Verdict | Evidence |
|---|---|---|---|
| 1 | Embedding + separate weight encoder | **Not supported** | embedding 0.3218 and scaled 0.3076 are *worse* than baseline 0.3366; weight response stays ~1-3% under every encoding |
| 2 | Entropy bonus | **Not supported at 9.6k; open at 64k** | no effect at 9.6k (entropy is 87% of max, nothing to prevent); at 64k entropy collapses to 0.05 in 2/3 seeds, so the mechanism is real at the real budget but untested there |
| 3 | gamma=lambda=1 / critic | **Partly supported** | gamma=lambda=1 has no effect (0.3380, p>0.5). Critic EV is genuinely bad (-0.35) and enlarging the critic is the only thing that improves it (d=64 gives -0.22), but the HV gain vanishes at n=8 (p=0.97) |
| 4 | Hypervolume-based repair reward | **Supported, modest** | +4.0%, monotone ordering, needs n=8 confirmation |
| 5 | Drop fp16 | **Supported for speed only** | 0 skipped steps, so the stated rationale was wrong. No HV effect (p=0.98), but 18% faster |

### On network size

- **Critic larger — worth doing, for the right reason.** d=64 is the only change that
  improves explained variance (-0.35 to -0.22). It does not improve HV at n=8, so treat it
  as repairing a broken component rather than as a performance win. EV remains negative at
  every size tried, so a wider critic is not sufficient — the target may simply not be
  predictable from a partial design.
- **Actor larger — actively harmful.** d=64 collapses entropy to 0.43 and d=128 x2 gives
  the worst HV measured (0.2964) with the largest spread. Extra actor capacity is spent
  fitting noise, consistent with section 5b: while the ratio is dominated by numerical
  noise, more capacity means more ability to fit that noise. Do not enlarge the actor
  before the ratio is valid, and if you do, pair it with an entropy bonus (0.05 restores
  entropy to 3.80 but still does not recover HV: 0.3239).

---

## 9. Recommendations

1. **Fix the seed budget before anything else.** With sd = 0.024 at 9.6k and **0.107** at
   64k evaluations, 6 runs per method cannot resolve differences smaller than roughly
   0.09 HV at the real budget. Several conclusions in this file reversed between n=3 and
   n=8. Any claim that method A beats method B needs enough seeds to support it, and the
   hypervolume plots should show per-seed spread rather than only the median band.
2. **Fix the positional encoding bug** (5a) regardless of its HV effect. As written, the
   synthesis policy receives no positional information at all and its output depends on
   batch layout — that is not the architecture the write-up describes.
3. **Fix the ratio mismatch** (5b) for the same reason: the PPO objective does not
   currently measure what it is supposed to measure. Expect no HV improvement on its own.
4. **Do not adopt changes 1, 2, 3 or 5 as performance improvements.** Take fp32 for the
   18% speedup and the simpler code.
5. **Re-test the repair reward at n=8**; adopt if the +4% holds.
6. **Re-test the entropy bonus at 64,000 evaluations**, where the collapse it targets
   actually occurs. This is the most promising untested item.

## Reproducing

```
python experiments/diag.py        # section 1
python experiments/sweep.py       # section 2
python experiments/confirm.py     # section 3
python experiments/fixedw.py      # section 4
python experiments/pe_bug.py      # section 5a
python experiments/ratio_fixed.py # section 5b
python experiments/final.py       # section 5c
python experiments/fullbudget.py  # section 6
python experiments/repair_exp.py  # section 7
```

Raw numbers are in the sibling `*.json` files. `variant.py` holds the parameterised
actor/critic and PPO loop; its `DEFAULTS` reproduce the repo exactly, and each experiment
flips one key. Requires `pygad` and `pymoo` in addition to the repo's dependencies.
