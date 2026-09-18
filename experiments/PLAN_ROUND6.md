# Round 6 plan — repair the metric, tune the GA, re-baseline

Executes recommendations 1–3 of `RESULTS_ROUND5.md`. Step 4 (fixing the learned
operator's state and reward) is **out of scope** and will not be started; work stops
after step 3 for review.

Hardware: 16 CPU cores, one RTX 5070. GA and repair runs are CPU-bound and shard 8-wide;
the synthesis PPO uses the GPU.

---

## Step 1 — repair the hypervolume metric

**The defect.** `max_values.npy` holds the per-objective best of a 16,000-sample random
search. All ten optimizer files divide raw objectives by it, and
`utils/hypervolume_utils.py:track_pareto_progress` does not clip the result. For the two
minimisation objectives an unexpectedly good design gives a small positive coordinate,
which is harmless. For the maximisation objective the code computes `1 - obj/max_value`,
so any design with frequency above the reference gets a **negative** coordinate and
pymoo integrates volume outside the unit cube. Best frequency actually reachable is
1.585 against a reference of 0.879, so the reference is 80% low and scores above 1.0 are
routine.

**1a. Establish real bounds.** Pool every design evaluated across a deliberately diverse
set of runs — random search, the old 8-bar GA, the 1-bar GA, 1-bar + elitism, and IMGA
with a trained actor, 2 seeds each — and take per-objective extremes over all feasible
designs. Diversity matters more than seed count here: the goal is a bound, not a mean.

**1b. Write the new reference.** Back up the existing file to
`max_values_random16k.npy` first — every historical number in rounds 1–5 is expressed
against it and it must stay recoverable. New values get a margin above the observed
extremes so a modestly better future method does not immediately escape again.

**1c. Add clipping as a safety net.** Clip normalised coordinates at 0 inside
`track_pareto_progress`, the single shared chokepoint. A negative coordinate is always a
metric error, not a good score, so this should never have been reachable. Belt and
braces: 1b makes escape unlikely, 1c makes it harmless.

**1d. Re-measure.** Everything after this point is on the new reference. Numbers will
not be comparable to rounds 1–5 and the writeup must say so on every table.

*Deliverable:* new `max_values.npy`, a backup of the old one, clipping in
`hypervolume_utils.py`, and a conversion note. *Cost:* ~30 min.

---

## Step 2 — tune the GA

Round 5 changed one setting (`mutation_num_genes`, 8 → 1) and probed three others at
n=4. `keep_elitism=8` alone was worth +0.166 on the clipped metric — more than every
method change in rounds 1–4 combined — but three data points is not a tuned setting, so
the elitism default was deliberately left alone. This step does the sweep properly.

**Stage A — the two interacting settings.** Elitism and mutation size are not
independent: the 1-bar mutation was compensating for the fact that nothing protected a
good design. With elitism on, the best mutation size may well move back up.

- `keep_elitism` ∈ {0, 2, 4, 8, 16, 32} (population is 64)
- `mutation_num_genes` ∈ {1, 2, 4}

18 configs × 4 seeds = 72 runs.

**Stage B — the rest, on Stage A's winner.**

- `parent_selection_type` ∈ {sss, nsga2, tournament, rank} — pygad warns on every run
  that `sss` with `keep_parents=0` is incoherent, and that warning has been ignored
- `crossover_type` ∈ {single_point, two_points, uniform}
- population/generation split ∈ {32×2000, 64×1000, 128×500} at fixed 64,000 NFE

~10 configs × 4 seeds = 40 runs.

**Stage C — confirm the winner** at n=8 against the round-5 default, paired on seed.

*Deliverable:* a tuned GA configuration with a paired significance test, and new
defaults in `genetic_algorithm.py` only if Stage C confirms them. *Cost:* ~45 min.

---

## Step 3 — re-baseline every method on the repaired metric

Every RL claim in rounds 1–5 was measured against a GA with two bad settings and a
metric that inflated small-mutation methods. This produces one table where every number
is on the same footing.

Arms, n=8 seeds, 64,000 NFE, all on the new reference:

| arm | device |
|---|---|
| random search | CPU |
| plain GA, old defaults (8 bars, no elitism) | CPU |
| plain GA, round-5 default (1 bar, no elitism) | CPU |
| plain GA, Stage C winner | CPU |
| IMGA, untrained actor | CPU |
| IMGA, trained actor (round-4 DOM_DES checkpoints) | CPU |
| synthesis PPO, `reward_mode='hv'`, `hv_reset_period=3` | GPU |

The synthesis PPO is included so the family that absorbed three rounds of tuning appears
in the same table as the baselines it was never fairly compared against.

*Deliverable:* the table, paired tests against the tuned GA, and `RESULTS_ROUND6.md`.
*Cost:* ~60 min, dominated by the IMGA arms (~10 min/run, 16 runs).

---

## Then stop

Report steps 1–3 and pause for review before touching the learned operator's state or
reward. Nothing in step 4 is started.

---

## What could invalidate this

- **Re-normalising changes the ranking, not just the scale.** Rescaling the frequency
  axis by ~1.8x reweights that objective in the hypervolume. Methods that won by finding
  extreme-frequency designs may lose ground. This is a real possibility, not a
  formality, and the round-5 conclusions must be re-checked against the new numbers
  rather than assumed to carry over.
- **Elitism may reopen the mutation-rate question.** If Stage A's optimum is elitism 8
  with 4 bars rather than 1 bar, then round 5's headline change was itself a workaround
  and the framing needs revising. Stage A is designed to detect exactly this.
- **n=4 in Stages A and B** is screening, not proof. Only the Stage C winner gets n=8,
  and any Stage A/B ordering that is not confirmed at n=8 stays labelled as screening.
  Round 4's central error was treating a 5-sample selected winner as an established
  result; the same trap applies here at every stage.
- **The tuned GA is selected on the same seeds it is scored on** unless Stage C uses
  fresh seeds. It will: Stages A/B use seeds 0–3, Stage C uses seeds 4–11.
