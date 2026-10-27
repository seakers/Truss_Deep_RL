#!/bin/bash
##############################################################################
# Submit the full sweep pipeline:
#   1. random_search.slurm  -> results/sweep_shared/randomsearch/max_values.npy
#   2. tune.slurm (array)   -> afterok on #1, writes tune_results/sweep_<jobid>.csv
#   3. rank.slurm           -> afterok on #2, writes the ranked CSV
#
# Usage:
#   bash run_sweep.sh              # full chain (RS + sweep + rank)
#   bash run_sweep.sh --skip-rs    # reuse existing max_values (reference behavior)
##############################################################################
set -euo pipefail
mkdir -p logs tune_results

SKIP_RS=0
[[ "${1:-}" == "--skip-rs" ]] && SKIP_RS=1

MAXVALS="results/sweep_shared/randomsearch/max_values.npy"

if [[ $SKIP_RS -eq 1 ]]; then
  # Reference behavior: submit the array directly against an existing max_values [5]
  if [[ ! -f "$MAXVALS" ]]; then
    echo "ERROR: --skip-rs given but $MAXVALS does not exist."
    echo "       Run without --skip-rs to generate it first."
    exit 1
  fi
  echo "Reusing existing max_values: $MAXVALS"
  TUNE_JOB=$(sbatch --parsable slurm/tune.slurm)
  echo "Submitted sweep array: job $TUNE_JOB"
else
  RS_JOB=$(sbatch --parsable slurm/random_search.slurm)
  echo "Submitted Random Search gate: job $RS_JOB"

  TUNE_JOB=$(sbatch --parsable \
    --dependency=afterok:"$RS_JOB" \
    slurm/tune.slurm)
  echo "Submitted sweep array: job $TUNE_JOB (starts after $RS_JOB succeeds)"
fi

RANK_JOB=$(sbatch --parsable \
  --dependency=afterok:"$TUNE_JOB" \
  --export=ALL,SWEEP_JOB_ID="$TUNE_JOB" \
  slurm/rank.slurm)
echo "Submitted ranking job: job $RANK_JOB (starts after $TUNE_JOB array completes)"

echo ""
echo "Watch progress:   squeue -u \$USER"
[[ $SKIP_RS -eq 0 ]] && echo "RS log:           logs/rs_${RS_JOB}.out"
echo "Sweep logs:       logs/tune_${TUNE_JOB}_*.out"
echo "Rank log:         logs/rank_${RANK_JOB}.out"
echo "Raw results:      tune_results/sweep_${TUNE_JOB}.csv"
echo "Ranked results:   tune_results/sweep_${TUNE_JOB}_ranked.csv"