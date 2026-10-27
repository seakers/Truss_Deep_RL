#!/bin/bash
#SBATCH --job-name=truss_rank
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=short
#SBATCH --output=logs/rank_%j.out
#SBATCH --error=logs/rank_%j.err
##############################################################################
# Finalize step: rank the sweep CSV by best_hv (descending) once the whole
# array has finished. Submitted as an afterok dependency of the sweep array.
#
# Requires SWEEP_JOB_ID exported into the environment at submit time
# (run_sweep.sh does this). Writes a ranked copy next to the raw CSV.
#
# best_hv is column 9:
#   task,method,nfe_budget,num_epochs,mini_batch_size,
#   learning_rate,clip_ratio,target_kl,best_hv,wall_seconds,exit_code
#
# Rows whose best_hv is "NA" (failed/unparsed) sort to the bottom:
# sort -g treats non-numeric NA as 0, and -r puts them last.
##############################################################################
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${SWEEP_JOB_ID:?SWEEP_JOB_ID not set — submit via run_sweep.sh}"

CSV="tune_results/sweep_${SWEEP_JOB_ID}.csv"
RANKED="tune_results/sweep_${SWEEP_JOB_ID}_ranked.csv"

if [[ ! -s "$CSV" ]]; then
  echo "ERROR: $CSV not found or empty. Did any array task write rows?"
  exit 1
fi

HEADER=$(head -1 "$CSV")

{
  echo "$HEADER"
  tail -n +2 "$CSV" | sort -t, -k9 -gr
} > "$RANKED"

echo "Ranked results written to: $RANKED"
echo ""
echo "Top 10 combos by best_hv:"
column -s, -t "$RANKED" | head -11   # header + 10 rows

# Per-method winners (handy when sweeping 2 methods at once)
echo ""
echo "Best combo per method:"
for M in "Design Synthesis PPO" "Design Repair PPO"; do
  BEST=$(awk -F, -v m="$M" 'NR>1 && $2==m' "$RANKED" | head -1)
  echo "  ${M}: ${BEST:-<no rows>}"
done