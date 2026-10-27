#!/bin/bash
#SBATCH --job-name=truss_rs
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=cpu          # Grace: RS is serial NumPy, no GPU needed
#SBATCH -A YOUR_ACCOUNT_HERE     # <-- TAMU Grace allocation (e.g. 1234567890)
#SBATCH --output=logs/rs_%j.out
#SBATCH --error=logs/rs_%j.err
#SBATCH --mail-type=END,FAIL
##############################################################################
# Random Search gate. Produces max_values.npy that every other truss
# method needs for objective normalization. tune.slurm depends on this
# job finishing successfully.
#
# Submit via run_sweep.sh, or standalone:
#   sbatch slurm/random_search.slurm
##############################################################################
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# ---- Environment (TAMU Grace) ----
module purge
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.4.0
source .venv/bin/activate

# Pin BLAS/OMP threads: per-NFE truss physics is serial NumPy; oversubscription hurts.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  GPU: ${CUDA_VISIBLE_DEVICES:-none}"

RS_DATE="sweep_shared/randomsearch"

# A reasonably sized Random Search so max_values is a good normalization bound.
python main.py \
  --methods "Random Search" \
  --num-runs 1 \
  --num-epochs 2000 \
  --mini-batch-size 32 \
  --date-str "$RS_DATE"

MAXVALS="results/${RS_DATE}/max_values.npy"
if [[ -f "$MAXVALS" ]]; then
  echo "SUCCESS: max_values written to $MAXVALS"
else
  echo "ERROR: max_values not found at $MAXVALS — PPO sweeps will fail."
  exit 1
fi
echo "Finished: $(date)"