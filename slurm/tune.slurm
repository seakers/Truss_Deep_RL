#!/bin/bash
#SBATCH --job-name=truss_tune
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/tune_%A_%a.out
#SBATCH --error=logs/tune_%A_%a.err
#SBATCH --array=0-17             # = (#LR * #CLIP * #KL * #MB) combos - 1; adjust below
##############################################################################
# Hyperparameter sweep over the NON-transfer PPO methods, iso-NFE budget.
# One array task = one hyperparameter combo, evaluated on all swept methods.
#
# Submit (with dependency) via run_sweep.sh. Do NOT submit this alone unless
# max_values already exists at the MAXVALS path below.
##############################################################################
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs tune_results

module purge
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.4.0
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Shared normalization from the Random Search gate ----
MAXVALS="results/sweep_shared/randomsearch/max_values.npy"
if [[ ! -f "$MAXVALS" ]]; then
  echo "ERROR: $MAXVALS missing. Run random_search.slurm first."
  exit 1
fi

# ---- Methods to sweep (NON-transfer PPO only) ----
METHODS=("Design Synthesis PPO" "Design Repair PPO")

# ---- Sweep configuration ----
# Iso-NFE: keep total evaluations constant across combos.
NFE_BUDGET=3200        # must be divisible by every value in BATCHES
LRS=(0.0005 0.0003 0.0001 0.00003 0.00001)   # 5
CLIPS=(0.1 0.2 0.3)                          # 3
KLS=(0.003 0.01)                             # 2
BATCHES=(16 32)                              # 2

NL=${#LRS[@]}; NC=${#CLIPS[@]}; NK=${#KLS[@]}; NB=${#BATCHES[@]}
idx=$SLURM_ARRAY_TASK_ID
MB=${BATCHES[$(( idx % NB ))]};  idx=$(( idx / NB ))
KL=${KLS[$(( idx % NK ))]};      idx=$(( idx / NK ))
CLIP=${CLIPS[$(( idx % NC ))]};  idx=$(( idx / NC ))
LR=${LRS[$(( idx % NL ))]}

EPOCHS=$(( NFE_BUDGET / MB ))

CSV="tune_results/sweep_${SLURM_ARRAY_JOB_ID}.csv"

# ---- Loop methods, report each individually ----
for METHOD in "${METHODS[@]}"; do
  MKEY=$(echo "$METHOD" | tr ' ' '_')
  TAG="${MKEY}_lr${LR}_clip${CLIP}_kl${KL}_mb${MB}"
  RUN_DATE="tune/${SLURM_ARRAY_JOB_ID}/${TAG}"
  LOG="tune_results/${TAG}.log"

  echo "----------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] $METHOD  (combo ${SLURM_ARRAY_TASK_ID}, epochs=$EPOCHS)"

  T0=$(date +%s)
  set +e
  python main.py \
    --methods "$METHOD" \
    --num-runs 1 \
    --num-epochs "$EPOCHS" \
    --mini-batch-size "$MB" \
    --learning-rate "$LR" \
    --clip-ratio "$CLIP" \
    --target-kl "$KL" \
    --max-values-path "$MAXVALS" \
    --date-str "$RUN_DATE" \
    --no-overlap-constraint
    > "$LOG" 2>&1
  RC=$?
  set -e
  WALL=$(( $(date +%s) - T0 ))

  # NOTE: truss main.py prints "Best HV for <method> across all runs: <float>".
  # If your PPO trainers instead print "Best HV this run: <float>", swap the grep.
  BEST_HV=$(grep -oE 'Best HV for .* across all runs: [0-9.]+' "$LOG" \
            | tail -1 | grep -oE '[0-9.]+$' || echo "NA")
  [[ -z "$BEST_HV" ]] && BEST_HV="NA"

  # ---- Append one row to the shared CSV under a lock ----
  (
    flock 200
    if [[ ! -s "$CSV" ]]; then
      echo "task,method,nfe_budget,num_epochs,mini_batch_size,learning_rate,clip_ratio,target_kl,best_hv,wall_seconds,exit_code" > "$CSV"
    fi
    echo "${SLURM_ARRAY_TASK_ID},${METHOD},${NFE_BUDGET},${EPOCHS},${MB},${LR},${CLIP},${KL},${BEST_HV},${WALL},${RC}" >> "$CSV"
  ) 200>"${CSV}.lock"

  echo "[$(date +%H:%M:%S)] $METHOD done: best_hv=${BEST_HV}  wall=${WALL}s  rc=${RC}"
done

echo "================================================================"
echo "Task ${SLURM_ARRAY_TASK_ID} complete. Rows appended to $CSV"
echo "================================================================"