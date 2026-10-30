#!/bin/bash
#SBATCH --job-name=truss_full
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --mail-type=END,FAIL
##############################################################################
# Full production run: all 9 truss methods at the real (large) budget.
#
# main.py runs Random Search first and persists max_values.npy, which the
# GA / PPO / warm-start / transfer methods then load for normalization.
# So this single invocation is self-contained — no separate gate needed.
#
# Submit:  sbatch slurm/run.slurm
##############################################################################
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

module purge
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.4.0
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# =====================================================================
# EDIT THESE after reviewing tune_results/sweep_<jobid>_ranked.csv
# =====================================================================
NUM_EPOCHS=2000           # dramatically larger than the sweep budget
MINI_BATCH_SIZE=64

# Synthesis parameters
SYNTHESIS_LR=0.0001
SYNTHESIS_CLIP_RATIO=0.2
SYNTHESIS_TARGET_KL=0.01
SYNTHESIS_GAMMA=0.999
SYNTHESIS_LAMBDA_GAE=0.9
SYNTHESIS_UPDATE_ITERATIONS=5
# Repair parameters
REPAIR_LR=0.00001
REPAIR_CLIP_RATIO=0.2
REPAIR_TARGET_KL=0.003
REPAIR_GAMMA=0.999
REPAIR_LAMBDA_GAE=0.9
REPAIR_UPDATE_ITERATIONS=5

CHUNK_SIZE=256
NUM_RUNS=10
# =====================================================================

RUN_DATE="full/${SLURM_JOB_ID}"   # outputs -> results/full/<jobid>/

echo "################################################################"
echo "# TRUSS FULL RUN"
echo "#   budget (NFE):  $((NUM_EPOCHS * MINI_BATCH_SIZE))"
echo "#   epochs:        $NUM_EPOCHS   mini_batch: $MINI_BATCH_SIZE"
echo "#   synth_lr=$SYNTHESIS_LR synth_clip=$SYNTHESIS_CLIP_RATIO synth_kl=$SYNTHESIS_TARGET_KL"
echo "#   repair_lr=$REPAIR_LR repair_clip=$REPAIR_CLIP_RATIO repair_kl=$REPAIR_TARGET_KL"
echo "#   runs/method:   $NUM_RUNS"
echo "#   output:        results/${RUN_DATE}/"
echo "#   host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "################################################################"

T0=$(date +%s)
python main.py \
  --num-runs "$NUM_RUNS" \
  --num-epochs "$NUM_EPOCHS" \
  --mini-batch-size "$MINI_BATCH_SIZE" \
  --synthesis-learning-rate "$SYNTHESIS_LR" \
  --synthesis-clip-ratio "$SYNTHESIS_CLIP_RATIO" \
  --synthesis-target-kl "$SYNTHESIS_TARGET_KL" \
  --synthesis-gamma "$SYNTHESIS_GAMMA" \
  --synthesis-lambda-gae "$SYNTHESIS_LAMBDA_GAE" \
  --synthesis-update-iterations "$SYNTHESIS_UPDATE_ITERATIONS" \
  --repair-learning-rate "$REPAIR_LR" \
  --repair-clip-ratio "$REPAIR_CLIP_RATIO" \
  --repair-target-kl "$REPAIR_TARGET_KL" \
  --repair-gamma "$REPAIR_GAMMA" \
  --repair-lambda-gae "$REPAIR_LAMBDA_GAE" \
  --repair-update-iterations "$REPAIR_UPDATE_ITERATIONS" \
  --chunk-size "$CHUNK_SIZE" \
  --date-str "$RUN_DATE" \
  --no-overlap-constraint
T1=$(date +%s)

echo "################################################################"
echo "# Finished in $(( (T1 - T0) / 60 )) min"
echo "# Results in results/${RUN_DATE}/"
echo "################################################################"
