#!/bin/bash
#SBATCH --job-name=charges
#SBATCH --output=logs/charges_%A_%a.out
#SBATCH --error=logs/charges_%A_%a.err
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=univ5464@ox.ac.uk
#
# ─── Array specification ───────────────────────────────────────────────
# 26278 complexes / 50 per chunk = 526 chunks → --array=0-525
#SBATCH --array=0-525%100
#
# ─── Configuration ─────────────────────────────────────────────────────
PROJECT_DIR="/data/stat-cadd/univ5464/HiQBind"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
MANIFEST="${SCRIPT_DIR}/manifest.tsv"
ENV_DIR="${PROJECT_DIR}/envs/charges_env"
CHUNK_SIZE=50

# ─── Environment setup ────────────────────────────────────────────────
module load Anaconda3
conda deactivate
conda activate "${ENV_DIR}"

# ─── Run ──────────────────────────────────────────────────────────────
mkdir -p "${SCRIPT_DIR}/logs"

echo "=== SLURM array task ${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "Manifest: ${MANIFEST}"
echo "Chunk size: ${CHUNK_SIZE}"
echo "Start time: $(date)"

python "${SCRIPT_DIR}/compute_charges.py" batch \
    "${MANIFEST}" \
    --chunk-id "${SLURM_ARRAY_TASK_ID}" \
    --chunk-size "${CHUNK_SIZE}"

echo "End time: $(date)"
echo "Exit code: $?"
