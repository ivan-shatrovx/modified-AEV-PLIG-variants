#!/bin/bash
#SBATCH --job-name=bench_chg
#SBATCH --output=logs/bench_charges_%A_%a.out
#SBATCH --error=logs/bench_charges_%A_%a.err
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
# Adjust based on output from `generate-manifest` (see SLURM array hints).
# Example: 500 complexes / 25 per chunk = 20 chunks → --array=0-19
#SBATCH --array=0-XXX%100
#
# ─── Configuration ─────────────────────────────────────────────────────
PROJECT_DIR="/data/stat-cadd/univ5464/binding_free_energy_benchmark"
SCRIPT_DIR="${PROJECT_DIR}"
MANIFEST="${SCRIPT_DIR}/manifest_benchmark.tsv"
ENV_DIR="/data/stat-cadd/univ5464/HiQBind/envs/charges_env"
CHUNK_SIZE=25

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

python "${SCRIPT_DIR}/compute_benchmark_charges.py" batch \
    "${MANIFEST}" \
    --chunk-id "${SLURM_ARRAY_TASK_ID}" \
    --chunk-size "${CHUNK_SIZE}"

echo "End time: $(date)"
echo "Exit code: $?"
