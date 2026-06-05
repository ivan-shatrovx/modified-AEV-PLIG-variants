#!/bin/bash
#SBATCH --job-name=hiqbind
#SBATCH --partition=medium
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/hiqbind_%A_%a.out
#SBATCH --error=logs/hiqbind_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=univ5464@ox.ac.uk

# ── Setup ────────────────────────────────────────
WORKDIR=/data/stat-cadd/univ5464/HiQBind
CONDA_ENV=/data/stat-cadd/univ5464/conda_envs/hiqbind

module purge
module load Anaconda3
source activate $CONDA_ENV
# CRITICAL: Force correct Python (Anaconda3 module puts system Python first)
export PATH=$CONDA_ENV/bin:$PATH

cd $WORKDIR/hiqbind

# ── Determine input from environment variables ───
# LIGAND_TYPE: "sm" or "poly"
# SLURM_ARRAY_TASK_ID: chunk number
CHUNK_ID=$SLURM_ARRAY_TASK_ID
INPUT_CSV=$WORKDIR/chunks/PDBBind_${LIGAND_TYPE}_chunk_${CHUNK_ID}.csv

if [ "$LIGAND_TYPE" == "poly" ]; then
    OUTPUT_DIR=$WORKDIR/dataset/pdbbind_opt_poly
    POLY_FLAG="--poly"
else
    OUTPUT_DIR=$WORKDIR/dataset/pdbbind_opt_sm
    POLY_FLAG=""
fi

mkdir -p $OUTPUT_DIR

echo "=== HiQBind Job ==="
echo "Job ID: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Type: $LIGAND_TYPE  Input: $INPUT_CSV"
echo "Output: $OUTPUT_DIR"
echo "Python: $(which python)"
echo "Start: $(date)"
echo "==================="

# ── Run workflow (serial mode — parallelism managed by SLURM) ──
python process.py -i $INPUT_CSV -d $OUTPUT_DIR $POLY_FLAG --serial

echo "End: $(date)"
