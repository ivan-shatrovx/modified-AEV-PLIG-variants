#!/bin/bash
set -e

WORKDIR=/data/stat-cadd/univ5464/HiQBind
CONDA_ENV=/data/stat-cadd/univ5464/conda_envs/hiqbind
N_SM_CHUNKS=50    # ~335 PDBIDs per chunk → ~26h each in serial
N_POLY_CHUNKS=10  # ~271 PDBIDs per chunk → ~22h each in serial

echo "=== PDBBind-opt Workflow ==="
echo "Working dir: $WORKDIR"

# Load environment for splitting
module load Anaconda3
source activate $CONDA_ENV
export PATH=$CONDA_ENV/bin:$PATH

cd $WORKDIR

# ── Split CSVs ──
echo ""
echo "Splitting input CSVs..."
if [ -f "pre_process/PDBBind_sm.csv" ]; then
    python split_csv.py pre_process/PDBBind_sm.csv $N_SM_CHUNKS chunks/
else
    echo "ERROR: PDBBind_sm.csv not found. Run Step 4 first."
    exit 1
fi

if [ -f "pre_process/PDBBind_poly.csv" ]; then
    python split_csv.py pre_process/PDBBind_poly.csv $N_POLY_CHUNKS chunks/
else
    echo "ERROR: PDBBind_poly.csv not found. Run Step 4 first."
    exit 1
fi

# Count actual chunks created
ACTUAL_SM=$(ls chunks/PDBBind_sm_chunk_*.csv 2>/dev/null | wc -l)
ACTUAL_POLY=$(ls chunks/PDBBind_poly_chunk_*.csv 2>/dev/null | wc -l)
echo "Created $ACTUAL_SM sm chunks, $ACTUAL_POLY poly chunks"

# Create output directories
mkdir -p dataset/pdbbind_opt_sm dataset/pdbbind_opt_poly

# ── Submit SLURM job arrays ──
echo ""
echo "Submitting jobs..."

if [ $ACTUAL_SM -gt 0 ]; then
    SM_JOB=$(sbatch --array=1-$ACTUAL_SM \
                    --export=LIGAND_TYPE=sm \
                    submit_hiqbind.sh | awk '{print $4}')
    echo "Small molecule array: $SM_JOB (tasks 1-$ACTUAL_SM)"
fi

if [ $ACTUAL_POLY -gt 0 ]; then
    POLY_JOB=$(sbatch --array=1-$ACTUAL_POLY \
                      --export=LIGAND_TYPE=poly \
                      submit_hiqbind.sh | awk '{print $4}')
    echo "Polymer array: $POLY_JOB (tasks 1-$ACTUAL_POLY)"
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in: $WORKDIR/logs/"
echo "==========================="
