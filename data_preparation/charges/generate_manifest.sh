#!/bin/bash
# generate_manifest.sh
# ====================
# Walk the HiQBind dataset and produce a manifest TSV listing every
# (protein_pdb, ligand_sdf, output_json) triple — one line per complex.
#
# Uses a single `find` pass, much faster than Python glob on shared filesystems.
#
# Usage:
#     bash generate_manifest.sh /data/stat-cadd/univ5464/HiQBind/dataset manifest.tsv

set -euo pipefail

DATASET_ROOT="${1:?Usage: bash generate_manifest.sh <dataset_root> <output_manifest>}"
MANIFEST="${2:?Usage: bash generate_manifest.sh <dataset_root> <output_manifest>}"

echo "Scanning ${DATASET_ROOT} ..."

find "${DATASET_ROOT}" -name "*_ligand_refined.sdf" -type f \
  | awk -F/ '{
      dir=""; for(i=2;i<NF;i++) dir=dir"/"$i;
      lig=$NF; sub(/_ligand_refined\.sdf$/, "", lig);
      printf "%s/%s_protein_refined.pdb\t%s\t%s/charges.json\n", dir, lig, $0, dir
    }' > "${MANIFEST}"

count=$(wc -l < "${MANIFEST}")
echo "Manifest written: ${MANIFEST}"
echo "Total complexes:  ${count}"
