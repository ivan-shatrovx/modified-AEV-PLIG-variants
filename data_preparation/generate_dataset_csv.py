"""
generate_dataset_csv.py — Scan HiQBind directory and produce data/dataset.csv.

Walks pdbbind_opt_sm/ and pdbbind_opt_poly/ under the HiQBind dataset root,
finds every ligand subfolder that contains charges.json, and matches the
PDB code to data/pdbbind_processed.csv for pK labels and train/valid/test splits.

Usage:
    python generate_dataset_csv.py [--dataset_root /path/to/HiQBind/dataset]
                                   [--pdbbind_csv data/pdbbind_processed.csv]
                                   [--output data/dataset.csv]
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm


def find_complexes(dataset_root: str) -> list:
    """Walk the HiQBind dataset and find all ligand subfolders with charges.json.

    Returns list of dicts with keys: pdb_code, unique_id, pdb_file, sdf_file, charges_json.
    """
    complexes = []
    subsets = ["pdbbind_opt_sm", "pdbbind_opt_poly"]

    for subset in subsets:
        subset_dir = os.path.join(dataset_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"Warning: subset directory not found: {subset_dir}")
            continue

        pdb_dirs = sorted(os.listdir(subset_dir))
        for pdb_code in tqdm(pdb_dirs, desc=f"Scanning {subset}"):
            pdb_dir = os.path.join(subset_dir, pdb_code)
            if not os.path.isdir(pdb_dir):
                continue

            # Each subdirectory matching {pdbcode}_{ligname}_{chain}_{resnum} is a ligand
            for entry in sorted(os.listdir(pdb_dir)):
                ligand_dir = os.path.join(pdb_dir, entry)
                if not os.path.isdir(ligand_dir):
                    continue

                charges_path = os.path.join(ligand_dir, "charges.json")
                if not os.path.isfile(charges_path):
                    continue

                # Find the refined files
                protein_file = os.path.join(ligand_dir, f"{entry}_protein_refined.pdb")
                sdf_file = os.path.join(ligand_dir, f"{entry}_ligand_refined.sdf")

                if not os.path.isfile(protein_file):
                    print(f"  Warning: missing protein: {protein_file}")
                    continue
                if not os.path.isfile(sdf_file):
                    print(f"  Warning: missing ligand SDF: {sdf_file}")
                    continue

                complexes.append({
                    "pdb_code": pdb_code.lower(),
                    "unique_id": entry,
                    "pdb_file": protein_file,
                    "sdf_file": sdf_file,
                    "charges_json": charges_path,
                })

    return complexes


def main():
    parser = argparse.ArgumentParser(description="Generate dataset.csv from HiQBind directory")
    parser.add_argument("--dataset_root", type=str,
                        default="/data/stat-cadd/univ5464/HiQBind/dataset",
                        help="Root of HiQBind dataset directory")
    parser.add_argument("--pdbbind_csv", type=str,
                        default="data/pdbbind_processed.csv",
                        help="PDBBind processed CSV with pK values and splits")
    parser.add_argument("--output", type=str,
                        default="data/dataset.csv",
                        help="Output CSV path")
    parser.add_argument("--max_tanimoto", type=float, default=0.9,
                        help="Exclude complexes with max Tanimoto to FEP benchmark >= this value (default: 0.9)")
    args = parser.parse_args()

    # Scan for complexes with charges.json
    print(f"Scanning {args.dataset_root} for complexes with charges.json...")
    complexes = find_complexes(args.dataset_root)
    print(f"Found {len(complexes)} complexes with charges.json")

    if len(complexes) == 0:
        print("No complexes found. Check the dataset root path.")
        return

    df = pd.DataFrame(complexes)

    # Load PDBBind labels
    pdbbind = pd.read_csv(args.pdbbind_csv, index_col=0)
    pdbbind = pdbbind[["PDB_code", "-logKd/Ki", "split_core", "max_tanimoto_fep_benchmark"]].rename(
        columns={"PDB_code": "pdb_code", "-logKd/Ki": "pK", "split_core": "split"}
    )
    print(f"PDBBind CSV has {len(pdbbind)} entries")

    # Filter out complexes with high ligand similarity to the Ross benchmark (prevent data leakage)
    before_filter = len(pdbbind)
    pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < args.max_tanimoto]
    print(f"Tanimoto filter (<{args.max_tanimoto}): kept {len(pdbbind)}/{before_filter} PDB entries")
    pdbbind = pdbbind.drop(columns=["max_tanimoto_fep_benchmark"])

    # Merge — only keep complexes that have pK labels
    before = len(df)
    df = df.merge(pdbbind, on="pdb_code", how="inner")
    print(f"Matched {len(df)} complexes (dropped {before - len(df)} without pK labels)")

    # Select and order columns for output
    df = df[["unique_id", "pdb_file", "sdf_file", "charges_json", "pK", "split"]]
    df = df.sort_values("unique_id").reset_index(drop=True)

    # Summary
    print(f"\nDataset summary:")
    print(f"  Total complexes: {len(df)}")
    print(f"  Unique PDB codes: {df['unique_id'].str[:4].str.lower().nunique()}")
    print(f"  Split distribution:")
    for split, count in df["split"].value_counts().items():
        print(f"    {split}: {count}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
