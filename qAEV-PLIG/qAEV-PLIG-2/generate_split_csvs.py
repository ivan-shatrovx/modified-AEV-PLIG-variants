"""
generate_split_csvs.py — Generate train/valid/test split CSVs for charged models.

Reads data/dataset.csv and produces two new split files:
  - data/dataset_ood.csv        (test = OOD test complexes)
  - data/dataset_ligand_bias.csv (test = zero-ligand-bias complexes)

Matching logic: base PDB code = unique_id[:4].lower()
Previous test rows that fall outside the new test set are moved to train.
"""

import pandas as pd

DATASET_CSV      = "data/dataset.csv"
OOD_CSV          = "../benchmarks/index_oodtest.csv"
LIGAND_BIAS_CSV  = "../benchmarks/zero_ligand_bias_test.csv"


def make_split(df: pd.DataFrame, benchmark_pdb_codes: set, name: str) -> pd.DataFrame:
    out = df.copy()
    base_pdb = out["unique_id"].str[:4].str.lower()

    in_benchmark = base_pdb.isin(benchmark_pdb_codes)

    # rows currently in test but NOT in the new benchmark → train
    was_test = out["split"] == "test"
    out.loc[was_test & ~in_benchmark, "split"] = "train"

    # rows in benchmark → test (regardless of previous split)
    out.loc[in_benchmark, "split"] = "test"

    counts = out["split"].value_counts().to_dict()
    print(f"{name}: train={counts.get('train',0)}  valid={counts.get('valid',0)}  test={counts.get('test',0)}")
    n_benchmark_in_dataset = in_benchmark.sum()
    print(f"  {n_benchmark_in_dataset} rows matched to benchmark ({len(benchmark_pdb_codes)} unique benchmark codes)")
    return out


def main():
    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded dataset.csv: {len(df)} rows")
    orig = df["split"].value_counts().to_dict()
    print(f"Original: train={orig.get('train',0)}  valid={orig.get('valid',0)}  test={orig.get('test',0)}\n")

    # OOD split
    ood_df = pd.read_csv(OOD_CSV)
    ood_test_codes = set(ood_df.loc[ood_df["split"] == "test", "PDB_code"].str.lower())
    print(f"OOD benchmark test codes: {len(ood_test_codes)}")
    df_ood = make_split(df, ood_test_codes, "dataset_ood")
    df_ood.to_csv("data/dataset_ood.csv", index=False)
    print("  → data/dataset_ood.csv\n")

    # Ligand bias split
    lb_df = pd.read_csv(LIGAND_BIAS_CSV)
    lb_codes = set(lb_df["key"].str.lower())
    print(f"Ligand-bias benchmark codes: {len(lb_codes)}")
    df_lb = make_split(df, lb_codes, "dataset_ligand_bias")
    df_lb.to_csv("data/dataset_ligand_bias.csv", index=False)
    print("  → data/dataset_ligand_bias.csv")


if __name__ == "__main__":
    main()
