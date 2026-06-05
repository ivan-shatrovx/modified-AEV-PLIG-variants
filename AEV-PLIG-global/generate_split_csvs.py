"""
generate_split_csvs.py — Generate train/valid/test split CSVs for global feature models.

Reads data/pdbbind_processed.csv and produces three new split files:
  - data/pdbbind_casf2016.csv     (current split unchanged, no Tanimoto filter)
  - data/pdbbind_ood.csv          (test = OOD test complexes)
  - data/pdbbind_ligand_bias.csv  (test = zero-ligand-bias complexes)

The FEP benchmark split uses pdbbind_processed.csv directly (Tanimoto filter is
applied inside create_pytorch_data.py when --filter_fep is passed).

Matching logic: PDB_code column (lowercase) vs benchmark key/PDB_code columns.
Previous test rows that fall outside the new test set are moved to train.
"""

import pandas as pd

PDBBIND_CSV     = "data/pdbbind_processed.csv"
OOD_CSV         = "../benchmarks/index_oodtest.csv"
LIGAND_BIAS_CSV = "../benchmarks/zero_ligand_bias_test.csv"


def make_split(df: pd.DataFrame, benchmark_pdb_codes: set, name: str) -> pd.DataFrame:
    out = df.copy()
    pdb_lower = out["PDB_code"].str.lower()

    in_benchmark = pdb_lower.isin(benchmark_pdb_codes)

    # rows currently in test but NOT in the new benchmark → train
    was_test = out["split_core"] == "test"
    out.loc[was_test & ~in_benchmark, "split_core"] = "train"

    # rows in benchmark → test (regardless of previous split)
    out.loc[in_benchmark, "split_core"] = "test"

    counts = out["split_core"].value_counts().to_dict()
    print(f"{name}: train={counts.get('train',0)}  valid={counts.get('valid',0)}  test={counts.get('test',0)}")
    print(f"  {in_benchmark.sum()} rows matched to benchmark ({len(benchmark_pdb_codes)} unique benchmark codes)")
    return out


def main():
    df = pd.read_csv(PDBBIND_CSV, index_col=0)
    print(f"Loaded pdbbind_processed.csv: {len(df)} rows")
    orig = df["split_core"].value_counts().to_dict()
    print(f"Original: train={orig.get('train',0)}  valid={orig.get('valid',0)}  test={orig.get('test',0)}\n")

    # CASF-2016: current split unchanged, just write a clean copy
    df_casf = df.copy()
    counts = df_casf["split_core"].value_counts().to_dict()
    print(f"pdbbind_casf2016 (unchanged): train={counts.get('train',0)}  valid={counts.get('valid',0)}  test={counts.get('test',0)}")
    df_casf.to_csv("data/pdbbind_casf2016.csv")
    print("  → data/pdbbind_casf2016.csv\n")

    # OOD split
    ood_df = pd.read_csv(OOD_CSV)
    ood_test_codes = set(ood_df.loc[ood_df["split"] == "test", "PDB_code"].str.lower())
    print(f"OOD benchmark test codes: {len(ood_test_codes)}")
    df_ood = make_split(df, ood_test_codes, "pdbbind_ood")
    df_ood.to_csv("data/pdbbind_ood.csv")
    print("  → data/pdbbind_ood.csv\n")

    # Ligand bias split
    lb_df = pd.read_csv(LIGAND_BIAS_CSV)
    lb_codes = set(lb_df["key"].str.lower())
    print(f"Ligand-bias benchmark codes: {len(lb_codes)}")
    df_lb = make_split(df, lb_codes, "pdbbind_ligand_bias")
    df_lb.to_csv("data/pdbbind_ligand_bias.csv")
    print("  → data/pdbbind_ligand_bias.csv")


if __name__ == "__main__":
    main()
