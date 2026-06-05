# Prediction Run Log — 2026-04-30

## Model

`20260315-114050_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark`

- Scaler: `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/output/trained_models/20260315-114050_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark.pickle`
- Ensemble members (×10): `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/output/trained_models/20260315-114050_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_{0..9}.model`

## Script

`/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/process_and_predict.py`

Run with `--skip_validation` for all three datasets.

## Global features source

`/Users/ivanshatrov/Desktop/metamodel_data.csv`

Joined to input CSVs on `unique_id` before running predictions.

---

## Dataset 1 — CASF-2016

| Role | Path |
|---|---|
| Input CSV | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/casf_2016_aev_all.csv` |
| Processed CSV | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/casf_2016_aev_all_processed.csv` |
| Graphs pickle | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/casf_2016_aev_all_graphs.pickle` |
| PyG dataset | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/processed/casf_2016_aev_all.pt` |
| **Predictions** | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/output/predictions/casf_2016_aev_all_predictions.csv` |

- Input: 285 complexes → 241 predicted (44 filtered: rare elements or RDKit-unreadable)
- 8 complexes lacked global features in metamodel_data.csv; 5 survived filtering and were imputed with column medians

## Dataset 2 — Zero-ligand bias

| Role | Path |
|---|---|
| Input CSV | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/0_ligand_bias_aev_all.csv` |
| Processed CSV | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/0_ligand_bias_aev_all_processed.csv` |
| Graphs pickle | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/0_ligand_bias_aev_all_graphs.pickle` |
| PyG dataset | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/processed/0_ligand_bias_aev_all.pt` |
| **Predictions** | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/output/predictions/0_ligand_bias_aev_all_predictions.csv` |

- Input: 365 complexes → 329 predicted (36 filtered: rare elements or RDKit-unreadable)
- 5 complexes lacked global features; all 5 were among those filtered out

## Dataset 3 — FEP benchmark (ross evaluation)

| Role | Path |
|---|---|
| Input CSV | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/ross_evaluation.csv` |
| Structure files | `/Users/ivanshatrov/Documents/fep_benchmark_processed/` |
| Processed CSV | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/ross_evaluation_processed.csv` |
| Graphs pickle | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/ross_evaluation_graphs.pickle` |
| PyG dataset | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/data/processed/ross_evaluation.pt` |
| **Predictions** | `/Users/ivanshatrov/AEV-PLIG-modifications/AEV-PLIG/output/predictions/ross_evaluation_predictions.csv` |

- Input: 1184 complexes → 1184 predicted (0 filtered)
- All global features present; no imputation needed
