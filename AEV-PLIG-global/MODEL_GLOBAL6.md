# DPK-6

A variant of AEV-PLIG that uses **6 Dpocket-derived pocket descriptors** as global features. 

---

## Modifications vs AEV-PLIG

### 1. Global feature vector (6 dimensions per complex)

All 6 features come from fpocket/dpocket and describe the binding pocket geometry and chemistry. No protein–ligand interaction counts (Arpeggio) or ligand molecular descriptors (RDKit) are used.

| Feature | Source | Description |
|---------|--------|-------------|
| `dpk_volume_score` | fpocket | Pocket volume score |
| `dpk_prop_polar_atm` | fpocket | Proportion of polar pocket atoms |
| `dpk_mean_loc_hyd_dens` | fpocket | Mean local hydrophobicity density |
| `dpk_charge_score` | fpocket | Pocket charge score |
| `dpk_flex` | fpocket | Pocket flexibility score |
| `dpk_aromatic` | fpocket | Sum of aromatic residue counts (PHE + HIS + TYR + TRP) |

`dpk_aromatic` is a derived feature: it is computed at dataset creation time as the sum of the four individual aromatic residue pocket counts from fpocket output.

### 2. Architecture changes

Identical to Global-10 but with `global_feat_dim = 6`:

```python
self.global_mlp = nn.Sequential(
    nn.Linear(6, 6),
    nn.BatchNorm1d(6),
    nn.LeakyReLU()
)
self.fc1 = nn.Linear(final_dim * 2 + 6, 1024)
```

### 3. Node features

Unchanged from the baseline AEV-PLIG (367 dimensions).

---

## Relationship to Global-10

Global-6 shares all graph generation and utility code with Global-10. The only model-specific files are:

| File | Role |
|------|------|
| `model_defs_dpk6.py` | `GATv2Net` with `global_feat_dim=6` |
| `create_pytorch_data_dpk6.py` | Reads only the 6 dpk columns from `metamodel_data.csv` |
| `training_dpk6.py` | Training loop for Global-6 |
| `process_and_predict_dpk6.py` | Inference |

---

## Training data

Same as Global-10: PDBBind 2020. Pocket features are computed using `data_preparation/global_features/run_dpocket_pipeline.py`.

This file was written using Claude Sonnet 4.6.
