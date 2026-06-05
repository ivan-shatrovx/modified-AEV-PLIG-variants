# Global-10

Extends AEV-PLIG by injecting **10 global pocket and ligand features** into the MLP head alongside the pooled GNN representation.

---

## Modifications vs AEV-PLIG

### 1. Global feature vector (10 dimensions per complex)

Ten scalar descriptors are computed externally (prior to training) and stored in `metamodel_data.csv`. They are loaded per complex at dataset creation time and stored as `data.global_features` in each PyTorch Geometric `Data` object.

| Feature | Source | Description |
|---------|--------|-------------|
| `arp_proximal` | Arpeggio | Count of proximal protein–ligand contacts |
| `arp_hbond` | Arpeggio | Count of hydrogen bonds |
| `arp_hydrophobic` | Arpeggio | Count of hydrophobic contacts |
| `rdk_MolLogP` | RDKit | Wildman–Crippen LogP |
| `rdk_fr_halogen` | RDKit | Number of halogen functional groups |
| `rdk_RingCount` | RDKit | Number of rings |
| `rdk_TPSA` | RDKit | Topological polar surface area |
| `dpk_volume_score` | fpocket | Pocket volume score |
| `dpk_prop_polar_atm` | fpocket | Proportion of polar pocket atoms |
| `dpk_mean_loc_hyd_dens` | fpocket | Mean local hydrophobic density |

Feature selection was performed using SHAP analysis on a metamodel trained on AEV-PLIG predictions and these descriptors.

### 2. Global feature MLP sub-network

Before concatenation, the 10-dimensional feature vector is passed through a small learned projection:

```python
self.global_mlp = nn.Sequential(
    nn.Linear(10, 10),
    nn.BatchNorm1d(10),
    nn.LeakyReLU()
)
```

### 3. Architecture changes

The pooling step changes from:

```python
# AEV-PLIG
x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)          # [B, 2·D]
self.fc1 = nn.Linear(final_dim * 2, 1024)
```

to:

```python
# Global-10
global_proj = self.global_mlp(data.global_features.view(-1, 10))
x = torch.cat([gmp(x, batch), gap(x, batch), global_proj], dim=1)   # [B, 2·D + 10]
self.fc1 = nn.Linear(final_dim * 2 + 10, 1024)
```

All other architecture details are unchanged.

### 4. Node features

Node features are **unchanged** from the baseline AEV-PLIG (367 dimensions). No per-atom features are added.

---

## Training data

PDBBind 2020 (refined set + general set). Graphs are generated using the original AEV-PLIG mol2/PDB pipeline. Global features are computed from the same structures using the scripts in `data_preparation/global_features/`.

This file was written using Claude Sonnet 4.6.
