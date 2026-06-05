# qAEV-PLIG-1

This file was written using AI. 

Extends AEV-PLIG by incorporating electrostatic information using a **fixed dielectric Coulomb model**.

---

## Modifications vs Baseline AEV-PLIG

### 1. New node features (×2 per ligand heavy atom)

The baseline node feature vector (one-hot element encoding + chemical features + AEV, 367 dimensions) is extended by two new features, giving **369 dimensions** total:

| Feature | Description |
|---------|-------------|
| `charge_i` | AM1-BCC partial charge of ligand atom *i* (elementary charge units) |
| `coulomb_potential_i` | Electrostatic potential at atom *i* due to all protein atoms within the cutoff: `V_i = (K / ε_r) · Σ_j (q_j / r_ij)` |

The Coulomb potential uses a **constant relative permittivity** ε_r = 4.0 (protein interior default) and a distance cutoff of 5.1 Å (matching the AEV radial cutoff).

### 2. New graph-level feature (×1 per complex)

The **total protein–ligand Coulombic interaction energy** is computed analytically:

```
E_coulomb = (K / ε_r) · Σ_i Σ_j (q_i · q_j / r_ij)
```

where the sum runs over all ligand heavy atoms *i* and all protein atoms *j* within the cutoff. This scalar is stored in the PyTorch Geometric `Data` object as `data.coulomb_energy`.

### 3. Architecture changes

The pooling step in `GATv2Net.forward()` changes from:

```python
# Baseline
x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)          # [B, 2·D]
self.fc1 = nn.Linear(final_dim * 2, 1024)
```

to:

```python
# qAEV-PLIG-1
x = torch.cat([gmp(x, batch), gap(x, batch),
               coulomb_energy.view(-1, 1)], dim=1)             # [B, 2·D + 1]
self.fc1 = nn.Linear(final_dim * 2 + 1, 1024)
```

All other architecture details (5 GATv2 layers, 3-layer MLP head, max+mean pooling) are unchanged from the baseline.

### 4. Charge sources

| Structure part | Method | Tool |
|----------------|--------|------|
| Ligand | AM1-BCC partial charges | antechamber / sqm |
| Protein | Amber ff14SB charges | OpenMM |

Charges are pre-computed per complex (see `data_preparation/charges/`) and stored in `charges.json` files. The protein Coulomb calculation uses **all protein atoms including hydrogens**, consistent with the ff14SB charge assignment.

---

## Physics parameters (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `COULOMB_K` | 332.0637 | e²/Å → kcal/mol conversion |
| `EPSILON_R` | 4.0 | Relative permittivity |
| `COULOMB_CUTOFF` | 5.1 Å | Interaction cutoff |

---

## Training data

HiQBind — energy-minimised structures derived from PDBBind 2020 (small molecules and polymeric ligands). See `data_preparation/hiqbind/` for the curation pipeline.
