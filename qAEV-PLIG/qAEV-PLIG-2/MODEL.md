# qAEV-PLIG-2

Extends AEV-PLIG with electrostatics using a **distance-dependent dielectric (DDD)** model and **multi-task training** on both pK and Coulomb energy.

---

## Modifications vs Baseline AEV-PLIG

### 1. New node features (×2 per ligand heavy atom)

Identical to qAEV-PLIG-1: the node feature vector is extended from 367 to **369 dimensions** with:

| Feature | Description |
|---------|-------------|
| `charge_i` | AM1-BCC partial charge of ligand atom *i* |
| `coulomb_potential_i` | Electrostatic potential at atom *i* due to protein atoms within cutoff, using distance-dependent dielectric |

The key difference from qAEV-PLIG-1 is the dielectric function used to compute these potentials.

### 2. Distance-dependent dielectric (Mehler–Solmajer)

Instead of a constant permittivity, qAEV-PLIG-2 uses the empirical distance-dependent dielectric from Mehler & Solmajer (1991), as implemented in AutoDock4:

```
ε(r) = A + B / (1 + K · exp(−λ · B · r))
```

with parameters:

| Parameter | Value |
|-----------|-------|
| `EPSILON_0` | 78.4 (bulk water at 25°C) |
| `A` | −8.5525 |
| `B` | 86.9525 (= ε₀ − A) |
| `LAMBDA` | 0.003627 Å⁻¹ |
| `K` | 7.7839 |
| `COULOMB_CUTOFF` | 12.0 Å (extended from 5.1 Å in qAEV-PLIG-1) |

This models the gradual transition from a low dielectric near the protein surface to bulk-water screening at longer distances.

### 3. Architecture — dual output heads

qAEV-PLIG-2 uses a **multi-task** architecture (`GATv2Net_DDD`) with two independent MLP branches sharing a common GNN backbone and pooling layer:

```
GNN backbone (5 × GATv2)
        ↓
  max-pool + mean-pool        [B, 2·D]
    ↙              ↘
pK head          Coulomb energy head
(3-layer MLP)    (1-layer MLP)
    ↓                  ↓
  pK pred          E_coulomb pred
```

The **Coulomb energy is not injected** as an extra input to the MLP (unlike qAEV-PLIG-1). The model is instead trained to predict Coulomb energy as a second task, forcing the GNN to encode electrostatic structure in the learned representations.

```python
# forward() returns a tuple
pk, ec = model(data)    # shapes [B, 1], [B, 1]
```

### 4. Multi-task training loss

```
L = w_pk · MSE(pk_pred, pk_true) + w_ec · MSE(ec_pred, ec_true_normalised)
```

Default weights: `w_pk = 1.0`, `w_ec = 1.0` (configurable in `config_qaev_plig_2.py`). The Coulomb energy targets are normalised to zero mean and unit variance before training; normalisation statistics are saved to `processed_ddd/coulomb_norm_stats.json`.

### 5. Dropout

The pK MLP head includes dropout (p = 0.2) after the first batch-norm layer, which is absent in the baseline and qAEV-PLIG-1.

---

## Key files

| File | Role |
|------|------|
| `config_qaev_plig_2.py` | DDD physics parameters and loss weights |
| `helpers_qaev_plig_2.py` | `epsilon_ddd()`, `compute_coulomb_potential_ddd()`, `compute_coulomb_energy_ddd()` |
| `model_defs.py` | `GATv2Net_DDD` with dual heads |
| `training.py` | Multi-task training loop |

`config.py` and `helpers.py` are shared with qAEV-PLIG-1 and provide the base AEV parameters and charge-loading utilities.

---

## Training data

Same as qAEV-PLIG-1: HiQBind (energy-minimised PDBBind 2020). Graphs are stored in a separate pickle (`graphs_ddd.pickle`) and PyTorch files in `data/processed_ddd/` to avoid collision with qAEV-PLIG-1 artefacts.

Claude Sonnet 4.6 was used to help create and format this file. 
