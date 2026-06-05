# Modified AEV-PLIG Variants

This repository contains four modified variants of [AEV-PLIG](https://github.com/oxpig/AEV-PLIG), a GATv2-based graph neural network for protein–ligand binding affinity prediction. Each variant extends the baseline model in a different direction. The baseline model and its original training code are not included here.

---

## Repository Structure

```
├── qAEV-PLIG/
│   ├── qAEV-PLIG-1/        Electrostatic model — fixed dielectric
│   └── qAEV-PLIG-2/        Electrostatic model — distance-dependent dielectric
├── AEV-PLIG-global/        Global pocket/ligand feature models (Global-10 and Global-6)
└── data_preparation/
    ├── generate_dataset_csv.py     Walk HiQBind dataset → dataset.csv
    ├── hiqbind/                    PDBBind structure curation pipeline (HiQBind)
    ├── charges/                    AM1-BCC + ff14SB charge computation
    └── global_features/            Pocket and molecular descriptor pipelines
```

---

## Model Variants

| Model | Directory | Description | Documentation |
|-------|-----------|-------------|---------------|
| qAEV-PLIG-1 | `qAEV-PLIG/qAEV-PLIG-1/` | Electrostatics via fixed dielectric Coulomb potential | [MODEL.md](qAEV-PLIG/qAEV-PLIG-1/MODEL.md) |
| qAEV-PLIG-2 | `qAEV-PLIG/qAEV-PLIG-2/` | Electrostatics via distance-dependent dielectric + multi-task training | [MODEL.md](qAEV-PLIG/qAEV-PLIG-2/MODEL.md) |
| Global-10 | `AEV-PLIG-global/` | 10 global pocket/ligand features injected into MLP | [MODEL_GLOBAL10.md](AEV-PLIG-global/MODEL_GLOBAL10.md) |
| Global-6 | `AEV-PLIG-global/` | 6 fpocket pocket descriptors injected into MLP | [MODEL_GLOBAL6.md](AEV-PLIG-global/MODEL_GLOBAL6.md) |

---

## Data Preparation

Full instructions for reproducing the training data are in [`data_preparation/`](data_preparation/).

### HiQBind dataset (qAEV-PLIG-1, qAEV-PLIG-2)

The electrostatic models are trained on HiQBind — a curated, energy-minimised version of PDBBind 2020. The curation pipeline lives in [`data_preparation/hiqbind/`](data_preparation/hiqbind/):

1. `pre_process/create_pdbbind_input.py` — filter PDBBind entries and produce input CSVs
2. `split_csv.py` — chunk the input for SLURM parallelism
3. `run_pdbbind_opt.sh` — orchestrate the SLURM job arrays
4. `submit_hiqbind.sh` — SLURM worker (runs `hiqbind/process.py` per chunk)

After curation, `generate_dataset_csv.py` walks the HiQBind output directory and produces the `dataset.csv` manifest used for training.

### Partial charges (qAEV-PLIG-1, qAEV-PLIG-2)

AM1-BCC ligand charges (via antechamber/sqm) and ff14SB protein charges (via OpenMM) are pre-computed and stored per complex. Scripts in [`data_preparation/charges/`](data_preparation/charges/):

- `generate_manifest.sh` — list all HiQBind complexes
- `compute_charges.py` — compute charges for one complex
- `submit_charges.sh` — SLURM array job to run at scale
- `compute_benchmark_charges.py` / `submit_benchmark_charges.sh` — same for benchmark sets

### Global features (Global-10, Global-6)

Features are computed from PDBBind + BindingNet + BindingDB structures. Scripts in [`data_preparation/global_features/`](data_preparation/global_features/):

- `run_dpocket_pipeline.py` — fpocket pocket descriptors (`dpk_*`)
- `run_arpeggio_pipeline.py` + `arpeggio_json_to_csv.py` — Arpeggio interaction contacts (`arp_*`)
- `run_rdkit_pipeline.py` — RDKit molecular descriptors (`rdk_*`)

---

## Training

Each model directory contains the full training pipeline. The general workflow is:

```
generate_split_csvs.py    →  generate_graphs.py    →  create_pytorch_data.py    →  training.py
(define train/test split)    (build graph pickle)     (build PyTorch .pt files)    (train model)
```

See the individual MODEL.md files for model-specific instructions and differences from the baseline.

This file was written using Claude Sonnet 4.6. 
