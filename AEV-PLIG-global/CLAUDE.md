# AEV-PLIG — Claude Code Context

## Project Overview
AEV-PLIG is a GNN-based binding affinity scoring function for protein-ligand complexes.
It uses GATv2 message-passing layers with AEV (atomic environment vector) node features,
followed by global mean pooling and an MLP to predict pKd/pKi.

## Tech Stack
- Python 3.8, PyTorch, PyTorch Geometric
- torchani (modified, in torchani_mod/)
- RDKit, biopandas, pandas, scikit-learn

## Repo Structure
- model_defs.py       — GATv2Net model class (forward pass, layers, MLP head)
- utils.py            — GraphDataset class, data loading helpers
- create_pytorch_data.py — converts pickle graphs → .pt files for training
- training.py         — training loop, CLI entry point
- process_and_predict.py — end-to-end prediction pipeline
- helpers.py          — metrics, model registry, collate_fn
- generate_*_graphs.py — graph construction from raw PDB/SDF files → pickle
- torchani_mod/       — modified TorchANI for AEV computation
- data/               — raw data, processed .pt files, pickles
- output/             — trained models, predictions

## How to Run
- Train: `python training.py --model=GATv2Net [args]`
- Predict: `python process_and_predict.py --dataset_csv=... --data_name=... --trained_model_name=...`
- Processed data lives in: `data/processed/*.pt`
- Trained models saved to: `output/trained_models/`

## Key Conventions
- PyTorch Geometric Data objects are built in create_pytorch_data.py and stored as .pt files
- Graphs are keyed by `unique_id` throughout the pipeline
- The GraphDataset class (utils.py) loads .pt files and serves Data objects to the DataLoader
- Do NOT modify files in data/pdbbind, data/bindingnet, data/bindingdb (raw source data)

## Behaviour Rules
- Ask for confirmation before making any edit to a file
- Show a clear plan (which files will change, what will change) before executing anything
- Make one logical change at a time, then pause for review
- Do not refactor anything outside the scope of the requested change
- After each file edit, summarise what changed and why