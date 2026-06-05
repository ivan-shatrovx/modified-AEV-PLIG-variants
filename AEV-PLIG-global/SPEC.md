# SPEC: Add Global Features to AEV-PLIG

## Objective
Extend AEV-PLIG to incorporate system-level ('global') features — scalar descriptors
of the whole protein-ligand complex — alongside the existing per-node GATv2 features.

## Global Features
The 10 global features are loaded from a CSV file (one row per complex, keyed on `unique_id`):
- arp_proximal
- rdk_MolLogP
- arp_hbond
- arp_hydrophobic
- rdk_fr_halogen
- rdk_RingCount
- dpk_volume_score
- rdk_TPSA
- dpk_prop_polar_atm
- pdk_mean_loc_hyd_dens

## Architecture Change
The modified forward pass should work as follows:

1. Node features pass through the 5 existing GATv2Conv layers (unchanged)
2. Global mean pooling is applied to produce a graph-level vector (unchanged)
3. The 10 global features are passed through a small normalisation MLP
   (`global_mlp`: Linear → BatchNorm → LeakyReLU) to produce a projected vector
4. The pooled graph vector and the projected global vector are concatenated
5. The concatenated vector is passed to the existing final MLP head

## Files to Change

### 1. create_pytorch_data.py
- Load the global features CSV into a dictionary: `{unique_id: np.array of 10 floats}`
- When constructing each PyG Data object, add a new attribute:
  `data.global_features = torch.tensor(global_dict[unique_id], dtype=torch.float)`
- If a `unique_id` is missing from the CSV, raise a clear error identifying which ID

### 2. utils.py — GraphDataset
- The Data objects already contain `global_features` (added in step 1 above)
- No structural change needed to GraphDataset itself, but verify that `global_features`
  is preserved correctly when .pt files are loaded and served by the DataLoader
- If any batching/collation issues arise with the new field, fix them here

### 3. model_defs.py — GATv2Net
- Add a constructor argument: `global_feat_dim=10`
- Add a `global_mlp` module in `__init__`:
    nn.Sequential(
        nn.Linear(global_feat_dim, global_feat_dim),
        nn.BatchNorm1d(global_feat_dim),
        nn.LeakyReLU()
    )
- In `forward(self, data)`:
    - Extract `global_vec = data.global_features`  (shape: [batch_size, 10])
    - After pooling, compute `global_proj = self.global_mlp(global_vec)`
    - Concatenate: `combined = torch.cat([pooled, global_proj], dim=-1)`
    - Pass `combined` into the existing final MLP
- Update the input dimension of the first layer of the final MLP to account for the
  extra 10 dimensions from global_proj

## What NOT to Change
- The GATv2Conv layer definitions and their hyperparameters
- The training loop (training.py)
- The graph generation scripts (generate_*_graphs.py)
- The pickle file format

## Verification Steps (run after each file change)
1. After create_pytorch_data.py: re-run it on a small test split, print one Data object
   and confirm `data.global_features` has shape [10] and correct values
2. After model_defs.py: instantiate GATv2Net and run a single forward pass with a 
   dummy batch, confirm no shape errors and output is a scalar per graph
3. After all changes: run a short training loop (1-2 epochs, small batch) and confirm
   loss decreases without errors
```

---

## @-mention Cheat Sheet
Use these **inside Claude Code** during the session to give precise context:
```
# When starting the session — give Claude the full spec:
@SPEC.md implement the global features spec. Start by reading all affected files 
before proposing any changes. Show me your plan first.

# When working on the data object:
@create_pytorch_data.py add global_features to the Data object as described in @SPEC.md

# When working on the model:
@model_defs.py add global_mlp and update the forward pass as described in @SPEC.md. 
Do not touch the GATv2Conv layer definitions.

# When working on the dataset:
@utils.py verify that global_features is handled correctly in GraphDataset 
when loading .pt files and batching.

# If something breaks:
@model_defs.py @utils.py the forward pass is throwing a shape error: [paste error here]. 
Read both files and diagnose before making any change.