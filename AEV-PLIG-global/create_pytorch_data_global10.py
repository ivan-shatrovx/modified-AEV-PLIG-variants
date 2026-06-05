import argparse
import numpy as np
import pandas as pd
import pickle
from utils import GraphDataset

GLOBAL_FEATURE_COLS = [
    'arp_proximal',
    'rdk_MolLogP',
    'arp_hbond',
    'arp_hydrophobic',
    'rdk_fr_halogen',
    'rdk_RingCount',
    'dpk_volume_score',
    'rdk_TPSA',
    'dpk_prop_polar_atm',
    'dpk_mean_loc_hyd_dens',
]

parser = argparse.ArgumentParser(description="Create PyTorch Geometric datasets (Global-10 model)")
parser.add_argument('--name',               type=str, required=True,
                    help='Output dataset name (used as filename prefix for .pt files)')
parser.add_argument('--pdbbind_split_csv',  type=str, default='data/pdbbind_processed.csv',
                    help='Path to pdbbind split CSV (must have PDB_code, -logKd/Ki, split_core columns)')
parser.add_argument('--global_features_csv', type=str, default='data/metamodel_data.csv',
                    help='Path to CSV containing global feature columns keyed on unique_id')
parser.add_argument('--filter_fep',         action='store_true',
                    help='Filter out rows with max_tanimoto_fep_benchmark >= 0.9 from all sources')
args = parser.parse_args()

print(f"Dataset name : {args.name}")
print(f"Split CSV    : {args.pdbbind_split_csv}")
print(f"Global feats : {args.global_features_csv}")
print(f"Filter FEP   : {args.filter_fep}")

print("loading global features from CSV")
_gf = pd.read_csv(args.global_features_csv, usecols=['unique_id'] + GLOBAL_FEATURE_COLS)
_nan_rows = _gf[GLOBAL_FEATURE_COLS].isna().any(axis=1)
if _nan_rows.any():
    print(f"WARNING: {_nan_rows.sum()} row(s) with NaN global features — imputing with column median:")
    print(_gf.loc[_nan_rows, 'unique_id'].tolist())
    for col in GLOBAL_FEATURE_COLS:
        _gf[col] = _gf[col].fillna(_gf[col].median())
global_dict = {row['unique_id']: np.array(row[GLOBAL_FEATURE_COLS], dtype=np.float32)
               for _, row in _gf.iterrows()}

print("loading graph from pickle file for pdbbind2020")
with open("data/pdbbind.pickle", 'rb') as handle:
    graphs_dict = pickle.load(handle)

pdbbind = pd.read_csv(args.pdbbind_split_csv, index_col=0)
pdbbind = pdbbind[['PDB_code', '-logKd/Ki', 'split_core', 'max_tanimoto_fep_benchmark']]
pdbbind = pdbbind.rename(columns={'PDB_code': 'unique_id', 'split_core': 'split', '-logKd/Ki': 'pK'})
if args.filter_fep:
    pdbbind = pdbbind[pdbbind['max_tanimoto_fep_benchmark'] < 0.9]
data = pdbbind[['unique_id', 'pK', 'split']]
print(data[['split']].value_counts())

# Drop any IDs absent from graphs_dict or global_dict rather than crashing at
# graph-construction time.  This matches the behaviour of the prediction pipeline
# which silently skips complexes that failed graph generation (rare elements /
# RDKit failures) or are missing from metamodel_data.csv.
_all_ids = set(data['unique_id'])
_missing_graphs = _all_ids - set(graphs_dict.keys())
_missing_global = _all_ids - set(global_dict.keys())
_drop = _missing_graphs | _missing_global
if _drop:
    if _missing_graphs:
        print(f"WARNING: {len(_missing_graphs)} IDs absent from pdbbind.pickle — skipping: {sorted(_missing_graphs)}")
    if _missing_global:
        print(f"WARNING: {len(_missing_global)} IDs absent from metamodel_data.csv — skipping: {sorted(_missing_global)}")
    data = data[~data['unique_id'].isin(_drop)]
    print(f"Proceeding with {len(data)} entries (dropped {len(_drop)})")

dataset = args.name

for split in ['train', 'valid', 'test']:
    df = data[data['split'] == split]
    ids, y = list(df['unique_id']), list(df['pK'])
    missing = [uid for uid in ids if uid not in global_dict]
    if missing:
        print(f'WARNING: {len(missing)} {split} IDs missing from global features, skipping: {missing}')
        pairs = [(uid, pk) for uid, pk in zip(ids, y) if uid in global_dict]
        ids, y = [p[0] for p in pairs], [p[1] for p in pairs]
    print(f'preparing {dataset}_{split}.pt ({len(ids)} samples)')
    if split == 'train':
        train_data = GraphDataset(root='data', dataset=dataset + '_' + split,
                                  ids=ids, y=y, graphs_dict=graphs_dict, global_dict=global_dict)
    else:
        GraphDataset(root='data', dataset=dataset + '_' + split,
                     ids=ids, y=y, graphs_dict=graphs_dict,
                     y_scaler=train_data.y_scaler, global_dict=global_dict)
