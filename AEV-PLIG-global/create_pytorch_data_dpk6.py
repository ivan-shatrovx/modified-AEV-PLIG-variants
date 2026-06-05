import argparse
import numpy as np
import pandas as pd
import pickle
from utils import GraphDataset

DPK_DIRECT_COLS  = ['dpk_volume_score', 'dpk_prop_polar_atm', 'dpk_mean_loc_hyd_dens',
                     'dpk_charge_score', 'dpk_flex']
DPK_AROMATIC_COLS = ['dpk_PHE', 'dpk_HIS', 'dpk_TYR', 'dpk_TRP']
OUTPUT_COLS = DPK_DIRECT_COLS + ['dpk_aromatic']  # 6 features total

parser = argparse.ArgumentParser(description="Create PyTorch Geometric datasets (Global-6 / DPK6 model)")
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
_gf = pd.read_csv(args.global_features_csv, usecols=['unique_id'] + DPK_DIRECT_COLS + DPK_AROMATIC_COLS)
_nan_rows = _gf[DPK_DIRECT_COLS + DPK_AROMATIC_COLS].isna().any(axis=1)
if _nan_rows.any():
    print(f"WARNING: {_nan_rows.sum()} row(s) with NaN global features — imputing with column median:")
    print(_gf.loc[_nan_rows, 'unique_id'].tolist())
    for col in DPK_DIRECT_COLS + DPK_AROMATIC_COLS:
        _gf[col] = _gf[col].fillna(_gf[col].median())
_gf['dpk_aromatic'] = _gf[DPK_AROMATIC_COLS].sum(axis=1)
global_dict = {row['unique_id']: np.array(row[OUTPUT_COLS], dtype=np.float32)
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
