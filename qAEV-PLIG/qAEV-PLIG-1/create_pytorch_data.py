"""
create_pytorch_data.py — Convert graph pickle to PyTorch Geometric datasets.

Reads:
  - data/graphs.pickle (from generate_graphs.py)
  - data/dataset.csv   (with columns: unique_id, pK, split)

Outputs:
  - data/processed/{dataset}_train.pt
  - data/processed/{dataset}_valid.pt
  - data/processed/{dataset}_test.pt
"""

import pandas as pd
import pickle
import argparse
from utils import GraphDataset
from config import DATASET_CSV


def main():
    parser = argparse.ArgumentParser(description="Create PyTorch Geometric datasets")
    parser.add_argument("--graphs", type=str, default="data/graphs.pickle",
                        help="Path to graphs pickle")
    parser.add_argument("--csv", type=str, default=DATASET_CSV,
                        help="Path to dataset CSV")
    parser.add_argument("--dataset_name", type=str, default="qaev_plig_1",
                        help="Dataset name prefix for output files")
    args = parser.parse_args()

    # Load graphs
    print(f"Loading graphs from {args.graphs}")
    with open(args.graphs, 'rb') as handle:
        graphs_dict = pickle.load(handle)
    print(f"Loaded {len(graphs_dict)} graphs")

    # Load dataset CSV
    data = pd.read_csv(args.csv)
    print(data[['split']].value_counts())

    # Filter to only complexes that have graphs
    available = set(graphs_dict.keys())
    before = len(data)
    data = data[data['unique_id'].isin(available)]
    if len(data) < before:
        print(f"Warning: {before - len(data)} complexes in CSV have no graph, skipped")

    dataset = args.dataset_name

    for split in ['train', 'valid', 'test']:
        df = data[data['split'] == split]
        if len(df) == 0:
            print(f"No {split} data, skipping")
            continue
        ids = list(df['unique_id'])
        y = list(df['pK'])
        print(f'Preparing {dataset}_{split}.pt ({len(ids)} samples)')

        if split == 'train':
            train_data = GraphDataset(root='data', dataset=dataset + '_' + split,
                                      ids=ids, y=y, graphs_dict=graphs_dict)
            y_scaler = train_data.y_scaler
            coulomb_scaler = train_data.coulomb_scaler
        else:
            GraphDataset(root='data', dataset=dataset + '_' + split,
                         ids=ids, y=y, graphs_dict=graphs_dict,
                         y_scaler=y_scaler, coulomb_scaler=coulomb_scaler)

    print("Done.")


if __name__ == "__main__":
    main()
