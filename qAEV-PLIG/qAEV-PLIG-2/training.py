"""
training.py — Multi-task training for AEV-PLIG-Coulomb-DDD.

Differences from training.py:
  - Uses GATv2Net_DDD (dual output: pK + E_coulomb)
  - Multi-task MSE loss: LOSS_WEIGHT_PK * mse_pk + LOSS_WEIGHT_COULOMB * mse_ec
  - Checkpoints on best validation pK RMSE (lower is better)
  - Saves checkpoints as {timestr}_model_GATv2Net_DDD_{dataset}_{seed}.model
  - Saves scaler as {timestr}_model_GATv2Net_DDD_{dataset}.pickle
  - Logs per-epoch pK loss and E_coulomb loss separately
"""

import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from math import sqrt
import numpy as np
import os
import pandas as pd
import argparse
import pickle

from helpers import rmse, pearson
from utils import GraphDataset_DDD, init_weights
from model_defs import GATv2Net_DDD
from config_ddd import LOSS_WEIGHT_PK, LOSS_WEIGHT_COULOMB
from config import DATASET_CSV


def predict_ddd(model, device, loader, y_scaler):
    """Return (labels_pk, preds_pk) both in original pK scale."""
    model.eval()
    total_pk_preds  = torch.Tensor()
    total_pk_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pk, _ec = model(data)
            total_pk_preds  = torch.cat((total_pk_preds,  pk.cpu()), 0)
            total_pk_labels = torch.cat((total_pk_labels, data.y.view(-1, 1).cpu()), 0)

    labels = y_scaler.inverse_transform(
        total_pk_labels.numpy().flatten().reshape(-1, 1)
    ).flatten()
    preds = y_scaler.inverse_transform(
        total_pk_preds.detach().numpy().flatten().reshape(-1, 1)
    ).flatten()
    return labels, preds


def train_epoch(model, device, train_loader, optimizer, epoch):
    """Run one training epoch; return (avg_total_loss, avg_pk_loss, avg_ec_loss)."""
    log_interval = 100
    model.train()
    total_loss = total_pk_loss = total_ec_loss = 0.0
    n_samples = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        pk_pred, ec_pred = model(data)
        pk_true = data.y.view(-1, 1).to(device)
        ec_true = data.coulomb_energy.view(-1, 1).to(device)

        loss_pk = F.mse_loss(pk_pred, pk_true)
        loss_ec = F.mse_loss(ec_pred, ec_true)
        loss    = LOSS_WEIGHT_PK * loss_pk + LOSS_WEIGHT_COULOMB * loss_ec

        loss.backward()
        optimizer.step()

        bs = len(data.y)
        total_loss    += loss.item()    * bs
        total_pk_loss += loss_pk.item() * bs
        total_ec_loss += loss_ec.item() * bs
        n_samples     += bs

        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch,
                batch_idx * bs,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
            ))

    n = n_samples if n_samples > 0 else 1
    avg_total = total_loss    / n
    avg_pk    = total_pk_loss / n
    avg_ec    = total_ec_loss / n
    print(f"Epoch {epoch} — total_loss: {avg_total:.4f}  "
          f"pk_loss: {avg_pk:.4f}  ec_loss: {avg_ec:.4f}")
    return avg_total, avg_pk, avg_ec


def _train(model, device, train_loader, valid_loader, optimizer, n_epochs,
           y_scaler, model_output_dir, model_file_name):
    """Training loop; checkpoint on best validation pK RMSE."""
    best_rmse = float('inf')
    rmse_history = []

    for epoch in range(n_epochs):
        train_epoch(model, device, train_loader, optimizer, epoch + 1)

        G, P = predict_ddd(model, device, valid_loader, y_scaler)
        val_rmse = rmse(G, P)
        val_pc   = pearson(G, P)
        rmse_history.append(val_rmse)

        # Use 8-epoch window average for stable checkpointing
        low = max(epoch - 7, 0)
        avg_rmse = np.mean(rmse_history[low:epoch + 1])
        if avg_rmse < best_rmse:
            torch.save(model.state_dict(),
                       os.path.join(model_output_dir, model_file_name))
            best_rmse = avg_rmse

        print(f'Validation — pK RMSE: {val_rmse:.4f}  Pearson r: {val_pc:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',             type=str,   default='qaev_plig_2')
    parser.add_argument('--batch_size',          type=int,   default=128)
    parser.add_argument('--epochs',              type=int,   default=200)
    parser.add_argument('--hidden_dim',          type=int,   default=256)
    parser.add_argument('--head',                type=int,   default=3)
    parser.add_argument('--lr',                  type=float, default=0.00012291937615434127)
    parser.add_argument('--activation_function', type=str,   default='leaky_relu')
    parser.add_argument('--dropout',             type=float, default=0.2)
    args = parser.parse_args()
    return args


def train_NN(args):
    dataset = args.dataset
    print(f'Running qAEV-PLIG-2 model on dataset: {dataset}')

    timestr         = time.strftime("%Y%m%d-%H%M%S")
    model_output_dir = os.path.join("output", "trained_models")
    os.makedirs(model_output_dir, exist_ok=True)

    train_data = GraphDataset_DDD(root='data', dataset=dataset + '_train',  y_scaler=None)
    valid_data = GraphDataset_DDD(root='data', dataset=dataset + '_valid',  y_scaler=train_data.y_scaler)
    test_data  = GraphDataset_DDD(root='data', dataset=dataset + '_test',   y_scaler=train_data.y_scaler)

    y_scaler = train_data.y_scaler

    print(f"Node features:  {train_data.num_node_features}")
    print(f"Edge features:  {train_data.num_edge_features}")

    seeds = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]

    for i, seed in enumerate(seeds):
        random.seed(seed)
        torch.manual_seed(seed)

        model_file_name = (f"{timestr}_model_GATv2Net_DDD_{dataset}_{i}.model")

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Seed {i}: device={device}')

        model = GATv2Net_DDD(
            node_feature_dim=train_data.num_node_features,
            edge_feature_dim=train_data.num_edge_features,
            config=args,
        )
        model.apply(init_weights)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        _train(model, device, train_loader, valid_loader, optimizer,
               args.epochs, y_scaler, model_output_dir, model_file_name)

        model.load_state_dict(
            torch.load(os.path.join(model_output_dir, model_file_name))
        )

        G_test, P_test = predict_ddd(model, device, test_loader, y_scaler)

        if i == 0:
            df_test = pd.DataFrame(data=G_test, columns=['truth'])

        df_test[f'preds_{i}'] = P_test

    df_test['preds'] = df_test.iloc[:, 1:].mean(axis=1)

    # Save y_scaler
    scaler_file = os.path.join(
        model_output_dir,
        f"{timestr}_model_GATv2Net_DDD_{dataset}.pickle",
    )
    with open(scaler_file, 'wb') as f:
        pickle.dump({'y_scaler': y_scaler}, f)

    test_preds = np.array(df_test['preds'])
    test_truth = np.array(df_test['truth'])
    print(f"Ensemble test Pearson r: {pearson(test_truth, test_preds):.4f}")
    print(f"Ensemble test RMSE:      {rmse(test_truth, test_preds):.4f}")


if __name__ == "__main__":
    start = time.time()
    args = parse_args()
    train_NN(args)
    print(f"Total time: {time.time() - start:.1f}s")
