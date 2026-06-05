"""
model_defs.py — GATv2Net with dual output heads for AEV-PLIG-Coulomb-DDD.

Key differences from model_defs.py:
  - Two independent MLP heads branch from the shared pooled representation:
      * fc_pk  head → pK prediction
      * fc_ec  head → E_coulomb prediction (in normalised space)
  - forward() returns (pk, ec) tuple — two tensors of shape [B, 1] each.
  - data.coulomb_energy is NOT read inside forward(); it is only used in the
    training loss.  The model must learn electrostatics from node features alone.
  - No graph_level_dim / coulomb_dim parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import BatchNorm


activation_function_dict = {
    "relu":       F.relu,
    "leaky_relu": F.leaky_relu,
}


class GATv2Net_DDD(torch.nn.Module):
    def __init__(self, node_feature_dim=369, edge_feature_dim=4, config=None,
                 # legacy keyword alias used by training script
                 num_features_xd=None):
        super(GATv2Net_DDD, self).__init__()

        if num_features_xd is not None and node_feature_dim == 369:
            node_feature_dim = num_features_xd

        self.number_GNN_layers = 5
        self.act = config.activation_function if config is not None else "leaky_relu"
        self.activation = activation_function_dict[self.act]
        self.dropout = getattr(config, 'dropout', 0.2) if config is not None else 0.2

        head       = config.head       if config is not None else 3
        hidden_dim = config.hidden_dim if config is not None else 256

        self.GNN_layers = nn.ModuleList()
        self.BN_layers  = nn.ModuleList()

        self.GNN_layers.append(GATv2Conv(node_feature_dim, hidden_dim,
                                         heads=head, edge_dim=edge_feature_dim))
        self.BN_layers.append(BatchNorm(hidden_dim * head))
        for _ in range(1, self.number_GNN_layers):
            self.GNN_layers.append(GATv2Conv(hidden_dim * head, hidden_dim,
                                             heads=head, edge_dim=edge_feature_dim))
            self.BN_layers.append(BatchNorm(hidden_dim * head))

        final_dim = hidden_dim * head   # dimension of pooled representation

        # ── pK head ──────────────────────────────────────────────────────────
        self.fc_pk1 = nn.Linear(final_dim * 2, 1024)
        self.bn_pk1 = nn.BatchNorm1d(1024)
        self.fc_pk2 = nn.Linear(1024, 512)
        self.bn_pk2 = nn.BatchNorm1d(512)
        self.fc_pk3 = nn.Linear(512, 256)
        self.bn_pk3 = nn.BatchNorm1d(256)
        self.out_pk = nn.Linear(256, 1)

        # ── Coulomb energy head ───────────────────────────────────────────────
        self.fc_ec1 = nn.Linear(final_dim * 2, 256)
        self.out_ec = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        for layer, bn in zip(self.GNN_layers, self.BN_layers):
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = bn(x)

        # Shared pooled representation
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)   # [B, 2*final_dim]

        # pK head
        pk = self.activation(self.fc_pk1(x))
        pk = self.bn_pk1(pk)
        pk = F.dropout(pk, p=self.dropout, training=self.training)
        pk = self.activation(self.fc_pk2(pk))
        pk = self.bn_pk2(pk)
        pk = self.activation(self.fc_pk3(pk))
        pk = self.bn_pk3(pk)
        pk = self.out_pk(pk)                                     # [B, 1]

        # Coulomb energy head
        ec = F.relu(self.fc_ec1(x))
        ec = self.out_ec(ec)                                     # [B, 1]

        return pk, ec
