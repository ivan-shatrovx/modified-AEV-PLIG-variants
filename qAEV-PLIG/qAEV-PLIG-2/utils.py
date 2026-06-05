"""
utils.py — Dataset classes for AEV-PLIG-Coulomb-DDD.

Key differences from utils.py:
  - processed_dir overridden to 'processed_ddd/' so DDD .pt files don't
    collide with the first-model files in 'processed/'.
  - coulomb_energy is NOT re-scaled here; it is z-score normalised externally
    in create_pytorch_data_ddd.py and stored pre-normalised in the .pt files.
  - GraphDatasetPredict_DDD stores a dummy coulomb_energy=0 because the DDD
    model's forward() does not read data.coulomb_energy.
"""

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from sklearn.preprocessing import StandardScaler


def init_weights(layer):
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is True:
            torch.nn.init.zeros_(layer.bias)


class GraphDataset_DDD(InMemoryDataset):
    """PyG InMemoryDataset for AEV-PLIG-Coulomb-DDD (training / evaluation).

    coulomb_energy in each Data object is the **pre-normalised** z-score value
    produced by create_pytorch_data_ddd.py.  This class does NOT apply any
    additional scaling to it.
    """

    def __init__(self, root='data', dataset=None,
                 ids=None, y=None, graphs_dict=None,
                 y_scaler=None):

        super(GraphDataset_DDD, self).__init__(root)
        self.dataset = dataset
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])
        else:
            self.process(ids, y, graphs_dict)
            self.load(self.processed_paths[0])

        # Scale y (pK) — identical to GraphDataset in utils.py
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(self._data.y, (self.__len__(), 1)))
        self.y_scaler = y_scaler
        self._data.y = [
            torch.tensor(element[0]).float()
            for element in self.y_scaler.transform(
                np.reshape(self._data.y, (self.__len__(), 1))
            )
        ]

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_ddd')

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, y, graphs_dict):
        assert len(ids) == len(y), 'Number of datapoints and labels must be the same'
        data_list = []
        for i in range(len(ids)):
            pdbcode = ids[i]
            label = y[i]
            c_size, features, edge_index, edge_features, coulomb_energy = graphs_dict[pdbcode]
            data_point = Data(
                x=torch.Tensor(np.array(features)),
                edge_index=torch.LongTensor(np.array(edge_index)).T,
                edge_attr=torch.Tensor(np.array(edge_features)),
                y=torch.FloatTensor(np.array([label])),
                coulomb_energy=torch.FloatTensor([coulomb_energy]),
            )
            data_list.append(data_point)

        print('Graph construction done. Saving to file.')
        self.save(data_list, self.processed_paths[0])


class GraphDatasetPredict_DDD(InMemoryDataset):
    """PyG InMemoryDataset for DDD model inference.

    Stores graphs without meaningful coulomb_energy (the DDD forward pass
    does not use it as input).
    """

    def __init__(self, root='data', dataset=None,
                 ids=None, graph_ids=None, graphs_dict=None):

        super(GraphDatasetPredict_DDD, self).__init__(root)
        self.dataset = dataset
        torch.serialization.add_safe_globals([Data])
        if os.path.isfile(self.processed_paths[0]):
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])
        else:
            self.process(ids, graph_ids, graphs_dict)
            self.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_ddd')

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, graph_ids, graphs_dict):
        assert len(ids) == len(graph_ids), 'Number of datapoints and labels must be the same'
        data_list = []
        for i in range(len(ids)):
            pdbcode = ids[i]
            graph_id = graph_ids[i]
            c_size, features, edge_index, edge_features, coulomb_energy = graphs_dict[pdbcode]
            data_point = Data(
                x=torch.Tensor(np.array(features)),
                edge_index=torch.LongTensor(np.array(edge_index)).T,
                edge_attr=torch.Tensor(np.array(edge_features)),
                y=torch.IntTensor(np.array([graph_id])),
                coulomb_energy=torch.FloatTensor([coulomb_energy]),
            )
            data_list.append(data_point)

        print('Graph construction done. Saving to file.')
        self.save(data_list, self.processed_paths[0])
