"""
process_and_predict.py — Data processing and inference for AEV-PLIG-Coulomb-DDD.

Differences from process_and_predict.py:
  - Uses generate_graphs_ddd functions (DDD Coulomb, 12 Å cutoff)
  - Loads GATv2Net_DDD which returns (pk, ec) tuple
  - Un-normalises Coulomb prediction using processed_ddd/coulomb_norm_stats.json
  - Prints / saves both pK and E_coulomb (kcal/mol) per complex
"""

import pandas as pd
import pickle
import json
import torch
import qcelemental as qcel
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from biopandas.pdb import PandasPdb
import os
import sys
import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import torchani
import torchani_mod

warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.ase")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.data")

from utils import GraphDatasetPredict_DDD
from torch_geometric.loader import DataLoader
from model_defs import GATv2Net_DDD
from helpers_ddd import (
    load_charges, get_heavy_atom_charges,
    get_protein_coords_and_charges,
    compute_coulomb_potential_ddd, compute_coulomb_energy_ddd,
)
from config_ddd import COULOMB_CUTOFF
from config import ELEMENT_LIST, ATOM_KEYS_CSV, AEV_RCR, AEV_ETAR, AEV_RSR, AEV_NUM_SPECIES


# ═══════════════════════════════════════════════════════════════════
# Graph-building helpers (mirrors generate_graphs_ddd.py)
# ═══════════════════════════════════════════════════════════════════

def LoadMolasDF(mol):
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            entry = [int(atom.GetIdx()), str(atom.GetSymbol())]
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            entry += [float(f"{pos.x:.4f}"), float(f"{pos.y:.4f}"), float(f"{pos.z:.4f}")]
            atoms.append(entry)
    df = pd.DataFrame(atoms, columns=["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"])
    return df


def LoadPDBasDF_old(PDB, atom_keys):
    prot_atoms = []
    with open(PDB) as f:
        for line in f:
            if line[:4] == "ATOM":
                name = line[12:16].replace(" ", "")
                if (len(name) < 4 and name[0] != "H") or \
                   (len(name) == 4 and name[1] != "H" and name[0] != "H"):
                    prot_atoms.append([
                        int(line[6:11]),
                        line[17:20] + "-" + name,
                        float(line[30:38]), float(line[38:46]), float(line[46:54]),
                    ])
    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = df.merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[
        ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
    ].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    return df


def LoadPDBasDF(pdb_path, atom_keys):
    allowed_residues = atom_keys["RESIDUE"].unique()
    ppdb = PandasPdb().read_pdb(pdb_path)
    protein = ppdb.df['ATOM']
    protein = protein[~protein["atom_name"].str.startswith("H")]
    protein = protein[~protein["atom_name"].str.startswith(tuple(map(str, range(10))))]
    disgard = protein[~protein["residue_name"].isin(allowed_residues)]
    if len(disgard) > 0:
        print(f"WARNING: unsupported residues in {pdb_path}:", disgard["residue_name"].unique())
    protein = protein[protein["residue_name"].isin(allowed_residues)]
    protein["PDB_ATOM"] = protein["residue_name"] + "-" + protein["atom_name"]
    protein = protein[['atom_number', 'PDB_ATOM', 'x_coord', 'y_coord', 'z_coord']].rename(
        columns={"atom_number": "ATOM_INDEX", "x_coord": "X", "y_coord": "Y", "z_coord": "Z"}
    )
    protein = protein.merge(atom_keys, how='left', on='PDB_ATOM').sort_values(by="ATOM_INDEX").reset_index(drop=True)
    return protein


def GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map):
    Target = LoadPDBasDF_old(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)

    RcR, EtaR, RsR = radial_coefs
    RcA = 2.0
    Zeta = torch.tensor([1.0])
    TsA  = torch.tensor([1.0])
    EtaA = torch.tensor([1.0])
    RsA  = torch.tensor([1.0])

    dc = RcR + 0.1
    for ax in ["X", "Y", "Z"]:
        Target = Target[Target[ax] < float(Ligand[ax].max()) + dc]
        Target = Target[Target[ax] > float(Ligand[ax].min()) - dc]

    Target = Target.merge(atom_map, on='ATOM_TYPE', how='left')

    mol_len   = torch.tensor(len(Ligand))
    atomicnums = np.append(np.ones(mol_len) * 6, Target["ATOM_NR"])
    atomicnums = torch.tensor(atomicnums, dtype=torch.int64).unsqueeze(0)

    coordinates = pd.concat([Ligand[['X', 'Y', 'Z']], Target[['X', 'Y', 'Z']]])
    coordinates = torch.tensor(coordinates.values).unsqueeze(0)

    atom_symbols = [qcel.periodictable.to_symbol(i) for i in range(1, AEV_NUM_SPECIES + 1)]
    AEVC = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, len(atom_symbols))
    SC   = torchani.SpeciesConverter(atom_symbols)
    sc   = SC((atomicnums, coordinates))
    aev  = AEVC.forward((sc.species, sc.coordinates), mol_len)

    n_rad_sub = len(EtaR) * len(RsR)
    indices   = list(np.arange(len(atom_symbols) * n_rad_sub))
    return Ligand, aev.aevs.squeeze(0)[:mol_len, indices]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def atom_features(atom):
    fl = []
    fl.extend(one_of_k_encoding(atom.GetSymbol(), ELEMENT_LIST))
    fl.append(len([n for n in atom.GetNeighbors() if n.GetSymbol() != "H"]))
    fl.append(len([n for n in atom.GetNeighbors() if n.GetSymbol() == "H"]))
    fl.append(atom.GetExplicitValence())
    fl.append(1 if atom.GetIsAromatic() else 0)
    fl.append(1 if atom.IsInRing() else 0)
    return np.array(fl)


def mol_to_graph(mol, mol_df, aevs, heavy_charges, coulomb_potentials, coulomb_energy):
    features, heavy_atom_index, idx_to_idx = [], [], {}
    counter = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df['ATOM_INDEX'] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())
            feat = np.concatenate([
                atom_features(atom),
                aevs[aev_idx, :].numpy().flatten(),
                np.array([heavy_charges[counter]]),
                np.array([coulomb_potentials[counter]]),
            ])
            features.append(feat)
            counter += 1

    edges = []
    for bond in mol.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i1 in heavy_atom_index and i2 in heavy_atom_index:
            bt = [float(b) for b in one_of_k_encoding(bond.GetBondType(), [1, 12, 2, 3])]
            edges.append([idx_to_idx[i1], idx_to_idx[i2]] + bt)
            edges.append([idx_to_idx[i2], idx_to_idx[i1]] + bt)

    df = pd.DataFrame(edges, columns=['atom1', 'atom2', 'single', 'aromatic', 'double', 'triple'])
    df = df.sort_values(by=['atom1', 'atom2'])
    return (len(mol_df),
            features,
            df[['atom1', 'atom2']].to_numpy().tolist(),
            df[['single', 'aromatic', 'double', 'triple']].to_numpy().tolist(),
            coulomb_energy)


# ═══════════════════════════════════════════════════════════════════
# Prediction helper
# ═══════════════════════════════════════════════════════════════════

def predict(model, device, loader, y_scaler):
    """Return (graph_ids, pk_preds, ec_preds_normalised)."""
    model.eval()
    model.to(device)
    all_graph_ids = torch.IntTensor().to(device)
    all_pk        = torch.Tensor().to(device)
    all_ec        = torch.Tensor().to(device)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pk, ec = model(data)
            all_graph_ids = torch.cat((all_graph_ids, data.y.view(-1, 1)), 0)
            all_pk        = torch.cat((all_pk, pk), 0)
            all_ec        = torch.cat((all_ec, ec), 0)

    graph_ids = all_graph_ids.cpu().numpy().flatten()
    pk_preds  = y_scaler.inverse_transform(
        all_pk.cpu().detach().numpy().flatten().reshape(-1, 1)
    ).flatten()
    ec_norm   = all_ec.cpu().detach().numpy().flatten()
    return graph_ids, pk_preds, ec_norm


# ═══════════════════════════════════════════════════════════════════
# Validation / graph-generation helpers
# ═══════════════════════════════════════════════════════════════════

def validate_row(row, atom_keys):
    try:
        LoadPDBasDF(row["pdb_file"], atom_keys)
        return None
    except Exception:
        return row["unique_id"]


def process_single_graph(row_dict, atom_keys, radial_coefs, atom_map):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    mol = Chem.SDMolSupplier(row_dict["sdf_file"], removeHs=False)[0]
    mol = Chem.AddHs(mol, addCoords=True)

    charges     = load_charges(row_dict["charges_json"])
    heavy_charges = get_heavy_atom_charges(mol, charges)
    prot_coords, prot_charges = get_protein_coords_and_charges(
        row_dict["pdb_file"], charges)

    lig_coords = np.array([
        [pos.x, pos.y, pos.z]
        for atom in mol.GetAtoms() if atom.GetSymbol() != "H"
        for pos in [mol.GetConformer().GetAtomPosition(atom.GetIdx())]
    ])

    potentials = compute_coulomb_potential_ddd(lig_coords, prot_coords, prot_charges,
                                               cutoff=COULOMB_CUTOFF)
    energy     = compute_coulomb_energy_ddd(lig_coords, np.array(heavy_charges),
                                            prot_coords, prot_charges, cutoff=COULOMB_CUTOFF)

    mol_df, aevs = GetMolAEVs_extended(row_dict["pdb_file"], mol,
                                       atom_keys, radial_coefs, atom_map)
    graph = mol_to_graph(mol, mol_df, aevs, heavy_charges, potentials, energy)
    return row_dict["unique_id"], graph


# ═══════════════════════════════════════════════════════════════════
# Pipeline stages
# ═══════════════════════════════════════════════════════════════════

def process_data(config):
    df = pd.read_csv(config.dataset_csv)
    print("Checking ligands readable by RDKit…")
    allowed_elements = set(ELEMENT_LIST)
    non_readable, rare_atoms_ids = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        suppl = Chem.SDMolSupplier(row["sdf_file"], removeHs=False)
        lig   = suppl[0] if len(suppl) > 0 else None
        if lig is None:
            non_readable.append(row["unique_id"])
        else:
            heavy_elements = {a.GetSymbol() for a in lig.GetAtoms() if a.GetSymbol() != "H"}
            if not heavy_elements.issubset(allowed_elements):
                rare_atoms_ids.append(row["unique_id"])

    df = df[~df["unique_id"].isin(non_readable + rare_atoms_ids)].reset_index(drop=True)

    if not config.skip_validation:
        print("Checking proteins readable by Biopandas…")
        atom_keys = pd.read_csv(ATOM_KEYS_CSV, sep=",")
        atom_keys["RESIDUE"] = atom_keys["PDB_ATOM"].apply(lambda x: x.split("-")[0])
        validate = partial(validate_row, atom_keys=atom_keys)
        with ProcessPoolExecutor(max_workers=config.num_workers) as ex:
            bad = [r for r in tqdm(ex.map(validate, df.to_dict("records")), total=len(df)) if r]
        df = df[~df["unique_id"].isin(bad)].reset_index(drop=True)

    out_csv = config.dataset_csv.replace(".csv", "_processed.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved processed CSV: {out_csv}")


def generate_graphs(config):
    processed_csv = config.dataset_csv.replace(".csv", "_processed.csv")
    df        = pd.read_csv(processed_csv)
    atom_keys = pd.read_csv(ATOM_KEYS_CSV, sep=",")
    atom_map  = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map  = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})
    radial_coefs = [AEV_RCR, AEV_ETAR, AEV_RSR]

    mol_graphs = {}
    with ProcessPoolExecutor(max_workers=config.num_workers) as ex:
        futures = [
            ex.submit(process_single_graph, row, atom_keys, radial_coefs, atom_map)
            for row in df.to_dict("records")
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Graphs"):
            try:
                uid, graph = fut.result()
                mol_graphs[uid] = graph
            except Exception as e:
                print("Error:", e)

    out_pickle = f"data/{config.data_name}_graphs_ddd.pickle"
    with open(out_pickle, 'wb') as f:
        pickle.dump(mol_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(mol_graphs)} graphs to {out_pickle}")


def make_predictions(config):
    model_name = config.trained_model_name

    # Load y_scaler
    with open(f'output/trained_models/{model_name}.pickle', 'rb') as f:
        scalers = pickle.load(f)
    y_scaler = scalers['y_scaler'] if isinstance(scalers, dict) else scalers

    # Load Coulomb norm stats for un-normalisation
    stats_path = config.coulomb_norm_stats
    with open(stats_path) as f:
        norm_stats = json.load(f)
    ec_mean = norm_stats["ec_mean"]
    ec_std  = norm_stats["ec_std"]

    processed_csv = config.dataset_csv.replace(".csv", "_processed.csv")
    data = pd.read_csv(processed_csv)

    with open(f"data/{config.data_name}_graphs_ddd.pickle", 'rb') as f:
        graphs_dict = pickle.load(f)

    data["graph_id"] = range(len(data))
    test_ids      = list(data["unique_id"])
    test_graph_ids = list(data["graph_id"])

    pt_path = f"data/processed_ddd/{config.data_name}.pt"
    if os.path.exists(pt_path):
        os.remove(pt_path)

    test_dataset = GraphDatasetPredict_DDD(
        root='data', dataset=config.data_name,
        ids=test_ids, graph_ids=test_graph_ids,
        graphs_dict=graphs_dict,
    )
    test_loader = DataLoader(test_dataset, batch_size=len(data), shuffle=False)

    model = GATv2Net_DDD(
        node_feature_dim=test_dataset.num_node_features,
        edge_feature_dim=test_dataset.num_edge_features,
        config=config,
    )

    for i in range(10):
        model_path = f'output/trained_models/{model_name}_{i}.model'
        model.load_state_dict(torch.load(model_path, map_location=config.device))

        graph_ids, pk_preds, ec_norm = predict(model, config.device, test_loader, y_scaler)

        if i == 0:
            df_out = pd.DataFrame({"graph_id": graph_ids})
        df_out[f'pk_preds_{i}']  = pk_preds
        df_out[f'ec_norm_{i}'] = ec_norm

    df_out['pK_pred']       = df_out[[f'pk_preds_{i}' for i in range(10)]].mean(axis=1)
    ec_norm_ensemble        = df_out[[f'ec_norm_{i}' for i in range(10)]].mean(axis=1).values
    df_out['E_coulomb_pred_kcal'] = ec_norm_ensemble * ec_std + ec_mean

    data = data.merge(df_out[['graph_id', 'pK_pred', 'E_coulomb_pred_kcal']], on='graph_id', how='left')

    out_path = f"output/predictions/{config.data_name}_ddd_predictions.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    print(data[['unique_id', 'pK_pred', 'E_coulomb_pred_kcal']].head())


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_name',   type=str, default='20231116-181233_model_GATv2Net_DDD')
    parser.add_argument('--dataset_csv',           type=str, default='data/example_dataset.csv')
    parser.add_argument('--data_name',             type=str, default='example')
    parser.add_argument('--hidden_dim',            type=int, default=256)
    parser.add_argument('--head',                  type=int, default=3)
    parser.add_argument('--activation_function',   type=str, default='leaky_relu')
    parser.add_argument('--dropout',               type=float, default=0.2)
    parser.add_argument('--num_workers',           type=int, default=0)
    parser.add_argument('--device',                type=str, default='auto')
    parser.add_argument('--skip_validation',       action='store_true')
    parser.add_argument('--coulomb_norm_stats',    type=str,
                        default='data/processed_ddd/coulomb_norm_stats.json',
                        help='Path to coulomb_norm_stats.json from create_pytorch_data.py')
    return parser.parse_args()


def get_device(param):
    if param.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif param.lower() == "cpu":
        return torch.device("cpu")
    else:
        if int(param) >= torch.cuda.device_count():
            sys.exit(f"CUDA device {param} not found")
        return torch.device(f"cuda:{param}")


if __name__ == "__main__":
    config = parse_args()

    if config.num_workers <= 0:
        config.num_workers = os.cpu_count()

    os.environ["OMP_NUM_THREADS"] = str(config.num_workers)
    os.environ["MKL_NUM_THREADS"] = str(config.num_workers)
    torch.set_num_threads(config.num_workers)
    config.device = get_device(config.device)
    print(f"Device: {config.device}")

    t0 = time.time()
    process_data(config)
    t1 = time.time()
    generate_graphs(config)
    print(f"Graph generation: {time.time()-t1:.1f}s")
    make_predictions(config)
    print(f"Total: {time.time()-t0:.1f}s")
