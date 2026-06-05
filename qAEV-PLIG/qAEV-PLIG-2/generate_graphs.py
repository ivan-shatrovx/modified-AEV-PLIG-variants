"""
generate_graphs.py — Graph generation for AEV-PLIG-Coulomb-DDD.

Identical to generate_graphs.py except:
  - Uses distance-dependent dielectric (DDD) Coulomb functions from helpers_qaev_plig_2.py
  - Larger Coulomb cutoff (12 Å vs 5.1 Å)
  - Writes output to a separate pickle (default: data/graphs_ddd.pickle)

Output tuple format: (c_size, features, edge_index, edge_attr, coulomb_energy_label)
Node feature dimension: 369  (367 original + charge(1) + coulomb_potential(1))
"""

import pandas as pd
import pickle
import torch
import torchani
import torchani_mod
import qcelemental as qcel
import numpy as np
from tqdm import tqdm
import os
from rdkit import Chem

from config import (
    AEV_RCR, AEV_ETAR, AEV_RSR,
    AEV_RCA, AEV_ZETA, AEV_TSA, AEV_ETAA, AEV_RSA,
    AEV_NUM_SPECIES,
    ATOM_KEYS_CSV, DATASET_CSV,
    ELEMENT_LIST,
)
from config_qaev_plig_2 import COULOMB_CUTOFF
from helpers_qaev_plig_2 import (
    load_charges, get_heavy_atom_charges,
    get_protein_coords_and_charges,
    compute_coulomb_potential_ddd, compute_coulomb_energy_ddd,
)


# ═══════════════════════════════════════════════════════════════════
# Reused graph-building helpers (identical to generate_graphs.py)
# ═══════════════════════════════════════════════════════════════════

def elements_to_atomicnums(elements):
    atomicnums = np.zeros(len(elements), dtype=int)
    for idx, e in enumerate(elements):
        atomicnums[idx] = qcel.periodictable.to_Z(e)
    return atomicnums


def LoadMolasDF(mol):
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)
    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
    return df


def LoadPDBasDF(PDB, atom_keys):
    prot_atoms = []
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            if (len(i[12:16].replace(" ", "")) < 4 and i[12:16].replace(" ", "")[0] != "H") or \
               (len(i[12:16].replace(" ", "")) == 4 and i[12:16].replace(" ", "")[1] != "H" and i[12:16].replace(" ", "")[0] != "H"):
                prot_atoms.append([int(i[6:11]),
                                   i[17:20] + "-" + i[12:16].replace(" ", ""),
                                   float(i[30:38]),
                                   float(i[38:46]),
                                   float(i[46:54])])
    f.close()
    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = df.merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[
        ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
    ].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    if list(df["ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types.")
    return df


def GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map):
    Target = LoadPDBasDF(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)

    RcR = radial_coefs[0]
    EtaR = radial_coefs[1]
    RsR = radial_coefs[2]

    RcA = 2.0
    Zeta = torch.tensor([1.0])
    TsA = torch.tensor([1.0])
    EtaA = torch.tensor([1.0])
    RsA = torch.tensor([1.0])

    distance_cutoff = RcR + 0.1
    for i in ["X", "Y", "Z"]:
        Target = Target[Target[i] < float(Ligand[i].max()) + distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min()) - distance_cutoff]

    Target = Target.merge(atom_map, on='ATOM_TYPE', how='left')

    mol_len = torch.tensor(len(Ligand))
    atomicnums = np.append(np.ones(mol_len) * 6, Target["ATOM_NR"])
    atomicnums = torch.tensor(atomicnums, dtype=torch.int64).unsqueeze(0)

    coordinates = pd.concat([Ligand[['X', 'Y', 'Z']], Target[['X', 'Y', 'Z']]])
    coordinates = torch.tensor(coordinates.values).unsqueeze(0)

    atom_symbols = []
    for i in range(1, AEV_NUM_SPECIES + 1):
        atom_symbols.append(qcel.periodictable.to_symbol(i))

    AEVC = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR,
                                     EtaA, Zeta, RsA, TsA, len(atom_symbols))
    SC = torchani.SpeciesConverter(atom_symbols)
    sc = SC((atomicnums, coordinates))
    aev = AEVC.forward((sc.species, sc.coordinates), mol_len)

    n = len(atom_symbols)
    n_rad_sub = len(EtaR) * len(RsR)
    indices = list(np.arange(n * n_rad_sub))

    return Ligand, aev.aevs.squeeze(0)[:mol_len, indices]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    feature_list = []
    feature_list.extend(one_of_k_encoding(atom.GetSymbol(), ELEMENT_LIST))
    feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"]))
    feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"]))
    feature_list.append(atom.GetExplicitValence())
    feature_list.append(1 if atom.GetIsAromatic() else 0)
    feature_list.append(1 if atom.IsInRing() else 0)
    return np.array(feature_list)


def mol_to_graph(mol, mol_df, aevs, heavy_atom_charges, coulomb_potentials, coulomb_energy):
    """Build graph with DDD Coulomb features. Returns 5-element tuple.

    Node features: [atom_features(15) | AEV(352) | charge(1) | coulomb_potential(1)] = 369
    """
    features = []
    heavy_atom_index = []
    idx_to_idx = {}
    counter = 0

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df['ATOM_INDEX'] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())

            base_features = atom_features(atom)
            aev_features = aevs[aev_idx, :].numpy().flatten()
            charge_feat = np.array([heavy_atom_charges[counter]])
            potential_feat = np.array([coulomb_potentials[counter]])

            feature = np.concatenate([base_features, aev_features, charge_feat, potential_feat])
            features.append(feature)
            counter += 1

    edges = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            bond_type = one_of_k_encoding(bond.GetBondType(), [1, 12, 2, 3])
            bond_type = [float(b) for b in bond_type]
            edge1 = [idx_to_idx[idx1], idx_to_idx[idx2]]
            edge1.extend(bond_type)
            edge2 = [idx_to_idx[idx2], idx_to_idx[idx1]]
            edge2.extend(bond_type)
            edges.append(edge1)
            edges.append(edge2)

    df = pd.DataFrame(edges, columns=['atom1', 'atom2', 'single', 'aromatic', 'double', 'triple'])
    df = df.sort_values(by=['atom1', 'atom2'])

    edge_index = df[['atom1', 'atom2']].to_numpy().tolist()
    edge_attr = df[['single', 'aromatic', 'double', 'triple']].to_numpy().tolist()

    return len(mol_df), features, edge_index, edge_attr, coulomb_energy


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate graphs for AEV-PLIG-Coulomb-DDD")
    parser.add_argument("--csv", type=str, default=DATASET_CSV,
                        help="Path to dataset CSV")
    parser.add_argument("--output", type=str, default="data/graphs_ddd.pickle",
                        help="Output pickle path")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    print(f"Number of complexes: {len(data)}")

    atom_keys = pd.read_csv(ATOM_KEYS_CSV, sep=",")
    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    radial_coefs = [AEV_RCR, AEV_ETAR, AEV_RSR]

    mol_graphs = {}
    failed_read = []
    failed_process = []

    for i, row in tqdm(data.iterrows(), total=len(data)):
        uid = row["unique_id"]
        sdf_path = row["sdf_file"]
        pdb_path = row["pdb_file"]
        charges_path = row["charges_json"]

        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if mol is None:
            print(f"Can't read ligand: {uid} ({sdf_path})")
            failed_read.append(uid)
            continue
        mol = Chem.AddHs(mol, addCoords=True)

        try:
            charges = load_charges(charges_path)
            heavy_charges = get_heavy_atom_charges(mol, charges)
            prot_coords, prot_charges = get_protein_coords_and_charges(pdb_path, charges)

            lig_coords = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() != "H":
                    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    lig_coords.append([pos.x, pos.y, pos.z])
            lig_coords = np.array(lig_coords)

            potentials = compute_coulomb_potential_ddd(lig_coords, prot_coords, prot_charges,
                                                       cutoff=COULOMB_CUTOFF)
            energy = compute_coulomb_energy_ddd(lig_coords, np.array(heavy_charges),
                                                prot_coords, prot_charges, cutoff=COULOMB_CUTOFF)

            mol_df, aevs = GetMolAEVs_extended(pdb_path, mol, atom_keys, radial_coefs, atom_map)

            graph = mol_to_graph(mol, mol_df, aevs, heavy_charges, potentials, energy)
            mol_graphs[uid] = graph

        except Exception as e:
            print(f"Failed processing {uid}: {e}")
            failed_process.append(uid)
            continue

    print(f"\nDone. Succeeded: {len(mol_graphs)}, "
          f"Failed read: {len(failed_read)}, Failed process: {len(failed_process)}")

    if failed_read:
        print(f"Failed to read: {failed_read}")
    if failed_process:
        print(f"Failed to process: {failed_process}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'wb') as handle:
        pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(mol_graphs)} graphs to {args.output}")

    if failed_read or failed_process:
        failures_file = args.output.replace(".pickle", "_failures.txt")
        with open(failures_file, 'w') as f:
            for uid in failed_read:
                f.write(f"read_fail\t{uid}\n")
            for uid in failed_process:
                f.write(f"process_fail\t{uid}\n")
        print(f"Failures saved to {failures_file}")
