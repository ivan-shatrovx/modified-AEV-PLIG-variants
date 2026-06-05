import json
import torch
import numpy as np
from math import sqrt
from scipy import stats
from model_defs import GATv2Net
from config import COULOMB_K, EPSILON_R, COULOMB_CUTOFF, DISTANCE_EPSILON

model_dict = {"GATv2Net": GATv2Net}

def get_num_parameters(model):
    """
    counts the number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_fn(batch):
    """
    function needed for data loaders
    """
    feature_list, protein_seq_list, label_list = [], [], []
    for _features, _protein_seq, _label in batch:
        #print(type(_features), type(_protein_seq), type(_label))
        feature_list.append(_features)
        protein_seq_list.append(_protein_seq)
        label_list.append(_label)
    return torch.Tensor(feature_list), torch.Tensor(protein_seq_list), torch.Tensor(label_list)


def rmse(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the RMSE
    """
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the MSE
    """
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the pearson correlation coefficient
    """
    rp = np.corrcoef(y, f)[0,1]
    return rp


def spearman(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the spearman correlation coefficient
    """
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y,f):
    """
    taken from https://github.com/thinng/GraphDTA

    computes the concordance index
    """
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Charge loading & Coulomb utilities
# ═══════════════════════════════════════════════════════════════════

def load_charges(charges_json_path: str) -> dict:
    """Load and return the parsed charges.json dict."""
    with open(charges_json_path, 'r') as f:
        return json.load(f)


def get_heavy_atom_charges(mol, charges_json: dict) -> list:
    """Return list of partial charges for heavy atoms only.

    The charges_json ligand charges are indexed over ALL atoms (incl. H),
    in RDKit MolFromMolFile(removeHs=False) order.  We iterate heavy atoms
    and use atom.GetIdx() to look up the correct charge.
    """
    ligand_charges = charges_json["ligand"]["charges"]
    heavy_charges = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            heavy_charges.append(ligand_charges[atom.GetIdx()])
    return heavy_charges


def load_protein_all_atoms(pdb_path: str) -> np.ndarray:
    """Read ALL ATOM records from a PDB file, return coordinates (N, 3).

    Reads every ATOM line (including H) in file order, matching the
    OpenMM topology order used in charges.json.  Only reads the first
    MODEL if multiple models are present.
    """
    coords = []
    in_first_model = True
    seen_model = False

    with open(pdb_path, 'r') as f:
        for line in f:
            record = line[:6].rstrip()
            if record == "MODEL":
                if seen_model:
                    # Second MODEL encountered — stop
                    break
                seen_model = True
                continue
            if record == "ENDMDL":
                if seen_model:
                    break
                continue
            if record == "ATOM":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    return np.array(coords, dtype=np.float64)


def get_protein_coords_and_charges(pdb_path: str, charges_json: dict):
    """Return protein atom coordinates (N,3) and charges (N,).

    Validates that the number of ATOM records matches charges_json.
    """
    coords = load_protein_all_atoms(pdb_path)
    charges = np.array(charges_json["protein"]["charges"], dtype=np.float64)
    if len(coords) != len(charges):
        raise ValueError(
            f"Protein atom count mismatch: PDB has {len(coords)} ATOM records "
            f"but charges.json has {len(charges)} charges"
        )
    return coords, charges


def compute_coulomb_potential(
    ligand_coords: np.ndarray,
    protein_coords: np.ndarray,
    protein_charges: np.ndarray,
    epsilon_r: float = EPSILON_R,
    cutoff: float = COULOMB_CUTOFF,
) -> np.ndarray:
    """Compute electrostatic potential at each ligand atom due to protein.

    Returns array of shape (N_lig,) in units of kcal/(mol·e).
    """
    prefactor = COULOMB_K / epsilon_r
    potentials = np.zeros(len(ligand_coords))

    # Pre-filter protein atoms to bounding box for efficiency
    prot_coords = protein_coords.copy()
    prot_charges = protein_charges.copy()
    for dim in range(3):
        lo = ligand_coords[:, dim].min() - cutoff
        hi = ligand_coords[:, dim].max() + cutoff
        mask = (prot_coords[:, dim] >= lo) & (prot_coords[:, dim] <= hi)
        prot_coords = prot_coords[mask]
        prot_charges = prot_charges[mask]

    for i, lig_pos in enumerate(ligand_coords):
        diffs = prot_coords - lig_pos
        dists = np.linalg.norm(diffs, axis=1)
        mask = (dists < cutoff) & (dists > DISTANCE_EPSILON)
        potentials[i] = prefactor * np.sum(prot_charges[mask] / dists[mask])

    return potentials


def compute_coulomb_energy(
    ligand_coords: np.ndarray,
    ligand_charges: np.ndarray,
    protein_coords: np.ndarray,
    protein_charges: np.ndarray,
    epsilon_r: float = EPSILON_R,
    cutoff: float = COULOMB_CUTOFF,
) -> float:
    """Compute total Coulomb interaction energy between ligand heavy atoms and protein.

    Returns scalar in kcal/mol.
    """
    prefactor = COULOMB_K / epsilon_r

    # Pre-filter protein atoms to bounding box
    prot_coords = protein_coords.copy()
    prot_charges = protein_charges.copy()
    for dim in range(3):
        lo = ligand_coords[:, dim].min() - cutoff
        hi = ligand_coords[:, dim].max() + cutoff
        mask = (prot_coords[:, dim] >= lo) & (prot_coords[:, dim] <= hi)
        prot_coords = prot_coords[mask]
        prot_charges = prot_charges[mask]

    energy = 0.0
    for i, (lig_pos, q_i) in enumerate(zip(ligand_coords, ligand_charges)):
        diffs = prot_coords - lig_pos
        dists = np.linalg.norm(diffs, axis=1)
        mask = (dists < cutoff) & (dists > DISTANCE_EPSILON)
        energy += prefactor * q_i * np.sum(prot_charges[mask] / dists[mask])

    return energy
