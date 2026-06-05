"""
helpers_qaev_plig_2.py — Coulomb utilities for AEV-PLIG-Coulomb-DDD.

Uses the Mehler–Solmajer distance-dependent dielectric (DDD) instead of a
constant permittivity.  Charge-loading helpers are re-imported directly from
helpers.py — no copy needed.
"""

import numpy as np

from config_qaev_plig_2 import COULOMB_K, DDD_A, DDD_B, DDD_LAMBDA, DDD_K, COULOMB_CUTOFF
from config import DISTANCE_EPSILON

# Re-export shared helpers so callers can import everything from one place.
from helpers import load_charges, get_heavy_atom_charges, get_protein_coords_and_charges  # noqa: F401


# ═══════════════════════════════════════════════════════════════════
# Distance-dependent dielectric
# ═══════════════════════════════════════════════════════════════════

def epsilon_ddd(r: np.ndarray) -> np.ndarray:
    """Mehler–Solmajer sigmoidal distance-dependent dielectric.

    epsilon(r) = A + B / (1 + k * exp(-lambda * B * r))

    Vectorised over r (Å).  Floor applied at 1.0 to prevent unphysical
    reversal of screening at very short distances where A < 0 dominates.
    """
    result = DDD_A + DDD_B / (1.0 + DDD_K * np.exp(-DDD_LAMBDA * DDD_B * r))
    return np.maximum(result, 1.0)


# ═══════════════════════════════════════════════════════════════════
# Coulomb potential and energy with DDD
# ═══════════════════════════════════════════════════════════════════

def compute_coulomb_potential_ddd(
    ligand_coords: np.ndarray,
    protein_coords: np.ndarray,
    protein_charges: np.ndarray,
    cutoff: float = COULOMB_CUTOFF,
) -> np.ndarray:
    """Electrostatic potential at each ligand heavy atom due to protein atoms.

    Uses per-pair distance-dependent dielectric:
        V_i = COULOMB_K * sum_{j, r_ij < cutoff} ( q_j / (epsilon(r_ij) * r_ij) )

    Returns array of shape (N_lig,) in kcal/(mol·e).
    """
    potentials = np.zeros(len(ligand_coords))

    # Bounding-box pre-filter for efficiency
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
        d = dists[mask]
        eps = epsilon_ddd(d)
        potentials[i] = COULOMB_K * np.sum(prot_charges[mask] / (eps * d))

    return potentials


def compute_coulomb_energy_ddd(
    ligand_coords: np.ndarray,
    ligand_charges: np.ndarray,
    protein_coords: np.ndarray,
    protein_charges: np.ndarray,
    cutoff: float = COULOMB_CUTOFF,
) -> float:
    """Total Coulomb interaction energy between ligand heavy atoms and protein.

    Uses per-pair distance-dependent dielectric:
        E = COULOMB_K * sum_i sum_{j, r_ij < cutoff} ( q_i * q_j / (epsilon(r_ij) * r_ij) )

    Returns scalar in kcal/mol.
    """
    # Bounding-box pre-filter
    prot_coords = protein_coords.copy()
    prot_charges = protein_charges.copy()
    for dim in range(3):
        lo = ligand_coords[:, dim].min() - cutoff
        hi = ligand_coords[:, dim].max() + cutoff
        mask = (prot_coords[:, dim] >= lo) & (prot_coords[:, dim] <= hi)
        prot_coords = prot_coords[mask]
        prot_charges = prot_charges[mask]

    energy = 0.0
    for lig_pos, q_i in zip(ligand_coords, ligand_charges):
        diffs = prot_coords - lig_pos
        dists = np.linalg.norm(diffs, axis=1)
        mask = (dists < cutoff) & (dists > DISTANCE_EPSILON)
        d = dists[mask]
        eps = epsilon_ddd(d)
        energy += COULOMB_K * q_i * np.sum(prot_charges[mask] / (eps * d))

    return float(energy)
