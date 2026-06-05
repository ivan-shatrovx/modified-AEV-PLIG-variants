"""
config.py — Centralised configuration for qAEV-PLIG-1.

All tuneable physics parameters and paths live here so they are
easy to find and change in one place.
"""

import torch

# ═══════════════════════════════════════════════════════════════════
# Electrostatics
# ═══════════════════════════════════════════════════════════════════

# Coulomb constant: converts e²/Å to kcal/mol
# k_e = 1 / (4π ε_0) in units where charges are in elementary charge
# and distances in Ångströms.
COULOMB_K = 332.0637  # kcal·Å / (mol·e²)

# Relative permittivity (dielectric constant).
# Common choices:
#   1.0   — vacuum (no screening)
#   4.0   — protein interior / protein–ligand interface (common default)
#   20.0  — partially solvated interface
#   78.5  — bulk water at 25°C
# This is the main parameter to experiment with.
EPSILON_R = 4.0

# Distance cutoff for Coulomb interactions (Å).
# Protein atoms beyond this distance from a ligand atom are ignored.
# Default matches the AEV radial cutoff.
COULOMB_CUTOFF = 5.1

# Small epsilon to prevent division by zero in 1/r calculations.
DISTANCE_EPSILON = 1e-8

# ═══════════════════════════════════════════════════════════════════
# AEV parameters (unchanged from original AEV-PLIG, ANI-2x settings)
# ═══════════════════════════════════════════════════════════════════

AEV_RCR = 5.1  # Radial cutoff (Å)
AEV_ETAR = torch.tensor([19.7])  # Radial decay
AEV_RSR = torch.tensor([
    0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68,
    2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83,
])  # Radial shifts

# Angular AEV coefficients (used internally by torchani_mod)
AEV_RCA = 2.0
AEV_ZETA = torch.tensor([1.0])
AEV_TSA = torch.tensor([1.0])
AEV_ETAA = torch.tensor([1.0])
AEV_RSA = torch.tensor([1.0])

# Number of atom types for AEV (elements 1-22)
AEV_NUM_SPECIES = 22

# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

DATASET_CSV = "data/dataset.csv"
ATOM_KEYS_CSV = "data/PDB_Atom_Keys.csv"

# ═══════════════════════════════════════════════════════════════════
# Element set for one-hot encoding of ligand atoms
# (must match the original AEV-PLIG set exactly)
# ═══════════════════════════════════════════════════════════════════

ELEMENT_LIST = ['F', 'N', 'Cl', 'O', 'Br', 'C', 'B', 'P', 'I', 'S']
