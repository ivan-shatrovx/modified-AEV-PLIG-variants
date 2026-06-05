"""
config_ddd.py — Configuration for AEV-PLIG-Coulomb-DDD.

Uses a distance-dependent dielectric (Mehler–Solmajer 1991 / AutoDock4)
instead of a constant permittivity, and multi-task output heads.
"""

# Coulomb constant: e²/Å → kcal/mol
COULOMB_K = 332.0637

# Mehler–Solmajer sigmoidal dielectric parameters (AutoDock4 / Protein Eng. 4:903, 1991)
EPSILON_0  = 78.4            # bulk water dielectric at 25 °C
DDD_A      = -8.5525
DDD_B      = EPSILON_0 - DDD_A   # = 86.9525
DDD_LAMBDA = 0.003627        # Å⁻¹
DDD_K      = 7.7839

# Distance cutoff for Coulomb sum — larger than AEV radial cutoff to capture
# longer-range electrostatics
COULOMB_CUTOFF = 12.0        # Å

# Multi-task loss weights (both on comparable scale after z-score normalisation)
LOSS_WEIGHT_PK      = 1.0
LOSS_WEIGHT_COULOMB = 1.0
