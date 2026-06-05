#!/usr/bin/env python3
"""
compute_charges.py
==================
Compute partial charges for protein–ligand complexes, suitable for use as
GNN node features in binding-affinity prediction.

Ligand : AM1-BCC partial charges via AmberTools antechamber
Protein: Amber ff14SB partial charges via OpenMM

Modes of operation
------------------
1. Single complex:
       python compute_charges.py single <protein.pdb> <ligand.sdf> <output.json>

2. Batch from manifest (for SLURM array jobs):
       python compute_charges.py batch <manifest.tsv> --chunk-id 0 --chunk-size 25

   Processes rows [chunk_id * chunk_size, (chunk_id+1) * chunk_size) from the
   manifest.  The SLURM wrapper sets --chunk-id=$SLURM_ARRAY_TASK_ID.

Atom ordering guarantees
------------------------
* Ligand charges are indexed identically to
  ``RDKit.Chem.MolFromMolFile(sdf, removeHs=False)``
* Protein charges follow the ATOM-record order of the input PDB, i.e.
  ``openmm.app.PDBFile(pdb).topology.atoms()``
* An assertion verifies len(charges) == num_atoms for both.

Dependencies
------------
* RDKit, OpenMM, PDBFixer  (pip)
* AmberTools antechamber+sqm on $PATH  (conda-forge)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility: locate antechamber binary
# ---------------------------------------------------------------------------

def _find_antechamber() -> str:
    """Return path to antechamber, searching PATH and common conda locations."""
    ac = shutil.which("antechamber")
    if ac:
        return ac
    for root in [
        Path.home() / "micromamba" / "envs",
        Path.home() / "mm" / "envs",
        Path.home() / "conda" / "envs",
        Path.home() / "miniforge3" / "envs",
        Path.home() / "mambaforge" / "envs",
        Path("/opt/conda/envs"),
    ]:
        if root.is_dir():
            for candidate in root.rglob("antechamber"):
                if os.access(candidate, os.X_OK):
                    return str(candidate)
    raise FileNotFoundError(
        "Cannot locate `antechamber`. Install AmberTools "
        "(conda install -c conda-forge ambertools) and ensure it is on $PATH."
    )


def _antechamber_env() -> dict[str, str]:
    """Return an env dict that puts antechamber's directory on PATH."""
    ac = _find_antechamber()
    env = os.environ.copy()
    ac_dir = str(Path(ac).resolve().parent)
    env["PATH"] = ac_dir + ":" + env.get("PATH", "")
    lib_dir = str(Path(ac_dir).parent / "lib")
    env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


# ═══════════════════════════════════════════════════════════════════════════
# 1.  LIGAND — AM1-BCC via antechamber
# ═══════════════════════════════════════════════════════════════════════════

def compute_ligand_charges(sdf_path: str | Path) -> dict[str, Any]:
    """
    Assign AM1-BCC partial charges to a ligand from an SDF file.

    Returns
    -------
    dict with keys:
        charges       : list[float]   — one charge per atom, SDF atom order
        atom_elements : list[str]     — element symbols in the same order
        atom_indices  : list[int]     — 0-based indices (identity map)
        num_atoms     : int
        net_charge    : int
        note          : str           — provenance / matching note

    Atom ordering
    -------------
    Indexed identically to:
      RDKit Chem.MolFromMolFile(sdf, removeHs=False).GetAtomWithIdx(i)
    """
    from rdkit import Chem

    sdf_path = Path(sdf_path).resolve()
    if not sdf_path.exists():
        raise FileNotFoundError(f"Ligand SDF not found: {sdf_path}")

    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse SDF: {sdf_path}")

    n_atoms_sdf = mol.GetNumAtoms()
    net_charge = Chem.GetFormalCharge(mol)
    elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms_sdf)]

    log.info(
        "Ligand: %d atoms, net charge %+d, file=%s",
        n_atoms_sdf, net_charge, sdf_path.name,
    )

    env = _antechamber_env()
    with tempfile.TemporaryDirectory(prefix="am1bcc_") as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "input.sdf"
        shutil.copy2(sdf_path, src)

        mol2_out = tmp / "output.mol2"
        cmd = [
            "antechamber",
            "-i",  str(src),
            "-fi", "sdf",
            "-o",  str(mol2_out),
            "-fo", "mol2",
            "-c",  "bcc",
            "-at", "gaff2",
            "-nc", str(net_charge),
            "-pf", "y",
        ]
        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, cwd=tmpdir, env=env,
            capture_output=True, text=True, timeout=1200,
        )
        if result.returncode != 0 or not mol2_out.exists():
            raise RuntimeError(
                f"antechamber failed (rc={result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        charges = _parse_mol2_charges(mol2_out)

    if len(charges) != n_atoms_sdf:
        raise RuntimeError(
            f"Atom-count mismatch: SDF has {n_atoms_sdf} atoms but "
            f"antechamber returned {len(charges)} charges."
        )

    return {
        "charges": charges,
        "atom_elements": elements,
        "atom_indices": list(range(n_atoms_sdf)),
        "num_atoms": n_atoms_sdf,
        "net_charge": net_charge,
        "note": (
            "AM1-BCC charges via antechamber/sqm. "
            "Atom order matches RDKit Chem.MolFromMolFile(sdf, removeHs=False)."
        ),
    }


def _parse_mol2_charges(mol2_path: Path) -> list[float]:
    """Extract per-atom charges from a Tripos mol2 file, preserving order."""
    charges: list[float] = []
    in_atom_block = False
    with open(mol2_path) as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom_block:
                break
            if in_atom_block and line.strip():
                parts = line.split()
                charges.append(float(parts[8]))
    return charges


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PROTEIN — Amber ff14SB via OpenMM
# ═══════════════════════════════════════════════════════════════════════════

def compute_protein_charges(pdb_path: str | Path) -> dict[str, Any]:
    """
    Assign Amber ff14SB partial charges to every atom in a protein PDB.

    Returns
    -------
    dict with keys:
        charges       : list[float]
        atom_labels   : list[ [chain_id, residue_name, residue_id, atom_name] ]
        atom_indices  : list[int]
        num_atoms     : int
        note          : str

    Atom ordering
    -------------
    Follows the ATOM record order of the input PDB, identical to
    openmm.app.PDBFile(pdb).topology.atoms().
    """
    import openmm
    import openmm.app as app

    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(f"Protein PDB not found: {pdb_path}")

    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    n_atoms_pdb = topology.getNumAtoms()
    log.info(
        "Protein: %d atoms, %d residues, %d chain(s), file=%s",
        n_atoms_pdb,
        topology.getNumResidues(),
        topology.getNumChains(),
        pdb_path.name,
    )

    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        names = [f"{r.name}:{r.id}" for r, _ in fixer.nonstandardResidues]
        raise ValueError(
            f"Non-standard residues found — ff14SB cannot parameterise them. "
            f"Residues: {', '.join(names)}. "
            f"Remove or replace them before running this script."
        )

    ff = app.ForceField("amber14/protein.ff14SB.xml")
    try:
        system = ff.createSystem(topology)
    except Exception as exc:
        raise RuntimeError(
            f"OpenMM could not parameterise the protein with ff14SB: {exc}"
        ) from exc

    nb_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb_force = force
            break
    if nb_force is None:
        raise RuntimeError("No NonbondedForce found in the OpenMM System.")

    n_particles = nb_force.getNumParticles()
    if n_particles != n_atoms_pdb:
        raise RuntimeError(
            f"Atom-count mismatch: PDB has {n_atoms_pdb} atoms but "
            f"NonbondedForce has {n_particles} particles."
        )

    charges: list[float] = []
    for i in range(n_particles):
        charge, _sigma, _epsilon = nb_force.getParticleParameters(i)
        charges.append(charge.value_in_unit(openmm.unit.elementary_charge))

    atom_labels: list[list[str]] = []
    for atom in topology.atoms():
        res = atom.residue
        atom_labels.append([
            res.chain.id,
            res.name,
            str(res.id),
            atom.name,
        ])

    assert len(charges) == n_atoms_pdb, (
        f"Final sanity check failed: {len(charges)} charges != {n_atoms_pdb} PDB atoms"
    )

    return {
        "charges": charges,
        "atom_labels": atom_labels,
        "atom_indices": list(range(n_atoms_pdb)),
        "num_atoms": n_atoms_pdb,
        "note": (
            "Amber ff14SB charges via OpenMM. "
            "Atom order matches openmm.app.PDBFile(pdb).topology.atoms() "
            "which preserves the ATOM-record order of the input PDB."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PROCESS A SINGLE COMPLEX
# ═══════════════════════════════════════════════════════════════════════════

def process_complex(
    protein_path: str | Path,
    ligand_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any] | None:
    """
    End-to-end charge computation for one protein–ligand complex.

    Returns the result dict, or None on failure (writes a .FAILED log).
    Skips if output already exists (idempotent for restarts).
    """
    protein_path = Path(protein_path).resolve()
    ligand_path = Path(ligand_path).resolve()
    output_path = Path(output_path).resolve()

    # Skip if already completed
    if output_path.exists():
        log.info("SKIP (already exists): %s", output_path)
        return {"status": "skipped"}

    result: dict[str, Any] = {
        "protein_file": str(protein_path),
        "ligand_file": str(ligand_path),
    }

    # ---- Ligand charges ---------------------------------------------------
    try:
        log.info("=== Computing ligand AM1-BCC charges ===")
        lig = compute_ligand_charges(ligand_path)
        result["ligand"] = lig
        log.info(
            "Ligand charges OK — %d atoms, sum=%.4f",
            lig["num_atoms"], sum(lig["charges"]),
        )
    except Exception:
        log.exception("FAILED: ligand charge computation")
        _write_failure_log(output_path, "ligand", protein_path, ligand_path)
        return None

    # ---- Protein charges --------------------------------------------------
    try:
        log.info("=== Computing protein ff14SB charges ===")
        prot = compute_protein_charges(protein_path)
        result["protein"] = prot
        log.info(
            "Protein charges OK — %d atoms, sum=%.4f",
            prot["num_atoms"], sum(prot["charges"]),
        )
    except Exception:
        log.exception("FAILED: protein charge computation")
        _write_failure_log(output_path, "protein", protein_path, ligand_path)
        return None

    # ---- Save -------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("Saved charges -> %s", output_path)

    return result


def _write_failure_log(
    output_path: Path, stage: str,
    protein_path: Path, ligand_path: Path,
) -> None:
    """Write a concise failure marker for triage."""
    fail_path = output_path.with_suffix(".FAILED")
    import traceback
    with open(fail_path, "w") as fh:
        fh.write(f"stage: {stage}\n")
        fh.write(f"protein: {protein_path}\n")
        fh.write(f"ligand:  {ligand_path}\n\n")
        traceback.print_exc(file=fh)
    log.error("Failure log written -> %s", fail_path)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  BATCH MODE — process a chunk of the manifest
# ═══════════════════════════════════════════════════════════════════════════

def run_batch(
    manifest_path: str | Path,
    chunk_id: int,
    chunk_size: int,
) -> None:
    """
    Process rows [chunk_id*chunk_size, (chunk_id+1)*chunk_size) from the
    manifest TSV.  Each row is: protein_pdb <TAB> ligand_sdf <TAB> output_json
    """
    manifest_path = Path(manifest_path).resolve()
    with open(manifest_path) as fh:
        all_lines = [line.strip() for line in fh if line.strip()]

    total = len(all_lines)
    start = chunk_id * chunk_size
    end = min(start + chunk_size, total)

    if start >= total:
        log.info(
            "Chunk %d starts at row %d but manifest has only %d rows. Nothing to do.",
            chunk_id, start, total,
        )
        return

    chunk = all_lines[start:end]
    log.info(
        "=== Batch chunk %d: rows %d-%d of %d (processing %d complexes) ===",
        chunk_id, start, end - 1, total, len(chunk),
    )

    n_ok, n_skip, n_fail = 0, 0, 0
    for i, line in enumerate(chunk):
        parts = line.split("\t")
        if len(parts) != 3:
            log.error("Malformed manifest line %d: %s", start + i, line)
            n_fail += 1
            continue

        prot, lig, out = parts
        log.info(
            "--- Complex %d/%d (global row %d) ---",
            i + 1, len(chunk), start + i,
        )
        result = process_complex(prot, lig, out)
        if result is None:
            n_fail += 1
        elif result.get("status") == "skipped":
            n_skip += 1
        else:
            n_ok += 1

    log.info(
        "=== Chunk %d complete: %d OK, %d skipped, %d failed ===",
        chunk_id, n_ok, n_skip, n_fail,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AM1-BCC (ligand) + ff14SB (protein) charges for GNN node features.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -- single mode --
    sp_single = subparsers.add_parser("single", help="Process a single complex")
    sp_single.add_argument("protein_pdb", help="Path to protein PDB file")
    sp_single.add_argument("ligand_sdf", help="Path to ligand SDF file")
    sp_single.add_argument(
        "output_json", nargs="?", default=None,
        help="Output JSON path (default: <ligand_stem>_charges.json)",
    )

    # -- batch mode --
    sp_batch = subparsers.add_parser("batch", help="Process a chunk from a manifest (for SLURM arrays)")
    sp_batch.add_argument("manifest", help="Path to manifest TSV")
    sp_batch.add_argument(
        "--chunk-id", type=int, required=True,
        help="0-based chunk index (typically $SLURM_ARRAY_TASK_ID)",
    )
    sp_batch.add_argument(
        "--chunk-size", type=int, default=25,
        help="Number of complexes per chunk (default: 25)",
    )

    args = parser.parse_args()

    if args.mode == "single":
        out = args.output_json
        if out is None:
            out = Path(args.ligand_sdf).stem + "_charges.json"
        result = process_complex(args.protein_pdb, args.ligand_sdf, out)
        sys.exit(0 if result is not None else 1)

    elif args.mode == "batch":
        run_batch(args.manifest, args.chunk_id, args.chunk_size)


if __name__ == "__main__":
    main()
