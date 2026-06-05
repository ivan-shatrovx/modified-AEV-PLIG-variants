#!/usr/bin/env python3
"""
compute_benchmark_charges.py
============================
Compute AM1-BCC (ligand) + Amber ff14SB (protein) partial charges for the
optimised FEP benchmark dataset produced by ``optimise_benchmark.py``.

Designed for parallelisation on Oxford ARC via SLURM array jobs, following
the same manifest / chunk pattern as the HiQBind charge pipeline.

Modes of operation
------------------
1. ``generate-manifest``
       Scan the optimised benchmark directory, produce a manifest TSV (for
       SLURM batching) **and** the dataset CSV that ``process_and_predict.py``
       expects.

2. ``fix-proteins``
       Safety-net pass: fix any remaining non-standard residues in optimised
       protein PDBs using PDBFixer.  Run once before submitting batch jobs
       (avoids race conditions from parallel writes to the same PDB).

3. ``single``
       Process one protein-ligand complex.

4. ``batch``
       Process a chunk of the manifest (for SLURM arrays).

Workflow on ARC
---------------
::

    # 1. Generate manifest + dataset CSV
    python compute_benchmark_charges.py generate-manifest \\
        --optimised-dir fep_benchmark_optimised \\
        --charges-dir  fep_benchmark_charges \\
        --manifest     manifest_benchmark.tsv \\
        --dataset-csv  fep_benchmark_dataset.csv

    # 2. Fix any remaining non-standard residues (run once)
    python compute_benchmark_charges.py fix-proteins \\
        --optimised-dir fep_benchmark_optimised

    # 3. Submit SLURM array job
    sbatch submit_benchmark_charges.sh

Dependencies
------------
* RDKit, OpenMM, PDBFixer  (pip / conda)
* AmberTools antechamber + sqm on $PATH  (conda-forge)
"""

from __future__ import annotations

import argparse
import csv
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


# ═══════════════════════════════════════════════════════════════════════════
# Utility — locate antechamber
# ═══════════════════════════════════════════════════════════════════════════

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
        "Cannot locate `antechamber`.  Install AmberTools "
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
# 1.  Fix non-standard residues (safety-net for Script 2)
# ═══════════════════════════════════════════════════════════════════════════

def fix_nonstandard_residues(pdb_path: Path) -> bool:
    """Replace non-standard residues in *pdb_path* in-place using PDBFixer.

    Returns True if any residues were replaced, False otherwise.
    """
    import openmm.app as app
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findNonstandardResidues()

    if not fixer.nonstandardResidues:
        return False

    names = [f"{r.name}:{r.id}" for r, _ in fixer.nonstandardResidues]
    log.info("Fixing non-standard residues in %s: %s", pdb_path.name, ", ".join(names))

    fixer.replaceNonstandardResidues()

    # The replacement residues may need missing atoms / hydrogens
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # don't fill sequence gaps
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    with open(pdb_path, "w") as fh:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    log.info("  Saved fixed protein -> %s", pdb_path)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LIGAND — AM1-BCC via antechamber
# ═══════════════════════════════════════════════════════════════════════════

def compute_ligand_charges(sdf_path: str | Path) -> dict[str, Any]:
    """Assign AM1-BCC partial charges to a ligand from an SDF file.

    Atom ordering matches ``RDKit.Chem.MolFromMolFile(sdf, removeHs=False)``.
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
# 3.  PROTEIN — Amber ff14SB via OpenMM
# ═══════════════════════════════════════════════════════════════════════════

def compute_protein_charges(pdb_path: str | Path) -> dict[str, Any]:
    """Assign Amber ff14SB partial charges to every atom in a protein PDB.

    Atom ordering follows ``openmm.app.PDBFile(pdb).topology.atoms()``,
    which preserves the ATOM-record order of the input PDB.
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
# 4.  PROCESS A SINGLE COMPLEX
# ═══════════════════════════════════════════════════════════════════════════

def process_complex(
    protein_path: str | Path,
    ligand_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any] | None:
    """End-to-end charge computation for one protein-ligand complex.

    Returns the result dict, or None on failure.
    Skips if output already exists (idempotent for restarts).
    """
    protein_path = Path(protein_path).resolve()
    ligand_path = Path(ligand_path).resolve()
    output_path = Path(output_path).resolve()

    if output_path.exists():
        log.info("SKIP (already exists): %s", output_path)
        return {"status": "skipped"}

    result: dict[str, Any] = {
        "protein_file": str(protein_path),
        "ligand_file": str(ligand_path),
    }

    # ── Ligand charges ────────────────────────────────────────────────
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

    # ── Protein charges ───────────────────────────────────────────────
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

    # ── Save ──────────────────────────────────────────────────────────
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
    import traceback

    fail_path = output_path.with_suffix(".FAILED")
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fail_path, "w") as fh:
        fh.write(f"stage: {stage}\n")
        fh.write(f"protein: {protein_path}\n")
        fh.write(f"ligand:  {ligand_path}\n\n")
        traceback.print_exc(file=fh)
    log.error("Failure log written -> %s", fail_path)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  BATCH MODE — process a chunk of the manifest
# ═══════════════════════════════════════════════════════════════════════════

def run_batch(
    manifest_path: str | Path,
    chunk_id: int,
    chunk_size: int,
) -> None:
    """Process rows [chunk_id*chunk_size, (chunk_id+1)*chunk_size) from the
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
            "Chunk %d starts at row %d but manifest has only %d rows.  Nothing to do.",
            chunk_id, start, total,
        )
        return

    chunk = all_lines[start:end]
    log.info(
        "=== Batch chunk %d: rows %d–%d of %d (processing %d complexes) ===",
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
# 6.  GENERATE MANIFEST + DATASET CSV
# ═══════════════════════════════════════════════════════════════════════════

def generate_manifest(
    optimised_dir: str | Path,
    charges_dir: str | Path,
    manifest_path: str | Path,
    dataset_csv_path: str | Path,
) -> None:
    """Scan the optimised benchmark directory and produce:

    1. A manifest TSV (protein_pdb <TAB> ligand_sdf <TAB> charges_json)
       for SLURM batch processing.
    2. A dataset CSV (unique_id, sdf_file, pdb_file, charges_json)
       for ``process_and_predict.py``.
    """
    optimised = Path(optimised_dir).resolve()
    charges = Path(charges_dir).resolve()

    rows: list[dict[str, str]] = []

    for group_dir in sorted(optimised.iterdir()):
        if not group_dir.is_dir():
            continue

        # Discover targets by protein PDB files
        for pdb_file in sorted(group_dir.glob("*_protein.pdb")):
            target_prefix = pdb_file.stem.removesuffix("_protein")

            # Find all ligand SDFs for this target
            for sdf_file in sorted(group_dir.glob(f"{target_prefix}_*.sdf")):
                # Skip the protein PDB that also starts with target_prefix
                if sdf_file.stem == f"{target_prefix}_protein":
                    continue

                ligand_name = sdf_file.stem.removeprefix(f"{target_prefix}_")
                unique_id = f"{target_prefix}_{ligand_name}"
                charges_json = charges / group_dir.name / f"{unique_id}_charges.json"

                rows.append({
                    "unique_id": unique_id,
                    "pdb_file": str(pdb_file),
                    "sdf_file": str(sdf_file),
                    "charges_json": str(charges_json),
                })

    # ── Write manifest TSV ────────────────────────────────────────────
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        for row in rows:
            fh.write(f"{row['pdb_file']}\t{row['sdf_file']}\t{row['charges_json']}\n")

    # ── Write dataset CSV ─────────────────────────────────────────────
    dataset_csv_path = Path(dataset_csv_path)
    dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["unique_id", "sdf_file", "pdb_file", "charges_json"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Manifest written: %s  (%d complexes)", manifest_path, len(rows))
    log.info("Dataset CSV written: %s", dataset_csv_path)

    # Print summary for SLURM array sizing
    n_chunks_25 = (len(rows) + 24) // 25
    n_chunks_50 = (len(rows) + 49) // 50
    log.info(
        "SLURM array hints:  --array=0-%d  (chunk-size 25)  |  --array=0-%d  (chunk-size 50)",
        n_chunks_25 - 1, n_chunks_50 - 1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7.  FIX-PROTEINS MODE
# ═══════════════════════════════════════════════════════════════════════════

def fix_all_proteins(optimised_dir: str | Path) -> None:
    """Walk the optimised directory and fix non-standard residues in every
    protein PDB, in-place.  Safe to run multiple times (idempotent).
    """
    optimised = Path(optimised_dir).resolve()
    n_fixed, n_ok = 0, 0

    for group_dir in sorted(optimised.iterdir()):
        if not group_dir.is_dir():
            continue
        for pdb_file in sorted(group_dir.glob("*_protein.pdb")):
            try:
                if fix_nonstandard_residues(pdb_file):
                    n_fixed += 1
                else:
                    n_ok += 1
            except Exception:
                log.exception("FAILED to fix: %s", pdb_file)

    log.info("Done.  %d proteins fixed, %d already OK.", n_fixed, n_ok)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute AM1-BCC (ligand) + ff14SB (protein) charges for the "
            "optimised FEP benchmark dataset."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── generate-manifest ─────────────────────────────────────────────
    sp_gm = subparsers.add_parser(
        "generate-manifest",
        help="Scan optimised dir → manifest TSV + dataset CSV",
    )
    sp_gm.add_argument(
        "--optimised-dir", required=True,
        help="Path to fep_benchmark_optimised",
    )
    sp_gm.add_argument(
        "--charges-dir", required=True,
        help="Directory for charge JSON output (e.g. fep_benchmark_charges)",
    )
    sp_gm.add_argument(
        "--manifest", required=True,
        help="Output manifest TSV path",
    )
    sp_gm.add_argument(
        "--dataset-csv", required=True,
        help="Output dataset CSV path (for process_and_predict.py)",
    )

    # ── fix-proteins ──────────────────────────────────────────────────
    sp_fp = subparsers.add_parser(
        "fix-proteins",
        help="Fix non-standard residues in all optimised protein PDBs",
    )
    sp_fp.add_argument(
        "--optimised-dir", required=True,
        help="Path to fep_benchmark_optimised",
    )

    # ── single ────────────────────────────────────────────────────────
    sp_single = subparsers.add_parser(
        "single",
        help="Process a single complex",
    )
    sp_single.add_argument("protein_pdb", help="Path to protein PDB file")
    sp_single.add_argument("ligand_sdf", help="Path to ligand SDF file")
    sp_single.add_argument(
        "output_json", nargs="?", default=None,
        help="Output JSON path (default: <ligand_stem>_charges.json)",
    )

    # ── batch ─────────────────────────────────────────────────────────
    sp_batch = subparsers.add_parser(
        "batch",
        help="Process a chunk from a manifest (for SLURM arrays)",
    )
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

    # ── Dispatch ──────────────────────────────────────────────────────
    if args.mode == "generate-manifest":
        generate_manifest(
            args.optimised_dir,
            args.charges_dir,
            args.manifest,
            args.dataset_csv,
        )

    elif args.mode == "fix-proteins":
        fix_all_proteins(args.optimised_dir)

    elif args.mode == "single":
        out = args.output_json
        if out is None:
            out = Path(args.ligand_sdf).stem + "_charges.json"
        result = process_complex(args.protein_pdb, args.ligand_sdf, out)
        sys.exit(0 if result is not None else 1)

    elif args.mode == "batch":
        run_batch(args.manifest, args.chunk_id, args.chunk_size)


if __name__ == "__main__":
    main()
