"""End-to-end pipeline orchestrator for active space Hamiltonian generation.

Usage:
    python -m framework.active_space_truncation.run_pipeline --molecule gly
    python -m framework.active_space_truncation.run_pipeline --all
"""

import argparse
import logging
import time
from typing import Optional

from .geometry import MoleculeGeometry, get_geometry, load_geometry_from_h5, list_available
from .step1_orbitals import OrbitalDiagnostics, run_orbital_analysis
from .step2_active_space import ActiveSpaceResult, validate_active_space
from .step3_hamiltonian import PipelineHamiltonian, build_qubit_hamiltonian

logger = logging.getLogger(__name__)


def _log(msg: str, quiet: bool) -> None:
    """Print to stdout when verbose, always log."""
    logger.info(msg)
    if not quiet:
        print(msg)


def run_pipeline(
    molecule: str = "gly",
    basis: str = "cc-pvdz",
    h5_path: Optional[str] = None,
    run_casscf: bool = False,
    run_basis_comparison: bool = False,
    quiet: bool = False,
) -> dict:
    """Run the full active space pipeline.

    Args:
        quiet: If True, suppress detailed print output (used when called from main.py).

    Returns dict with geometry, diagnostics, active_space, hamiltonian.
    """
    t0 = time.time()

    _log("=" * 60, quiet)
    _log(f"ACTIVE SPACE PIPELINE: {molecule.upper()}", quiet)
    _log(f"Basis: {basis}", quiet)
    _log("=" * 60, quiet)

    # Load geometry
    _log("\n[1/4] Loading geometry...", quiet)
    if h5_path:
        geometry = load_geometry_from_h5(h5_path)
    else:
        geometry = get_geometry(molecule)

    _log(f"  Molecule: {geometry.name} ({geometry.formula})", quiet)
    _log(f"  Atoms: {geometry.n_atoms}, Electrons: {geometry.n_electrons}", quiet)

    # Step 1
    _log("\n[2/4] Running HF + MP2 orbital analysis...", quiet)
    t1 = time.time()
    diagnostics = run_orbital_analysis(geometry, basis=basis, run_basis_comparison=run_basis_comparison)
    _log(f"  Completed in {time.time() - t1:.1f}s", quiet)
    _log(diagnostics.summary(), quiet)

    # Step 2
    _log("\n[3/4] Validating active space with CASCI...", quiet)
    t2 = time.time()
    active_space = validate_active_space(geometry, diagnostics, run_casscf=run_casscf)
    _log(f"  Completed in {time.time() - t2:.1f}s", quiet)
    _log(active_space.summary(), quiet)

    # Save active orbital visualization
    try:
        from visualize_molecules.save_active_orbitals import save_active_orbital_summary
        img_path = save_active_orbital_summary(geometry, diagnostics)
        _log(f"  Active orbital image: {img_path}", quiet)
    except Exception as e:
        _log(f"  (Could not save orbital image: {e})", quiet)

    # Step 3
    _log("\n[4/4] Building qubit Hamiltonian...", quiet)
    t3 = time.time()
    hamiltonian = build_qubit_hamiltonian(geometry, active_space, diagnostics)
    _log(f"  Completed in {time.time() - t3:.1f}s", quiet)
    _log(hamiltonian.summary(), quiet)

    # Summary
    total_time = time.time() - t0
    _log("\n" + "=" * 60, quiet)
    _log("PIPELINE COMPLETE", quiet)
    _log("=" * 60, quiet)
    _log(f"Total time:            {total_time:.1f}s", quiet)
    _log(f"Molecule:              {geometry.name} ({geometry.formula})", quiet)
    _log(f"Basis:                 {basis}", quiet)
    _log(f"HF energy:             {diagnostics.hf_energy:.10f} Ha", quiet)
    _log(f"MP2 energy:            {diagnostics.mp2_energy:.10f} Ha", quiet)
    _log(f"CASCI energy:          {active_space.casci_energy:.10f} Ha", quiet)
    _log(f"Active space:          ({active_space.n_active_electrons}e, {active_space.n_active_orbitals}o)", quiet)
    _log(f"Qubits:                {hamiltonian.n_qubits}", quiet)
    _log(f"Hamiltonian terms:     {hamiltonian.n_terms}", quiet)
    _log(f"Core energy:           {hamiltonian.core_energy:.10f} Ha", quiet)
    _log(f"CASCI below HF:        {active_space.energy_below_hf}", quiet)
    _log(f"Correlation recovered: {active_space.correlation_recovered:.1%}", quiet)
    _log("=" * 60, quiet)

    return {
        "geometry": geometry,
        "diagnostics": diagnostics,
        "active_space": active_space,
        "hamiltonian": hamiltonian,
    }


def main():
    parser = argparse.ArgumentParser(description="Active Space Pipeline")
    parser.add_argument("--molecule", "-m", default="gly", help="Molecule abbreviation")
    parser.add_argument("--basis", "-b", default="cc-pvdz", help="Basis set")
    parser.add_argument("--h5-path", default=None, help="Path to H5 geometry file")
    parser.add_argument("--casscf", action="store_true", help="Also run CASSCF")
    parser.add_argument("--basis-comparison", action="store_true", help="Run STO-3G comparison")
    parser.add_argument("--all", action="store_true", help="Run all available molecules")
    args = parser.parse_args()

    if args.all:
        molecules = list_available()
        print(f"Running pipeline for {len(molecules)} molecules: {molecules}\n")
        all_results = {}
        for mol in molecules:
            try:
                result = run_pipeline(
                    molecule=mol, basis=args.basis,
                    run_casscf=args.casscf,
                    run_basis_comparison=args.basis_comparison,
                )
                all_results[mol] = result
            except Exception as e:
                print(f"\nERROR running {mol}: {e}\n")
        print(f"\nCompleted {len(all_results)}/{len(molecules)} molecules.")
    else:
        run_pipeline(
            molecule=args.molecule, basis=args.basis,
            h5_path=args.h5_path, run_casscf=args.casscf,
            run_basis_comparison=args.basis_comparison,
        )


if __name__ == "__main__":
    main()
