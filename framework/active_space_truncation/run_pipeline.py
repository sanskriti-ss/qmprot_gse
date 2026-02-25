"""End-to-end pipeline orchestrator for active space Hamiltonian generation.

Usage:
    python -m framework.active_space_truncation.run_pipeline --molecule gly
    python -m framework.active_space_truncation.run_pipeline --all
"""

import argparse
import time
from typing import Optional

from .geometry import MoleculeGeometry, get_geometry, load_geometry_from_h5, list_available
from .step1_orbitals import OrbitalDiagnostics, run_orbital_analysis
from .step2_active_space import ActiveSpaceResult, validate_active_space
from .step3_hamiltonian import PipelineHamiltonian, build_qubit_hamiltonian


def run_pipeline(
    molecule: str = "gly",
    basis: str = "cc-pvdz",
    h5_path: Optional[str] = None,
    run_casscf: bool = False,
    run_basis_comparison: bool = False,
) -> dict:
    """Run the full active space pipeline.

    Returns dict with geometry, diagnostics, active_space, hamiltonian.
    """
    t0 = time.time()

    print("=" * 60)
    print(f"ACTIVE SPACE PIPELINE: {molecule.upper()}")
    print(f"Basis: {basis}")
    print("=" * 60)

    # Load geometry
    print("\n[1/4] Loading geometry...")
    if h5_path:
        geometry = load_geometry_from_h5(h5_path)
    else:
        geometry = get_geometry(molecule)

    print(f"  Molecule: {geometry.name} ({geometry.formula})")
    print(f"  Atoms: {geometry.n_atoms}, Electrons: {geometry.n_electrons}")

    # Step 1
    print("\n[2/4] Running HF + MP2 orbital analysis...")
    t1 = time.time()
    diagnostics = run_orbital_analysis(geometry, basis=basis, run_basis_comparison=run_basis_comparison)
    print(f"  Completed in {time.time() - t1:.1f}s")
    print(diagnostics.summary())

    # Step 2
    print("\n[3/4] Validating active space with CASCI...")
    t2 = time.time()
    active_space = validate_active_space(geometry, diagnostics, run_casscf=run_casscf)
    print(f"  Completed in {time.time() - t2:.1f}s")
    print(active_space.summary())

    # Step 3
    print("\n[4/4] Building qubit Hamiltonian...")
    t3 = time.time()
    hamiltonian = build_qubit_hamiltonian(geometry, active_space, diagnostics)
    print(f"  Completed in {time.time() - t3:.1f}s")
    print(hamiltonian.summary())

    # Summary
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time:            {total_time:.1f}s")
    print(f"Molecule:              {geometry.name} ({geometry.formula})")
    print(f"Basis:                 {basis}")
    print(f"HF energy:             {diagnostics.hf_energy:.10f} Ha")
    print(f"MP2 energy:            {diagnostics.mp2_energy:.10f} Ha")
    print(f"CASCI energy:          {active_space.casci_energy:.10f} Ha")
    print(f"Active space:          ({active_space.n_active_electrons}e, {active_space.n_active_orbitals}o)")
    print(f"Qubits:                {hamiltonian.n_qubits}")
    print(f"Hamiltonian terms:     {hamiltonian.n_terms}")
    print(f"CASCI below HF:        {active_space.energy_below_hf}")
    print(f"Correlation recovered: {active_space.correlation_recovered:.1%}")
    print("=" * 60)

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
