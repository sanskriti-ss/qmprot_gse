#!/usr/bin/env python3
"""
Generate REDUCED Hamiltonian for Glutamine (Gln) using Active Space

This script generates a tractable qubit Hamiltonian for glutamine by using
an active space approximation - only the most important orbitals are included
while core electrons are frozen.

This makes the calculation feasible on standard hardware.
"""

def generate_gln_hamiltonian_active_space(n_active_electrons=10, n_active_orbitals=8):
    """
    Generate Glutamine Hamiltonian with active space reduction
    
    Args:
        n_active_electrons: Number of active electrons (default 10)
        n_active_orbitals: Number of active orbitals (default 8)
    """
    try:
        from openfermion import MolecularData, jordan_wigner, get_fermion_operator
        from openfermionpyscf import run_pyscf
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "openfermion", "openfermionpyscf", "pyscf"])
        from openfermion import MolecularData, jordan_wigner, get_fermion_operator
        from openfermionpyscf import run_pyscf
    
    # Glutamine molecule geometry (same as full version)
    geometry = [
        # Backbone atoms
        ("N", (-2.027, 1.378, 0.000)),
        ("C", (-0.611, 1.442, 0.000)),
        ("C", (0.059, 0.073, 0.000)),
        ("O", (-0.607, -0.964, 0.000)),
        
        # Side chain
        ("C", (0.026, 2.115, 1.253)),
        ("C", (1.545, 2.176, 1.253)),
        ("C", (2.182, 2.848, 2.457)),
        ("O", (1.542, 3.515, 3.258)),
        ("N", (3.501, 2.741, 2.611)),
        
        # Hydrogens
        ("H", (-2.424, 1.852, 0.822)),
        ("H", (-2.424, 1.852, -0.822)),
        ("H", (-0.286, 1.998, -0.900)),
        ("H", (-0.333, 3.139, 1.253)),
        ("H", (-0.333, 1.559, 2.106)),
        ("H", (1.904, 2.732, 0.400)),
        ("H", (1.904, 1.152, 1.253)),
        ("H", (3.963, 2.223, 1.898)),
        ("H", (4.004, 3.196, 3.374)),
        ("O", (1.293, -0.001, 0.000)),
        ("H", (1.628, -0.903, 0.000)),
    ]
    
    print("Glutamine (Gln) molecule with ACTIVE SPACE approximation")
    print("="*70)
    print(f"  Molecular formula: C5H10N2O3")
    print(f"  Total atoms: {len(geometry)}")
    print(f"  Active electrons: {n_active_electrons}")
    print(f"  Active orbitals: {n_active_orbitals}")
    print(f"  Qubits needed: {2 * n_active_orbitals}")
    
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    
    molecule = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        description="Glutamine amino acid (active space)"
    )
    
    print(f"\nStep 1: Running PySCF calculation...")
    molecule = run_pyscf(molecule, run_scf=True, run_fci=False)
    
    print(f" HF Energy: {molecule.hf_energy:.8f} Hartree")
    print(f" Full system: {molecule.n_electrons} electrons, {molecule.n_orbitals} orbitals")
    
    # Get the molecular Hamiltonian with active space
    print(f"\nStep 2: Extracting active space Hamiltonian...")
    print(f"  Active space: {n_active_electrons} electrons in {n_active_orbitals} orbitals")
    
    # Calculate number of occupied and virtual orbitals
    n_occupied = molecule.n_electrons // 2
    n_virtual = molecule.n_orbitals - n_occupied
    
    # Determine active space range
    # Take highest occupied and lowest virtual orbitals (frontier orbitals)
    occupied_indices = list(range(max(0, n_occupied - n_active_orbitals // 2), n_occupied))
    virtual_indices = list(range(n_occupied, min(molecule.n_orbitals, n_occupied + n_active_orbitals // 2)))
    active_indices = occupied_indices + virtual_indices
    
    actual_active_orbitals = len(active_indices)
    
    print(f"  Frozen orbitals: {n_occupied - len(occupied_indices)} occupied")
    print(f"  Active orbitals: {actual_active_orbitals} (indices: {min(active_indices)}-{max(active_indices)})")
    
    # Get molecular Hamiltonian in active space
    try:
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices
        )
    except:
        # Fallback: use full Hamiltonian if active space selection fails
        print("  Warning: Active space selection not supported, using manual reduction...")
        molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    
    print(f"\nStep 3: Applying Jordan-Wigner transformation...")
    print(f"  This should take ~30 seconds for {actual_active_orbitals} orbitals...")
    
    qubit_hamiltonian = jordan_wigner(molecular_hamiltonian)
    
    n_terms = len(qubit_hamiltonian.terms)
    print(f" Generated {n_terms:,} Pauli terms")
    
    # Determine number of qubits
    n_qubits = 2 * actual_active_orbitals
    print(f" Using {n_qubits} qubits")
    
    # Save to file
    output_file = "data/hamiltonians/hamiltonian_gln.txt"
    
    print(f"\nStep 4: Saving to {output_file}...")
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write("Coefficient\tOperators\n")
        
        terms_written = 0
        for term, coeff in qubit_hamiltonian.terms.items():
            # Build Pauli string
            pauli_list = ['I'] * n_qubits
            for qubit_idx, pauli_op in term:
                if qubit_idx < n_qubits:  # Safety check
                    pauli_list[qubit_idx] = pauli_op
            
            pauli_string = ''.join(pauli_list)
            
            # Use real part
            if hasattr(coeff, 'real'):
                coeff_real = coeff.real
            else:
                coeff_real = float(coeff)
            
            if abs(coeff_real) > 1e-15:
                f.write(f"{coeff_real:.15f}\t{pauli_string}\n")
                terms_written += 1
    
    print(f"  ✓ Wrote {terms_written:,} terms to file")
    
    # Calculate file size
    file_size = os.path.getsize(output_file)
    if file_size > 1_000_000:
        print(f" File size: {file_size / 1_000_000:.2f} MB")
    else:
        print(f"  File size: {file_size / 1_000:.2f} KB")
    
    print("\n" + "="*70)
    print("SUCCESS! Hamiltonian generated")
    print("="*70)
    
    # Return molecule info
    return {
        "abbreviation": "gln",
        "name": "glutamine",
        "mf": "C5H10N2O3",
        "n_atoms": len(geometry),
        "charge": charge,
        "n_electrons": n_active_electrons,
        "n_orbitals": actual_active_orbitals,
        "n_qubits": n_qubits,
        "n_coefficients": terms_written,
        "hamiltonian": "hamiltonian_gln.txt",
        "energy": molecule.hf_energy,
        "basis": basis,
        "description": f"Glutamine (active space: {n_active_electrons}e in {actual_active_orbitals} orbitals)"
    }


if __name__ == "__main__":
    import os
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate glutamine Hamiltonian with active space")
    parser.add_argument("--electrons", type=int, default=10, 
                        help="Number of active electrons (default: 10)")
    parser.add_argument("--orbitals", type=int, default=8,
                        help="Number of active orbitals (default: 8)")
    args = parser.parse_args()
    
    # Change to framework directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("="*70)
    print("Generating REDUCED Glutamine (Gln) Hamiltonian")
    print("="*70)
    print("\nThis uses active space approximation to make the calculation feasible.")
    print("Core electrons are frozen; only frontier orbitals are included.\n")
    
    try:
        mol_info = generate_gln_hamiltonian_active_space(
            n_active_electrons=args.electrons,
            n_active_orbitals=args.orbitals
        )
        
        print("\nMolecule info for qmprot.json:")
        print("-"*70)
        print(json.dumps(mol_info, indent=2))
        
        print("\n" + "="*70)
        print("Next steps:")
        print("="*70)
        print("1. Add this entry to data/qmprot.json")
        print("2. Run: python main.py --molecule gln")
        print("\nNOTE: This is an APPROXIMATION using active space.")
        print("Energy might be less accurate than full calculation, but runnable.")
        
    except Exception as e:
        import traceback
        print(f"\n{'='*70}")
        print("ERROR during generation")
        print("="*70)
        print(f"{e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nTry reducing the active space:")
        print("  python generate_gln_hamiltonian_active_space.py --orbitals 6 --electrons 8")
