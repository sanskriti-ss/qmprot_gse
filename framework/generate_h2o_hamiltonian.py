#!/usr/bin/env python3
"""
Generate Hamiltonian for Water (H2O) molecule

This script generates a qubit Hamiltonian for the water molecule
using PySCF and OpenFermion with Jordan-Wigner transformation.

Run this to generate hamiltonian_h2o.txt for testing the VQE framework.
"""

def generate_h2o_hamiltonian():
    """Generate H2O Hamiltonian using PySCF and OpenFermion"""
    try:
        from openfermion import MolecularData, jordan_wigner
        from openfermionpyscf import run_pyscf
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "openfermion", "openfermionpyscf", "pyscf"])
        from openfermion import MolecularData, jordan_wigner
        from openfermionpyscf import run_pyscf
    
    # Water molecule geometry (experimental geometry)
    # O-H bond length: 0.9572 Angstrom
    # H-O-H angle: 104.52 degrees
    import numpy as np
    
    # Convert to Cartesian coordinates
    bond_length = 0.9572  # Angstrom
    angle = 104.52  # degrees
    angle_rad = np.radians(angle / 2)
    
    h1_x = bond_length * np.sin(angle_rad)
    h1_y = bond_length * np.cos(angle_rad)
    h2_x = -bond_length * np.sin(angle_rad)
    h2_y = bond_length * np.cos(angle_rad)
    
    # Geometry in Angstrom
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (h1_x, h1_y, 0.0)),
        ("H", (h2_x, h2_y, 0.0)),
    ]
    
    print("Water molecule geometry:")
    for atom, coords in geometry:
        print(f"  {atom}: ({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f})")
    
    # Create MolecularData object
    basis = "sto-3g"  # Minimal basis for testing
    multiplicity = 1   # Singlet
    charge = 0
    
    molecule = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        description="Water molecule"
    )
    
    print(f"\nRunning PySCF calculation with {basis} basis...")
    molecule = run_pyscf(molecule, run_scf=True, run_fci=False)
    
    print(f"  HF Energy: {molecule.hf_energy:.8f} Hartree")
    print(f"  Number of electrons: {molecule.n_electrons}")
    print(f"  Number of orbitals: {molecule.n_orbitals}")
    print(f"  Number of qubits: {2 * molecule.n_orbitals}")
    
    # Get fermionic Hamiltonian
    fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
    
    # Convert to qubit Hamiltonian using Jordan-Wigner transformation
    print("\nApplying Jordan-Wigner transformation...")
    qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    
    n_terms = len(qubit_hamiltonian.terms)
    print(f"  Number of Pauli terms: {n_terms}")
    
    # Determine number of qubits from the Hamiltonian
    n_qubits = 0
    for term in qubit_hamiltonian.terms:
        if term:  # Non-empty term
            max_qubit = max(qubit_idx for qubit_idx, _ in term)
            n_qubits = max(n_qubits, max_qubit + 1)
    
    print(f"  Number of qubits needed: {n_qubits}")
    
    # Save to file in framework format
    output_file = "data/hamiltonians/hamiltonian_h2o.txt"
    
    print(f"\nSaving to {output_file}...")
    
    with open(output_file, "w") as f:
        f.write("Coefficient\tOperators\n")
        
        for term, coeff in qubit_hamiltonian.terms.items():
            # Build Pauli string
            pauli_list = ['I'] * n_qubits
            for qubit_idx, pauli_op in term:
                pauli_list[qubit_idx] = pauli_op
            
            pauli_string = ''.join(pauli_list)
            
            # Write coefficient and operators
            # Use real part (imaginary should be ~0 for molecular Hamiltonians)
            if hasattr(coeff, 'real'):
                coeff_real = coeff.real
            else:
                coeff_real = float(coeff)
            
            if abs(coeff_real) > 1e-15:  # Skip very small terms
                f.write(f"{coeff_real:.15f}\t{pauli_string}\n")
    
    print(f"Done! Hamiltonian saved to {output_file}")
    
    # Return molecule info for JSON
    return {
        "abbreviation": "h2o",
        "name": "water",
        "mf": "H2O",
        "n_atoms": 3,
        "charge": charge,
        "n_electrons": molecule.n_electrons,
        "n_orbitals": molecule.n_orbitals,
        "n_qubits": n_qubits,
        "n_coefficients": n_terms,
        "hamiltonian": "hamiltonian_h2o.txt",
        "energy": molecule.hf_energy,
        "basis": basis,
    }


if __name__ == "__main__":
    import os
    
    # Change to framework directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create output directory if needed
    os.makedirs("data/hamiltonians", exist_ok=True)
    
    # Generate Hamiltonian
    mol_info = generate_h2o_hamiltonian()
    
    print("\n" + "="*50)
    print("Molecule info for qmprot.json:")
    print("="*50)
    import json
    print(json.dumps(mol_info, indent=2))
