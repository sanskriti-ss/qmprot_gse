# Hamiltonian Files Directory

Place your Hamiltonian files here in the following format:

## File Naming Convention

```
hamiltonian_<molecule_abbreviation>.txt
```

Examples:
- `hamiltonian_trp.txt` (tryptophan)
- `hamiltonian_his.txt` (histidine)
- `hamiltonian_gly.txt` (glycine)

## File Format

Each Hamiltonian file should be a tab-separated file with the following format:

```
Coefficient	Operators
0.123456	IIII
-0.234567	ZIIZ
0.345678	XXYY
...
```

### Column Descriptions

1. **Coefficient**: The numerical coefficient (real number) for the Pauli term
2. **Operators**: The Pauli string (e.g., `IIII`, `XYZZ`, `ZIZI`)

### Pauli String Convention

- `I` = Identity
- `X` = Pauli-X
- `Y` = Pauli-Y  
- `Z` = Pauli-Z

The length of the Pauli string determines the number of qubits.

## Generating Hamiltonians

You can generate Hamiltonians using the code in `QMProt/hamiltonian.ipynb`:

```python
from openfermion import MolecularData, jordan_wigner
from openfermionpyscf import run_pyscf

# Define your molecule
geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]
molecule = MolecularData(geometry, "sto-3g", 1, 0)
molecule = run_pyscf(molecule)

# Get Hamiltonian
fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)

# Save to file
with open("hamiltonian_h2.txt", "w") as f:
    f.write("Coefficient\tOperators\n")
    for term, coeff in qubit_hamiltonian.terms.items():
        pauli_str = "".join([op[1] if op else "I" for op in term])
        f.write(f"{coeff.real}\t{pauli_str}\n")
```

## QMProt Dataset

The QMProt dataset provides Hamiltonians for 20 amino acids and related molecules.
See the accompanying `qmprot.json` for molecule metadata.
