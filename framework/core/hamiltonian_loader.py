"""
Hamiltonian Loader Module

Loads Hamiltonians from:
1. H5 files (molecule.h5) from QMProt datasets - DEFAULT
2. Text files (hamiltonian_xxx.txt) with format: Coefficient\tOperators - LEGACY
3. JSON metadata files (qmprot.json) for molecule properties
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
import h5py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HAMILTONIAN_MAX_TERMS, HAMILTONIAN_TARGET_QUBITS

logger = logging.getLogger(__name__)


@dataclass
class Molecule:
    """Data class for molecule information"""
    abbreviation: str
    name: str
    n_qubits: int
    n_coefficients: int
    reference_energy: float
    hamiltonian_file: str
    n_electrons: Optional[int] = None
    n_orbitals: Optional[int] = None
    charge: int = 0
    spin: int = 0
    basis: str = "sto-3g"
    coordinates: Optional[List[Dict]] = None
    molecular_formula: Optional[str] = None
    truncated_ground_state_energy: Optional[float] = None  # Ground state of truncated system
    core_energy: Optional[float] = None  # Frozen core + nuclear repulsion (active space mode)
    
    
@dataclass
class QubitHamiltonian:
    """Data class for qubit Hamiltonian"""
    molecule: Molecule
    coefficients: np.ndarray
    pauli_strings: List[str]
    n_qubits: int
    n_terms: int
    
    def truncate(self, max_terms: int = 1000, target_qubits: int = 8) -> 'QubitHamiltonian':
        """
        Truncate the Hamiltonian by keeping only the most important terms.
        
        Args:
            max_terms: Maximum number of terms to keep
            target_qubits: Target number of qubits for the reduced system
            
        Returns:
            Truncated QubitHamiltonian object
        """
        if len(self.coefficients) <= max_terms:
            return self  # Already small enough
        
        logger.info(f"Truncating Hamiltonian from {len(self.coefficients)} to max {max_terms} terms...")
        
        # Step 1: Sort by coefficient magnitude
        coeff_idx_pairs = [(abs(c), i, c) for i, c in enumerate(self.coefficients)]
        coeff_idx_pairs.sort(reverse=True)
        
        # Step 2: Keep terms with largest coefficients that fit within qubit limit
        kept_indices = []
        used_wires = set()
        
        for abs_coeff, idx, coeff in coeff_idx_pairs:
            if len(kept_indices) >= max_terms:
                break
            
            # Get wires used by this term
            pauli_str = self.pauli_strings[idx]
            term_wires = {i for i, p in enumerate(pauli_str) if p != 'I'}
            
            # Check if adding this term would exceed qubit limit
            new_wires = used_wires.union(term_wires)
            if len(new_wires) <= target_qubits:
                kept_indices.append(idx)
                used_wires = new_wires
        
        # Step 3: Create truncated hamiltonian
        kept_indices.sort()  # Keep original order
        truncated_coeffs = self.coefficients[kept_indices]
        truncated_pauli = [self.pauli_strings[i] for i in kept_indices]
        
        # Remap wires to be consecutive from 0
        if used_wires:
            wire_mapping = {old: new for new, old in enumerate(sorted(used_wires))}
            actual_n_qubits = len(wire_mapping)
            remapped_pauli = []
            surviving_coeffs = []
            n_dropped = 0
            for pauli_str, coeff in zip(truncated_pauli, truncated_coeffs):
                remapped = ['I'] * actual_n_qubits
                remap_ok = True
                for wire, pauli in enumerate(pauli_str):
                    if pauli != 'I':
                        if wire in wire_mapping:
                            remapped[wire_mapping[wire]] = pauli
                        else:
                            remap_ok = False
                            break
                if remap_ok:
                    remapped_pauli.append(''.join(remapped))
                    surviving_coeffs.append(coeff)
                else:
                    n_dropped += 1
            if n_dropped > 0:
                logger.info(f"Dropped {n_dropped} terms with unmapped wires")
            truncated_pauli = remapped_pauli
            truncated_coeffs = np.array(surviving_coeffs)
        else:
            actual_n_qubits = self.n_qubits
        
        logger.info(f"Truncated to {len(truncated_coeffs)} terms on {actual_n_qubits} qubits")
        
        # Step 4: Calculate ground state energy of truncated Hamiltonian
        truncated_ground_state_energy = self._calculate_ground_state_energy(
            truncated_coeffs, truncated_pauli, actual_n_qubits
        )
        
        # Create a new molecule entry with truncated ground state energy
        truncated_molecule = Molecule(
            abbreviation=self.molecule.abbreviation,
            name=self.molecule.name,
            n_qubits=actual_n_qubits,
            n_coefficients=len(truncated_coeffs),
            reference_energy=self.molecule.reference_energy,  # Keep original full system reference
            hamiltonian_file=self.molecule.hamiltonian_file,
            n_electrons=self.molecule.n_electrons,
            n_orbitals=self.molecule.n_orbitals,
            charge=self.molecule.charge,
            spin=self.molecule.spin,
            basis=self.molecule.basis,
            coordinates=self.molecule.coordinates,
            molecular_formula=self.molecule.molecular_formula,
            truncated_ground_state_energy=truncated_ground_state_energy,  # Ground state of truncated system
        )
        
        return QubitHamiltonian(
            molecule=truncated_molecule,
            coefficients=truncated_coeffs,
            pauli_strings=truncated_pauli,
            n_qubits=actual_n_qubits,
            n_terms=len(truncated_coeffs),
        )
    
    def _openfermion_to_pauli_string(self, pauli_term_str: str, n_qubits: int) -> str:
        """
        Convert OpenFermion format 'Z(0) X(2) Y(3)' to simple Pauli string 'ZIXIY'.
        """
        pauli_map = {}
        
        # Handle 'Identity(0)' or just 'Identity'
        if 'Identity' in pauli_term_str:
            return 'I' * n_qubits
        
        # Parse terms like 'Z(0)', 'X(1)', etc
        import re
        pattern = r'([IXYZ])\((\d+)\)'
        matches = re.findall(pattern, pauli_term_str)
        
        for pauli, qubit_str in matches:
            qubit = int(qubit_str)
            if qubit < n_qubits:
                pauli_map[qubit] = pauli
        
        # Build full string
        result = []
        for i in range(n_qubits):
            result.append(pauli_map.get(i, 'I'))
        
        return ''.join(result)
    
    def _calculate_ground_state_energy(self, coefficients, pauli_strings, n_qubits):
        """
        Estimate ground state energy of the Hamiltonian.
        
        For small systems (n_qubits <= 10), uses exact diagonalization.
        For larger systems, uses a lower bound estimate.
        """
        try:
            if n_qubits > 18:
                # For large systems, use a lower bound: sum of negative terms
                logger.info(f"System too large for exact diagonalization ({n_qubits} qubits). Using lower bound.")
                negative_sum = sum(c for c in coefficients if c < 0)
                return negative_sum if negative_sum < 0 else min(coefficients)

            import numpy as np
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh

            hilbert_dim = 2 ** n_qubits

            # Sparse Pauli matrices
            I2 = csr_matrix(np.eye(2, dtype=complex))
            pauli_sparse = {
                'I': I2,
                'X': csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
                'Y': csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
                'Z': csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
            }
            from scipy.sparse import kron as sp_kron

            # Build sparse Hamiltonian
            from scipy.sparse import csr_matrix as _csr
            H = _csr((hilbert_dim, hilbert_dim), dtype=complex)

            for coeff, pauli_str in zip(coefficients, pauli_strings):
                if abs(coeff) < 1e-12:
                    continue

                # Convert OpenFermion format if needed
                if '(' in str(pauli_str):
                    pauli_str = self._openfermion_to_pauli_string(pauli_str, n_qubits)

                # Build Kronecker product of single-qubit Pauli matrices (sparse)
                op = pauli_sparse[pauli_str[0]]
                for i in range(1, len(pauli_str)):
                    op = sp_kron(op, pauli_sparse[pauli_str[i]], format='csr')

                H = H + coeff * op

            # Diagonalize — use sparse eigsh for large systems, dense for small
            if n_qubits <= 10:
                eigenvalues = np.linalg.eigvalsh(H.toarray())
                ground_state_energy = float(eigenvalues[0])
            else:
                eigenvalues, _ = eigsh(H, k=1, which='SA')
                ground_state_energy = float(eigenvalues[0])
            
            logger.info(f"Ground state energy of {n_qubits}-qubit truncated system: {ground_state_energy:.6f}")
            return ground_state_energy
            
        except Exception as e:
            logger.warning(f"Could not calculate ground state energy: {e}. Using sum of negative coefficients as lower bound.")
            try:
                negative_sum = sum(c for c in coefficients if c < 0)
                return negative_sum if negative_sum < 0 else min(coefficients)
            except:
                return 0.0
    
    def to_pennylane(self):
        """Convert to PennyLane Hamiltonian format"""
        import pennylane as qml
        import re

        pauli_map = {
            'X': qml.PauliX,
            'Y': qml.PauliY,
            'Z': qml.PauliZ,
        }

        coeffs = []
        ops = []

        for coeff, pauli_string in zip(self.coefficients, self.pauli_strings):
            if np.abs(coeff) < 1e-12:
                continue

            coeffs.append(coeff)

            # Parse Pauli string – supports both OpenFermion "X(0) Z(2)"
            # and simple "IXZI" formats.
            pauli_ops = []
            ps = str(pauli_string)

            if '(' in ps:
                # OpenFermion format: "X(0) Z(2)"
                for m in re.finditer(r'([XYZ])\((\d+)\)', ps):
                    op_char, qubit = m.group(1), int(m.group(2))
                    pauli_ops.append(pauli_map[op_char](qubit))
            else:
                # Simple IXYZ string
                for i, p in enumerate(ps):
                    if p in pauli_map:
                        pauli_ops.append(pauli_map[p](i))

            if pauli_ops:
                if len(pauli_ops) == 1:
                    ops.append(pauli_ops[0])
                else:
                    op = pauli_ops[0]
                    for extra in pauli_ops[1:]:
                        op = op @ extra
                    ops.append(op)
            else:
                # All identity
                ops.append(qml.Identity(0))

        return qml.Hamiltonian(coeffs, ops)
    
    def to_qiskit(self):
        """Convert to Qiskit SparsePauliOp format"""
        from qiskit.quantum_info import SparsePauliOp
        
        # Qiskit uses reverse qubit ordering
        pauli_labels = [ps[::-1] for ps in self.pauli_strings]
        
        return SparsePauliOp.from_list(
            [(label, coeff) for label, coeff in zip(pauli_labels, self.coefficients)
             if np.abs(coeff) >= 1e-12]
        )
    
    def to_openfermion(self):
        """Convert to OpenFermion QubitOperator format"""
        from openfermion import QubitOperator
        
        hamiltonian = QubitOperator()
        
        for coeff, pauli_string in zip(self.coefficients, self.pauli_strings):
            if '(' in str(pauli_string):
                pauli_string = self._openfermion_to_pauli_string(pauli_string,self.n_qubits)
            if np.abs(coeff) < 1e-12:
                continue
            
            term = []
            for i, p in enumerate(pauli_string):
                if p != 'I':
                    term.append((i, p))
            
            hamiltonian += QubitOperator(tuple(term), coeff)
        
        return hamiltonian


class HamiltonianLoader:
    """Load and parse Hamiltonians from files or QMProt datasets"""

    def __init__(self, 
                 hamiltonians_dir: Union[str, Path],
                 molecules_json: Optional[Union[str, Path]] = None):
        """
        Initialize the Hamiltonian loader.
        Args:
            hamiltonians_dir: Directory containing Hamiltonian .txt files or datasets
            molecules_json: Path to QMProt.json metadata file
        """
        self.hamiltonians_dir = Path(hamiltonians_dir)
        self.molecules_json = Path(molecules_json) if molecules_json else None
        self.molecules: Dict[str, Molecule] = {}

        if self.molecules_json and self.molecules_json.exists():
            self._load_molecules_metadata()
    
    def _load_molecules_metadata(self):
        """Load molecule metadata from JSON file"""
        try:
            with open(self.molecules_json, 'r') as f:
                data = json.load(f)
            
            # Load test molecules (H2, H2O, etc.)
            for mol in data.get("test_molecules", []):
                self._add_molecule_from_dict(mol)
            
            # Load amino acids
            for mol in data.get("amino_acids", []):
                self._add_molecule_from_dict(mol)
            
            # Load other molecules
            for mol in data.get("other_molecules", []):
                self._add_molecule_from_dict(mol)
                
            logger.info(f"Loaded metadata for {len(self.molecules)} molecules")
            
        except Exception as e:
            logger.error(f"Error loading molecules JSON: {e}")
    
    def _add_molecule_from_dict(self, mol_dict: Dict):
        """Add a molecule from dictionary data"""
        abbrev = mol_dict.get("abbreviation", "")
        if not abbrev:
            return
            
        self.molecules[abbrev] = Molecule(
            abbreviation=abbrev,
            name=mol_dict.get("name", ""),
            n_qubits=mol_dict.get("n_qubits", 0),
            n_coefficients=mol_dict.get("n_coefficients", 0),
            reference_energy=mol_dict.get("energy", 0.0),
            hamiltonian_file=mol_dict.get("hamiltonian", f"hamiltonian_{abbrev}.txt"),
            n_electrons=mol_dict.get("n_electrons"),
            n_orbitals=mol_dict.get("n_orbitals"),
            charge=mol_dict.get("charge", 0),
            spin=mol_dict.get("spin", 0),
            basis=mol_dict.get("basis", "sto-3g"),
            coordinates=mol_dict.get("coordinates"),
            molecular_formula=mol_dict.get("mf"),
        )
    
    def load_hamiltonian(self, 
                         molecule_abbrev: Optional[str] = None,
                         hamiltonian_file: Optional[Union[str, Path]] = None) -> QubitHamiltonian:
        """
        Load a Hamiltonian from file.
        
        Automatically detects mode based on directory structure:
        - If hamiltonians_dir contains subdirectories with .h5 files -> H5 mode
        - If hamiltonians_dir contains hamiltonian_*.txt files -> Legacy mode
        """
        # Detect if using datasets/ (H5 mode) or data/hamiltonians (legacy mode)
        is_h5_mode = self._detect_h5_mode(molecule_abbrev)
        
        if is_h5_mode:
            return self._load_from_h5(molecule_abbrev)
        else:
            return self._load_from_txt(molecule_abbrev, hamiltonian_file)
    
    def _detect_h5_mode(self, molecule_abbrev: Optional[str] = None) -> bool:
        """Detect whether we're in H5 mode or legacy mode"""
        # Check if the directory name suggests datasets mode
        if self.hamiltonians_dir.name == "datasets":
            return True
        
        # Check if there are .h5 files in subdirectories
        if molecule_abbrev:
            h5_path = self.hamiltonians_dir / molecule_abbrev / f"{molecule_abbrev}.h5"
            if h5_path.exists():
                return True
        
        # Check for any .h5 files
        h5_files = list(self.hamiltonians_dir.glob("**/*.h5"))
        if h5_files:
            return True
        
        return False
    
    def _load_from_h5(self, molecule_abbrev: str) -> QubitHamiltonian:
        """Load Hamiltonian from an H5 dataset file"""
        if molecule_abbrev is None:
            raise ValueError("Must provide molecule_abbrev for H5 dataset mode.")
        
        # Find the .h5 file
        h5_path = self.hamiltonians_dir / molecule_abbrev / f"{molecule_abbrev}.h5"
        if not h5_path.exists():
            # Try alternative paths
            alt_paths = [
                self.hamiltonians_dir / f"{molecule_abbrev}.h5",
                self.hamiltonians_dir / molecule_abbrev.lower() / f"{molecule_abbrev.lower()}.h5",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    h5_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"H5 dataset file not found: {h5_path}")
        
        logger.info(f"Loading Hamiltonian from H5 file: {h5_path}")
        
        # Read the H5 file
        with h5py.File(h5_path, 'r') as f:
            # Extract metadata
            name = f['name'][()].decode() if 'name' in f else molecule_abbrev
            abbreviation = f['abbreviation'][()].decode() if 'abbreviation' in f else molecule_abbrev
            n_qubits = int(f['n_qubits'][()]) if 'n_qubits' in f else 0
            n_coefficients = int(f['n_coefficients'][()]) if 'n_coefficients' in f else 0
            reference_energy = float(f['energy'][()]) if 'energy' in f else 0.0
            n_electrons = int(f['n_electrons'][()]) if 'n_electrons' in f else None
            n_orbitals = int(f['n_orbitals'][()]) if 'n_orbitals' in f else None
            charge = int(f['charge'][()]) if 'charge' in f else 0
            spin = int(f['spin'][()]) if 'spin' in f else 0
            basis = f['basis'][()].decode() if 'basis' in f else "sto-3g"
            mf = f['mf'][()].decode() if 'mf' in f else None
            
            # Get all hamiltonian chunk keys
            ham_keys = sorted([k for k in f.keys() if k.startswith('hamiltonian')],
                            key=lambda x: (len(x), x))  # Sort to get proper order
            
            # Early truncation: limit chunks to read based on expected max terms
            max_terms_to_load = HAMILTONIAN_MAX_TERMS * 5  # Load 5x more than needed for selection
            estimated_terms_per_chunk = n_coefficients // len(ham_keys) if len(ham_keys) > 0 else n_coefficients
            chunks_to_read = min(len(ham_keys), max(1, max_terms_to_load // max(estimated_terms_per_chunk, 1)))
            
            # For very large molecules, read even fewer chunks
            if n_coefficients > 10_000_000:  # > 10M terms
                chunks_to_read = min(chunks_to_read, 50)  # Max 50 chunks for huge molecules
            
            if chunks_to_read < len(ham_keys):
                logger.info(f"Early truncation: reading {chunks_to_read}/{len(ham_keys)} chunks (estimated ~{chunks_to_read * estimated_terms_per_chunk:,} terms)")
                ham_keys = ham_keys[:chunks_to_read]
            
            # Collect hamiltonian chunks
            hamiltonian_chunks = []
            for key in ham_keys:
                chunk = f[key][()]
                if isinstance(chunk, bytes):
                    hamiltonian_chunks.append(chunk.decode())
                else:
                    hamiltonian_chunks.append(str(chunk))
        
        if not hamiltonian_chunks:
            raise ValueError(f"No hamiltonian data found in {h5_path}")
        
        # Combine and parse hamiltonian chunks
        full_hamiltonian = "".join(hamiltonian_chunks)
        coefficients, pauli_strings = self._parse_hamiltonian_string(full_hamiltonian, n_qubits=n_qubits)

        if not pauli_strings:
            raise ValueError(f"No valid Hamiltonian terms found in {h5_path}")

        # Use n_qubits from H5 metadata; only fallback to string length for legacy simple format
        actual_n_qubits = n_qubits if n_qubits > 0 else (len(pauli_strings[0]) if pauli_strings else 0)
        
        molecule = Molecule(
            abbreviation=abbreviation,
            name=name,
            n_qubits=actual_n_qubits,
            n_coefficients=len(coefficients),
            reference_energy=reference_energy,
            hamiltonian_file=str(h5_path),
            n_electrons=n_electrons,
            n_orbitals=n_orbitals,
            charge=charge,
            spin=spin,
            basis=basis,
            molecular_formula=mf,
        )
        
        logger.info(f"Loaded {len(coefficients)} terms from {h5_path.name} ({actual_n_qubits} qubits)")
        
        return QubitHamiltonian(
            molecule=molecule,
            coefficients=np.array(coefficients),
            pauli_strings=pauli_strings,
            n_qubits=actual_n_qubits,
            n_terms=len(coefficients),
        )
    
    def _parse_hamiltonian_string(self, hamiltonian_str: str, n_qubits: int = 0) -> Tuple[List[float], List[str]]:
        """Parse a hamiltonian string into coefficients and simple IXYZ Pauli strings.

        Handles OpenFermion format: 'coeff\\tZ(0) @ Z(1) @ X(5)'
        Converts to simple Pauli strings: 'ZZIIIXI...'

        Args:
            hamiltonian_str: Raw hamiltonian string from H5 or txt file
            n_qubits: Number of qubits. If 0, inferred from max qubit index.
        """
        import re
        lines = hamiltonian_str.split("\n")
        valid_lines = [line.strip() for line in lines
                      if line.strip() and "Coefficient" not in line and "Operators" not in line]

        coefficients = []
        raw_ops = []
        max_qubit = 0

        for line in valid_lines:
            # Tab-delimited: coeff<TAB>operators
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split(None, 1)  # fallback: split on first whitespace
            if len(parts) < 2:
                continue
            try:
                coeff = float(parts[0])
            except ValueError:
                continue

            op_str = parts[1].strip()
            coefficients.append(coeff)
            raw_ops.append(op_str)

            # Track the highest qubit index
            for m in re.finditer(r'\((\d+)\)', op_str):
                q = int(m.group(1))
                if q > max_qubit:
                    max_qubit = q

        # Use provided n_qubits or infer from max qubit index
        if n_qubits == 0:
            n_qubits = max_qubit + 1

        # Convert OpenFermion strings to simple IXYZ format
        pauli_strings = []
        for op_str in raw_ops:
            if 'Identity' in op_str:
                pauli_strings.append('I' * n_qubits)
                continue

            chars = ['I'] * n_qubits
            for m in re.finditer(r'([IXYZ])\((\d+)\)', op_str):
                p, q = m.group(1), int(m.group(2))
                if q < n_qubits:
                    chars[q] = p
            pauli_strings.append(''.join(chars))

        return coefficients, pauli_strings
    
    def _load_from_txt(self, 
                       molecule_abbrev: Optional[str] = None,
                       hamiltonian_file: Optional[Union[str, Path]] = None) -> QubitHamiltonian:
        """Load Hamiltonian from a legacy .txt file"""
        if molecule_abbrev:
            # Look up molecule metadata
            if molecule_abbrev not in self.molecules:
                raise ValueError(f"Molecule '{molecule_abbrev}' not found in metadata. "
                                f"Available: {list(self.molecules.keys())}")
            
            molecule = self.molecules[molecule_abbrev]
            file_path = self.hamiltonians_dir / molecule.hamiltonian_file
            
        elif hamiltonian_file:
            # Direct file path provided
            file_path = Path(hamiltonian_file)
            if not file_path.is_absolute():
                file_path = self.hamiltonians_dir / file_path
            
            # Create minimal molecule info
            molecule = Molecule(
                abbreviation=file_path.stem.replace("hamiltonian_", ""),
                name="Unknown",
                n_qubits=0,  # Will be determined from file
                n_coefficients=0,  # Will be determined from file
                reference_energy=0.0,
                hamiltonian_file=str(file_path),
            )
        else:
            raise ValueError("Must provide either molecule_abbrev or hamiltonian_file")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Hamiltonian file not found: {file_path}")
        
        logger.info(f"Loading Hamiltonian from {file_path}")
        
        # Parse the file
        coefficients, pauli_strings = self._parse_hamiltonian_file(file_path)
        
        if not pauli_strings:
            raise ValueError(f"No valid Hamiltonian terms found in {file_path}")
        
        # Determine number of qubits from Pauli strings
        n_qubits = len(pauli_strings[0]) if pauli_strings else 0
        
        # Update molecule info with actual values
        molecule.n_qubits = n_qubits
        molecule.n_coefficients = len(coefficients)
        
        return QubitHamiltonian(
            molecule=molecule,
            coefficients=np.array(coefficients),
            pauli_strings=pauli_strings,
            n_qubits=n_qubits,
            n_terms=len(coefficients),
        )
    
    def _parse_hamiltonian_file(self, file_path: Path) -> Tuple[List[float], List[str]]:
        """
        Parse a Hamiltonian text file.
        
        Expected format:
        Coefficient\tOperators
        0.123456\tIIII
        -0.234567\tZIIZ
        ...
        
        Returns:
            Tuple of (coefficients, pauli_strings)
        """
        coefficients = []
        pauli_strings = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header if present
        start_idx = 0
        if lines and ('Coefficient' in lines[0] or 'coefficient' in lines[0].lower()):
            start_idx = 1
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
            
            if len(parts) >= 2:
                try:
                    coeff = float(parts[0])
                    pauli_str = parts[1].strip()
                    
                    coefficients.append(coeff)
                    pauli_strings.append(pauli_str)
                except ValueError:
                    continue
        
        logger.info(f"Loaded {len(coefficients)} terms from {file_path.name}")
        return coefficients, pauli_strings
    
    def load_multiple(self, 
                      molecule_abbrevs: Optional[List[str]] = None,
                      hamiltonian_files: Optional[List[Union[str, Path]]] = None) -> List[QubitHamiltonian]:
        """
        Load multiple Hamiltonians.
        
        Args:
            molecule_abbrevs: List of molecule abbreviations
            hamiltonian_files: List of Hamiltonian file paths
            
        Returns:
            List of QubitHamiltonian objects
        """
        hamiltonians = []
        
        if molecule_abbrevs:
            for abbrev in molecule_abbrevs:
                try:
                    h = self.load_hamiltonian(molecule_abbrev=abbrev)
                    hamiltonians.append(h)
                except FileNotFoundError as e:
                    logger.warning(f"Could not load {abbrev}: {e}")
        
        if hamiltonian_files:
            for file_path in hamiltonian_files:
                try:
                    h = self.load_hamiltonian(hamiltonian_file=file_path)
                    hamiltonians.append(h)
                except FileNotFoundError as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        return hamiltonians
    
    def list_available_molecules(self) -> List[str]:
        """List all molecules with metadata or available .h5 datasets"""
        molecules = set(self.molecules.keys())
        
        # Also check for .h5 files in subdirectories
        if self.hamiltonians_dir.exists():
            for subdir in self.hamiltonians_dir.iterdir():
                if subdir.is_dir():
                    h5_file = subdir / f"{subdir.name}.h5"
                    if h5_file.exists():
                        molecules.add(subdir.name)
        
        return sorted(list(molecules))
    
    def list_available_hamiltonians(self) -> List[str]:
        """List all Hamiltonian files in the directory (both .txt and .h5)"""
        if not self.hamiltonians_dir.exists():
            return []
        
        hamiltonians = set()
        
        # Check for legacy .txt files
        for f in self.hamiltonians_dir.glob("hamiltonian_*.txt"):
            hamiltonians.add(f.stem.replace("hamiltonian_", ""))
        
        # Check for .h5 files in subdirectories
        for subdir in self.hamiltonians_dir.iterdir():
            if subdir.is_dir():
                h5_file = subdir / f"{subdir.name}.h5"
                if h5_file.exists():
                    hamiltonians.add(subdir.name)
        
        return sorted(list(hamiltonians))
    
    def get_molecule_info(self, abbreviation: str) -> Optional[Molecule]:
        """Get molecule information by abbreviation"""
        return self.molecules.get(abbreviation)
