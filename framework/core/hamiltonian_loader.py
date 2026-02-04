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
            remapped_pauli = []
            for pauli_str in truncated_pauli:
                remapped = ['I'] * len(pauli_str)
                for wire, pauli in enumerate(pauli_str):
                    if wire in wire_mapping:
                        remapped[wire_mapping[wire]] = pauli
                    elif pauli != 'I':
                        # Wire not in mapping, skip
                        break
                else:
                    remapped_pauli.append(''.join(remapped))
                    continue
                remapped_pauli.append(pauli_str)  # Keep original if remap failed
            truncated_pauli = remapped_pauli
            actual_n_qubits = len(wire_mapping)
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
            if n_qubits > 10:
                # For large systems, use a lower bound: sum of negative terms
                logger.info(f"System too large for exact diagonalization ({n_qubits} qubits). Using lower bound.")
                negative_sum = sum(c for c in coefficients if c < 0)
                return negative_sum if negative_sum < 0 else min(coefficients)
            
            import numpy as np
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            
            # For small systems, use exact diagonalization
            hilbert_dim = 2 ** n_qubits
            
            # Pauli matrices
            pauli_map = {
                'I': np.array([[1, 0], [0, 1]], dtype=complex),
                'X': np.array([[0, 1], [1, 0]], dtype=complex),
                'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
                'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            }
            
            # Build Hamiltonian matrix
            H = np.zeros((hilbert_dim, hilbert_dim), dtype=complex)
            
            for coeff, pauli_str in zip(coefficients, pauli_strings):
                if abs(coeff) < 1e-12:
                    continue
                
                # Convert OpenFermion format if needed
                if '(' in str(pauli_str):
                    pauli_str = self._openfermion_to_pauli_string(pauli_str, n_qubits)
                
                # Build single-qubit operators
                op = pauli_map[pauli_str[0]]
                for i in range(1, len(pauli_str)):
                    op = np.kron(op, pauli_map[pauli_str[i]])
                
                H += coeff * op
            
            # Diagonalize to find ground state
            eigenvalues = np.linalg.eigvalsh(H)
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
        
        coeffs = []
        ops = []
        
        for coeff, pauli_string in zip(self.coefficients, self.pauli_strings):
            if np.abs(coeff) < 1e-12:
                continue
            
            coeffs.append(coeff)
            
            # Parse Pauli string to PennyLane operators
            pauli_ops = []
            for i, p in enumerate(pauli_string):
                if p == 'X':
                    pauli_ops.append(qml.PauliX(i))
                elif p == 'Y':
                    pauli_ops.append(qml.PauliY(i))
                elif p == 'Z':
                    pauli_ops.append(qml.PauliZ(i))
                # 'I' is identity, skip
            
            if pauli_ops:
                if len(pauli_ops) == 1:
                    ops.append(pauli_ops[0])
                else:
                    ops.append(pauli_ops[0])
                    for op in pauli_ops[1:]:
                        ops[-1] = ops[-1] @ op
            else:
                # All identity - use Identity on qubit 0
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
        coefficients, pauli_strings = self._parse_hamiltonian_string(full_hamiltonian)
        
        if not pauli_strings:
            raise ValueError(f"No valid Hamiltonian terms found in {h5_path}")
        
        # Determine actual n_qubits from pauli strings if not set
        actual_n_qubits = len(pauli_strings[0]) if pauli_strings else n_qubits
        
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
    
    def _parse_hamiltonian_string(self, hamiltonian_str: str) -> Tuple[List[float], List[str]]:
        """Parse a hamiltonian string into coefficients and pauli strings"""
        lines = hamiltonian_str.split("\n")
        valid_lines = [line.strip() for line in lines 
                      if line.strip() and "Coefficient" not in line and "Operators" not in line]
        
        coefficients = []
        pauli_strings = []
        
        for line in valid_lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coeff = float(parts[0])
                    pauli_str = parts[1].strip()
                    coefficients.append(coeff)
                    pauli_strings.append(pauli_str)
                except ValueError:
                    continue
        
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
