#!/usr/bin/env python3
"""
VQE Framework Main Runner

Main entry point for running VQE algorithms on protein Hamiltonians.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# hide warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATASETS_DIR, HAMILTONIANS_DIR, MOLECULES_JSON, RESULTS_DIR, PLOTS_DIR,
    DEFAULT_OPTIMIZER, MAX_ITERATIONS, CONVERGENCE_THRESHOLD,
    N_SHOTS, RANDOM_SEED, LOG_LEVEL, LOG_FILE, HAMILTONIAN_MODE,
    HAMILTONIAN_MAX_TERMS, HAMILTONIAN_TARGET_QUBITS,
    BACKEND_TYPE, NOISE_MODEL, NOISE_STRENGTH,
)
import ast
from core import HamiltonianLoader, ResultsManager
from core.hamiltonian_loader import QubitHamiltonian
from core.backend_manager import BackendConfig
from algorithms import ALGORITHMS, get_algorithm, list_algorithms
from plotting import VQEVisualizer

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE) if LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_if_qmprot_hamiltonian():
    vars_path = Path(__file__).parent / ".vars"
    if vars_path.exists():
        with open(vars_path) as f:
            for line in f:
                if line.strip().startswith("if_qmprot_hamiltonian"):
                    key, val = line.split("=", 1)
                    return ast.literal_eval(val.strip())
    return False


class VQEFramework:
    """
    Main framework class for running VQE experiments.
    """
    
    def __init__(self,
                 hamiltonians_dir: Optional[Path] = None,
                 molecules_json: Optional[Path] = None,
                 results_dir: Optional[Path] = None,
                 plots_dir: Optional[Path] = None,
                 legacy_mode: bool = False):
        """
        Initialize the VQE Framework.
        
        Args:
            hamiltonians_dir: Directory containing Hamiltonian files
            molecules_json: Path to molecules metadata JSON
            results_dir: Directory for saving results
            plots_dir: Directory for saving plots
            legacy_mode: If True, use legacy .txt hamiltonian files; if False (default), use .h5 datasets
        """
        self.legacy_mode = legacy_mode
        
        # Determine hamiltonian source based on mode
        if legacy_mode:
            # Legacy mode: use .txt files from data/hamiltonians
            self.hamiltonians_dir = hamiltonians_dir or HAMILTONIANS_DIR
            logger.info("Using LEGACY mode (.txt hamiltonian files)")
        else:
            # Default mode: use .h5 files from datasets/
            self.hamiltonians_dir = hamiltonians_dir or DATASETS_DIR
            logger.info("Using H5 mode (.h5 dataset files)")
        
        self.molecules_json = molecules_json or MOLECULES_JSON
        self.results_dir = results_dir or RESULTS_DIR
        self.plots_dir = plots_dir or PLOTS_DIR
        
        # Initialize components
        self.loader = HamiltonianLoader(self.hamiltonians_dir, self.molecules_json)
        self.results_manager = ResultsManager(self.results_dir)
        self.visualizer = None  # Lazy initialization
        
        logger.info(f"VQE Framework initialized")
        logger.info(f"Hamiltonians directory: {self.hamiltonians_dir}")
        logger.info(f"Available algorithms: {list_algorithms()}")
    
    def run_single(self,
                   molecule: str,
                   algorithm: str,
                   **kwargs) -> dict:
        """
        Run a single VQE experiment.
        
        Args:
            molecule: Molecule abbreviation or Hamiltonian file path
            algorithm: Algorithm name
            **kwargs: Additional algorithm parameters
            
        Returns:
            VQEResult as dictionary
        """
        logger.info(f"Running {algorithm} on {molecule}")
        
        # Load Hamiltonian
        if Path(molecule).exists():
            hamiltonian = self.loader.load_hamiltonian(hamiltonian_file=molecule)
        else:
            hamiltonian = self.loader.load_hamiltonian(molecule_abbrev=molecule)
        
        # Truncate hamiltonian if too many terms
        max_terms = kwargs.get("max_hamiltonian_terms", HAMILTONIAN_MAX_TERMS)
        target_qubits = kwargs.get("target_qubits", HAMILTONIAN_TARGET_QUBITS)
        if hamiltonian.n_terms > max_terms:
            hamiltonian = hamiltonian.truncate(max_terms=max_terms, target_qubits=target_qubits)
        
        # Get algorithm class
        AlgorithmClass = get_algorithm(algorithm)
        
        # Merge default params with kwargs
        params = {
            "optimizer": kwargs.get("optimizer", DEFAULT_OPTIMIZER),
            "max_iterations": kwargs.get("max_iterations", MAX_ITERATIONS),
            "convergence_threshold": kwargs.get("convergence_threshold", CONVERGENCE_THRESHOLD),
            "n_shots": kwargs.get("n_shots", N_SHOTS),
            "random_seed": kwargs.get("random_seed", RANDOM_SEED),
        }
        params.update(kwargs)
        
        # Build backend config from params
        backend_type = params.pop("backend_type", BACKEND_TYPE)
        noise_model = params.pop("noise_model", None)
        noise_strength = params.pop("noise_strength", NOISE_STRENGTH)
        
        if backend_type == "noisy" and noise_model:
            backend_config = BackendConfig.noisy(
                n_qubits=hamiltonian.n_qubits,
                noise_model=noise_model,
                noise_strength=noise_strength,
            )
        else:
            backend_config = BackendConfig.statevector(
                n_qubits=hamiltonian.n_qubits,
            )
        params["backend_config"] = backend_config
        
        # Create and run algorithm
        vqe = AlgorithmClass(hamiltonian, **params)
        result = vqe.run()
        
        # Store result
        self.results_manager.add_result(result)
        
        return result.to_dict()
    
    def run_molecule(self,
                     molecule: str,
                     algorithms: Optional[List[str]] = None,
                     **kwargs) -> List[dict]:
        """
        Run all algorithms on a single molecule.
        
        Args:
            molecule: Molecule abbreviation
            algorithms: List of algorithm names (all if None)
            **kwargs: Additional parameters
            
        Returns:
            List of VQEResult dictionaries
        """
        if algorithms is None:
            algorithms = list_algorithms()
        
        results = []
        for alg in algorithms:
            try:
                result = self.run_single(molecule, alg, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running {alg} on {molecule}: {e}")
        
        return results
    
    def run_algorithm(self,
                      algorithm: str,
                      molecules: Optional[List[str]] = None,
                      **kwargs) -> List[dict]:
        """
        Run a single algorithm on all molecules.
        
        Args:
            algorithm: Algorithm name
            molecules: List of molecule abbreviations (all available if None)
            **kwargs: Additional parameters
            
        Returns:
            List of VQEResult dictionaries
        """
        if molecules is None:
            molecules = self.loader.list_available_hamiltonians()
        
        results = []
        for mol in molecules:
            try:
                result = self.run_single(mol, algorithm, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running {algorithm} on {mol}: {e}")
        
        return results
    
    def run_all(self,
                molecules: Optional[List[str]] = None,
                algorithms: Optional[List[str]] = None,
                **kwargs) -> List[dict]:
        """
        Run all algorithms on all molecules.
        
        Args:
            molecules: List of molecules (all available if None)
            algorithms: List of algorithms (all if None)
            **kwargs: Additional parameters
            
        Returns:
            List of all VQEResult dictionaries
        """
        if molecules is None:
            molecules = self.loader.list_available_hamiltonians()
        if algorithms is None:
            algorithms = list_algorithms()
        
        logger.info(f"Running {len(algorithms)} algorithms on {len(molecules)} molecules")
        
        all_results = []
        for mol in molecules:
            for alg in algorithms:
                try:
                    result = self.run_single(mol, alg, **kwargs)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error running {alg} on {mol}: {e}")
        
        return all_results
    
    def save_results(self, filename: Optional[str] = None):
        """Save all results to files"""
        self.results_manager.save_all_results(filename)
        self.results_manager.save_to_csv()
    
    def generate_plots(self):
        """Generate all visualization plots"""
        if self.visualizer is None:
            self.visualizer = VQEVisualizer(self.results_manager, self.plots_dir)
        self.visualizer.generate_all_plots()
    
    def plot_molecule(self, molecule: str):
        """Generate plot for a specific molecule"""
        if self.visualizer is None:
            self.visualizer = VQEVisualizer(self.results_manager, self.plots_dir)
        self.visualizer.plot_molecule_comparison(molecule)
    
    def get_summary(self) -> dict:
        """Get summary statistics"""
        return self.results_manager.get_summary_stats()
    
    def list_molecules(self) -> List[str]:
        """List available molecules"""
        return self.loader.list_available_hamiltonians()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="VQE Framework for Protein Hamiltonians",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all algorithms on all molecules
  python main.py --all

  # Run specific algorithm on specific molecule
  python main.py --molecule trp --algorithm vanilla_vqe

  # Run all algorithms on one molecule
  python main.py --molecule his --all-algorithms

  # Run one algorithm on all molecules
  python main.py --algorithm adapt_vqe --all-molecules

  # Generate plots only (from existing results)
  python main.py --plot-only --results-file results.json

  # List available options
  python main.py --list-algorithms
  python main.py --list-molecules
        """
    )
    
    # Mode selection
    parser.add_argument('--all', action='store_true',
                       help='Run all algorithms on all molecules')
    parser.add_argument('--all-algorithms', action='store_true',
                       help='Run all algorithms on specified molecule')
    parser.add_argument('--all-molecules', action='store_true',
                       help='Run specified algorithm on all molecules')
    parser.add_argument('--plot-only', action='store_true',
                       help='Generate plots from existing results')
    
    # Specification
    parser.add_argument('--molecule', '-m', type=str, nargs='+',
                       help='Molecule abbreviation (e.g., trp, his). Accepts multiple values')
    parser.add_argument('--algorithm', '-a', type=str,
                       help='Algorithm name')
    parser.add_argument('--molecules', type=str, nargs='+',
                       help='Multiple molecules')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       help='Multiple algorithms')
    
    # Paths
    parser.add_argument('--hamiltonians-dir', type=str,
                       help='Directory containing Hamiltonian files')
    parser.add_argument('--molecules-json', type=str,
                       help='Path to molecules metadata JSON')
    parser.add_argument('--results-dir', type=str,
                       help='Directory for results')
    parser.add_argument('--plots-dir', type=str,
                       help='Directory for plots')
    parser.add_argument('--results-file', type=str,
                       help='Load results from file')
    
    # VQE parameters
    parser.add_argument('--optimizer', type=str, default=DEFAULT_OPTIMIZER,
                       help=f'Optimizer (default: {DEFAULT_OPTIMIZER})')
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS,
                       help=f'Max iterations (default: {MAX_ITERATIONS})')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of ansatz layers')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')
    
    # Hamiltonian truncation parameters
    parser.add_argument('--max-hamiltonian-terms', type=int, default=HAMILTONIAN_MAX_TERMS,
                       help=f'Max hamiltonian terms to keep (default: {HAMILTONIAN_MAX_TERMS})')
    parser.add_argument('--target-qubits', type=int, default=HAMILTONIAN_TARGET_QUBITS,
                       help=f'Target number of qubits for truncation (default: {HAMILTONIAN_TARGET_QUBITS})')
    
    # Backend & noise parameters
    parser.add_argument('--backend-type', type=str, default=BACKEND_TYPE,
                       choices=['statevector', 'noisy'],
                       help='Simulation backend type (default: statevector)')
    parser.add_argument('--noise-model', type=str, default=None,
                       choices=['depolarizing', 'bitflip', 'phaseflip',
                                'amplitude_damping', 'phase_damping'],
                       help='Noise model for noisy backend')
    parser.add_argument('--noise-strength', type=float, default=NOISE_STRENGTH,
                       help=f'Noise strength / error probability (default: {NOISE_STRENGTH})')
    parser.add_argument('--run-both-backends', action='store_true',
                       help='Run every experiment twice: once statevector, once noisy')
    
    # Listing
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List available algorithms')
    parser.add_argument('--list-molecules', action='store_true',
                       help='List available molecules')
    
    # Output
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results (default: True)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate plots (default: True)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Do not generate plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    # Hamiltonian mode
    parser.add_argument('--legacy', action='store_true',
                       help='Use legacy .txt hamiltonian files instead of .h5 datasets')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle listing
    if args.list_algorithms:
        print("Available algorithms:")
        for alg in list_algorithms():
            print(f"  - {alg}")
        return
    
    # Initialize framework
    framework = VQEFramework(
        hamiltonians_dir=Path(args.hamiltonians_dir) if args.hamiltonians_dir else None,
        molecules_json=Path(args.molecules_json) if args.molecules_json else None,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        plots_dir=Path(args.plots_dir) if args.plots_dir else None,
        legacy_mode=args.legacy,
    )
    
    if args.list_molecules:
        print("Available molecules with Hamiltonians:")
        for mol in framework.list_molecules():
            print(f"  - {mol}")
        return
    
    # VQE parameters
    vqe_params = {
        "optimizer": args.optimizer,
        "max_iterations": args.max_iterations,
        "n_layers": args.n_layers,
        "random_seed": args.seed,
        "max_hamiltonian_terms": args.max_hamiltonian_terms,
        "target_qubits": args.target_qubits,
        "backend_type": args.backend_type,
        "noise_model": args.noise_model,
        "noise_strength": args.noise_strength,
    }
    
    # ── Helper: run with --run-both-backends support ────────────────────
    def _run_experiment(run_fn, *run_args, **run_kwargs):
        """Execute run_fn once per requested backend."""
        if args.run_both_backends:
            # First pass: statevector
            sv_params = dict(run_kwargs)
            sv_params["backend_type"] = "statevector"
            sv_params["noise_model"] = None
            logger.info("=== Running with STATEVECTOR backend ===")
            run_fn(*run_args, **sv_params)

            # Second pass: noisy
            noisy_params = dict(run_kwargs)
            noisy_params["backend_type"] = "noisy"
            noisy_params["noise_model"] = args.noise_model or "depolarizing"
            noisy_params["noise_strength"] = args.noise_strength
            logger.info("=== Running with NOISY backend ===")
            run_fn(*run_args, **noisy_params)
        else:
            run_fn(*run_args, **run_kwargs)

    # Run VQE
    if args.plot_only:
        if args.results_file:
            results = framework.results_manager.load_results(args.results_file)
            framework.results_manager.add_results(results)
        framework.generate_plots()
        
    elif args.all:
        _run_experiment(
            framework.run_all,
            molecules=args.molecules,
            algorithms=args.algorithms,
            **vqe_params,
        )
        
    elif args.all_algorithms and args.molecule:
        if isinstance(args.molecule, list):
            for mol in args.molecule:
                _run_experiment(framework.run_molecule, mol,
                                algorithms=args.algorithms, **vqe_params)
        else:
            _run_experiment(framework.run_molecule, args.molecule,
                            algorithms=args.algorithms, **vqe_params)
        
    elif args.all_molecules and args.algorithm:
        _run_experiment(framework.run_algorithm, args.algorithm,
                        molecules=args.molecules, **vqe_params)
        
    elif args.molecule and args.algorithm:
        if isinstance(args.molecule, list):
            for mol in args.molecule:
                _run_experiment(framework.run_single, mol, args.algorithm, **vqe_params)
        else:
            _run_experiment(framework.run_single, args.molecule, args.algorithm, **vqe_params)
        
    else:
        parser.print_help()
        return
    
    # Save and plot
    if not args.no_save:
        framework.save_results()
    
    if not args.no_plot and not args.plot_only:
        framework.generate_plots()
    
    # Print summary
    summary = framework.get_summary()
    if summary:
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
