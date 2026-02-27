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
    TRUNCATION_MODE, ACTIVE_SPACE_BASIS,
)
import ast
from core import HamiltonianLoader, ResultsManager
from core.hamiltonian_loader import QubitHamiltonian
from core.backend_manager import BackendConfig
from algorithms import ALGORITHMS, get_algorithm, list_algorithms
from plotting import VQEVisualizer
from plotting import molecule_plots

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
        self._run_csv_path = None  # Incremental CSV in timestamped run folder

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

        truncation_mode = kwargs.pop("truncation_mode", TRUNCATION_MODE)
        logger.info(f"Truncation mode: {truncation_mode}")

        if truncation_mode == "active_space":
            # Active space truncation: PySCF HF -> MP2 -> CASCI -> OpenFermion
            from active_space_truncation.run_pipeline import run_pipeline as run_active_space_pipeline
            basis = kwargs.pop("active_space_basis", ACTIVE_SPACE_BASIS)
            pipeline_result = run_active_space_pipeline(molecule=molecule, basis=basis, quiet=True)
            hamiltonian = pipeline_result["hamiltonian"].qubit_hamiltonian
            core_energy = pipeline_result["hamiltonian"].core_energy
            casci_energy = pipeline_result["active_space"].casci_energy
            logger.info(f"Active space Hamiltonian: {hamiltonian.n_qubits} qubits, {hamiltonian.n_terms} terms")
            logger.info(f"Core energy (frozen core + nuclear repulsion): {core_energy:.10f} Ha")
            logger.info(f"CASCI reference energy: {casci_energy:.10f} Ha")
        else:
            # Coefficient-based truncation (legacy): load H5 and keep largest |coeff| terms
            if Path(molecule).exists():
                hamiltonian = self.loader.load_hamiltonian(hamiltonian_file=molecule)
            else:
                hamiltonian = self.loader.load_hamiltonian(molecule_abbrev=molecule)

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

        # Incrementally save to CSV in timestamped run folder
        self._append_result_to_run_csv(result)

        return result.to_dict()

    def _append_result_to_run_csv(self, result):
        """Append a single VQEResult to the incremental CSV in the timestamped run folder."""
        import csv
        from datetime import datetime

        # Create timestamped run folder on first call
        if self._run_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.plots_dir / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            self._run_csv_path = run_dir / "run_results.csv"
            self._run_dir = run_dir
            logger.info(f"Incremental results will be saved to {self._run_csv_path}")

        fields = [
            "molecule_abbrev", "molecule_name", "algorithm_name",
            "calculated_energy", "reference_energy", "error", "relative_error",
            "n_iterations", "n_qubits", "n_parameters", "runtime_seconds",
            "converged", "hf_energy",
        ]

        write_header = not self._run_csv_path.exists()
        try:
            with open(self._run_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                row = {k: getattr(result, k, None) for k in fields}
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"Could not append to run CSV: {e}")

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
        
        # Also use molecule_plots CSV export for consistency
        if self.results_manager.results:
            from utils.csv_export import export_results_to_csv
            csv_path = self.results_dir / "results" / "csv" / "molecule_plots_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            export_results_to_csv(self.results_manager.results, str(csv_path))
    
    def generate_plots(self):
        """Generate all visualization plots"""
        if self.visualizer is None:
            self.visualizer = VQEVisualizer(self.results_manager, self.plots_dir)
            # If a run folder was already created for incremental CSV, reuse it
            if hasattr(self, '_run_dir') and self._run_dir is not None:
                self.visualizer.output_dir = self._run_dir
        self.visualizer.generate_all_plots()

        # Also generate specialized molecule plots with HF reference
        if self.results_manager.results:
            logger.info("Generating specialized molecule plots with HF reference...")
            molecule_plots.plot_selected_molecules_with_hf_ref(
                results=self.results_manager.results,
                output_dir=str(self.plots_dir),
                filename="selected_molecules_withHFref.png"
            )

        # Save JSON and CSV into the timestamped plot folder
        plot_dir = self.visualizer.output_dir
        if self.results_manager.results:
            import json
            # JSON
            json_path = plot_dir / "all_results.json"
            with open(json_path, 'w') as f:
                json.dump([r.to_dict() for r in self.results_manager.results], f, indent=2)
            logger.info(f"Saved results JSON to {json_path}")
            # CSV
            try:
                from utils.csv_export import export_results_to_csv
                csv_path = plot_dir / "all_results.csv"
                export_results_to_csv(self.results_manager.results, str(csv_path))
                logger.info(f"Saved results CSV to {csv_path}")
            except Exception as e:
                logger.warning(f"Could not save CSV to plot folder: {e}")
    
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
    parser.add_argument('--truncation-mode', type=str, default=TRUNCATION_MODE,
                       choices=['active_space', 'coefficient'],
                       help=f'Truncation method: active_space (PySCF-based, default) or coefficient (legacy magnitude-based)')
    parser.add_argument('--active-space-basis', type=str, default=ACTIVE_SPACE_BASIS,
                       help=f'Basis set for active space truncation (default: {ACTIVE_SPACE_BASIS})')
    parser.add_argument('--max-hamiltonian-terms', type=int, default=HAMILTONIAN_MAX_TERMS,
                       help=f'Max hamiltonian terms to keep for coefficient mode (default: {HAMILTONIAN_MAX_TERMS})')
    parser.add_argument('--target-qubits', type=int, default=HAMILTONIAN_TARGET_QUBITS,
                       help=f'Target number of qubits for coefficient truncation (default: {HAMILTONIAN_TARGET_QUBITS})')
    
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
        "truncation_mode": args.truncation_mode,
        "active_space_basis": args.active_space_basis,
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
