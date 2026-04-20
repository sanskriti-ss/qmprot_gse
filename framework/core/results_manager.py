"""
Results Manager Module

Handles saving, loading, and organizing VQE results.
"""
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging

from .base_vqe import VQEResult

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manage VQE results storage and retrieval"""
    
    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize the results manager.
        
        Args:
            results_dir: Directory to store results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.json_dir = self.results_dir / "json"
        self.csv_dir = self.results_dir / "csv"
        self.json_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)
        
        # In-memory results cache
        self.results: List[VQEResult] = []
    
    def add_result(self, result: VQEResult):
        """Add a result to the manager"""
        self.results.append(result)
    
    def add_results(self, results: List[VQEResult]):
        """Add multiple results"""
        self.results.extend(results)
    
    def save_result(self, result: VQEResult, filename: Optional[str] = None) -> Path:
        """
        Save a single result to JSON file.
        
        Args:
            result: VQEResult to save
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.molecule_abbrev}_{result.algorithm_name}_{timestamp}.json"
        
        filepath = self.json_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Saved result to {filepath}")
        return filepath
    
    def save_all_results(self, filename: Optional[str] = None) -> Path:
        """
        Save all results to a single JSON file.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"all_results_{timestamp}.json"
        
        filepath = self.json_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "n_results": len(self.results),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def save_to_csv(self, filename: Optional[str] = None) -> Path:
        """
        Save results summary to CSV.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_summary_{timestamp}.csv"
        
        filepath = self.csv_dir / filename

        columns = [
            "molecule_abbrev",
            "molecule_name",
            "algorithm",
            "calculated_energy",
            "reference_energy",
            "error",
            "relative_error",
            "iterations",
            "n_iterations",
            "n_qubits",
            "n_parameters",
            "runtime_seconds",
            "converged",
            "backend_type",
            "noise_model",
            "noise_strength",
            "hf_energy",
        ]

        rows = []
        for r in self.results:
            rows.append(
                {
                    "molecule_abbrev": r.molecule_abbrev,
                    "molecule_name": r.molecule_name,
                    "algorithm": r.algorithm_name,
                    "calculated_energy": r.calculated_energy,
                    "reference_energy": r.reference_energy,
                    "error": r.error,
                    "relative_error": r.relative_error,
                    "iterations": r.n_iterations,
                    "n_iterations": r.n_iterations,
                    "n_qubits": r.n_qubits,
                    "n_parameters": r.n_parameters,
                    "runtime_seconds": r.runtime_seconds,
                    "converged": r.converged,
                    "backend_type": r.backend_type,
                    "noise_model": r.noise_model,
                    "noise_strength": r.noise_strength,
                    "hf_energy": r.hf_energy,
                }
            )

        if not rows:
            df = pd.DataFrame(columns=columns)
            logger.warning(
                "Saved CSV with headers only (0 results). "
                "No run reached add_result — check logs for errors in run_single / pipeline / VQE."
            )
        else:
            df = pd.DataFrame(rows)

        df.to_csv(filepath, index=False)

        logger.info(f"Saved CSV summary to {filepath} ({len(self.results)} row(s))")
        return filepath
    
    def load_results(self, filepath: Union[str, Path]) -> List[VQEResult]:
        """
        Load results from JSON (full ``VQEResult`` dump) or a summary CSV.

        CSV must match ``save_to_csv`` columns; ``algorithm`` is mapped to
        ``algorithm_name``. Rows have empty ``convergence_history`` (plots that
        need histories work best from ``all_results_*.json``).
        """
        filepath = Path(filepath)
        if filepath.suffix.lower() == ".csv":
            return self._load_results_csv(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Handle single result or multiple results
        if "results" in data:
            results_data = data["results"]
        else:
            results_data = [data]
        
        results = []
        for rd in results_data:
            result = VQEResult(
                molecule_abbrev=rd["molecule_abbrev"],
                molecule_name=rd["molecule_name"],
                algorithm_name=rd["algorithm_name"],
                calculated_energy=rd["calculated_energy"],
                reference_energy=rd["reference_energy"],
                error=rd["error"],
                relative_error=rd["relative_error"],
                n_iterations=rd["n_iterations"],
                n_qubits=rd["n_qubits"],
                n_parameters=rd["n_parameters"],
                runtime_seconds=rd["runtime_seconds"],
                convergence_history=rd.get("convergence_history", []),
                optimal_parameters=np.array(rd["optimal_parameters"]) if rd.get("optimal_parameters") else None,
                final_gradient_norm=rd.get("final_gradient_norm"),
                converged=rd.get("converged", False),
                metadata=rd.get("metadata", {}),
                backend_type=rd.get("backend_type", "statevector"),
                noise_model=rd.get("noise_model"),
                noise_strength=rd.get("noise_strength", 0.0),
                hf_energy=rd.get("hf_energy"),
            )
            results.append(result)

        return results

    def _load_results_csv(self, filepath: Path) -> List[VQEResult]:
        df = pd.read_csv(filepath)
        if "algorithm_name" not in df.columns and "algorithm" in df.columns:
            df = df.rename(columns={"algorithm": "algorithm_name"})
        # Backward/forward compatibility: allow either naming.
        if "n_iterations" not in df.columns and "iterations" in df.columns:
            df = df.rename(columns={"iterations": "n_iterations"})
        required = {
            "molecule_abbrev",
            "molecule_name",
            "algorithm_name",
            "calculated_energy",
            "reference_energy",
            "error",
            "relative_error",
            "n_iterations",
            "n_qubits",
            "n_parameters",
            "runtime_seconds",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {filepath} missing columns {sorted(missing)}. "
                "Use a results_summary_*.csv from this framework or full JSON."
            )

        results: List[VQEResult] = []
        for _, row in df.iterrows():
            rd = row.where(pd.notna(row), None).to_dict()

            def _f(key: str, default: Optional[float] = None) -> Optional[float]:
                v = rd.get(key, default)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return default
                return float(v)

            def _i(key: str) -> int:
                v = rd[key]
                return int(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0

            def _b(key: str) -> bool:
                v = rd.get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return False
                if isinstance(v, bool):
                    return v
                return str(v).strip().lower() in ("1", "true", "yes")

            noise_model = rd.get("noise_model")
            if noise_model is not None and isinstance(noise_model, float) and np.isnan(noise_model):
                noise_model = None
            if isinstance(noise_model, str) and not noise_model.strip():
                noise_model = None

            hf = _f("hf_energy", None)
            n_strength = _f("noise_strength", 0.0)
            if n_strength is None:
                n_strength = 0.0

            results.append(
                VQEResult(
                    molecule_abbrev=str(rd["molecule_abbrev"]),
                    molecule_name=str(rd["molecule_name"]),
                    algorithm_name=str(rd["algorithm_name"]),
                    calculated_energy=float(rd["calculated_energy"]),
                    reference_energy=float(rd["reference_energy"]),
                    error=float(rd["error"]),
                    relative_error=float(rd["relative_error"]),
                    n_iterations=_i("n_iterations"),
                    n_qubits=_i("n_qubits"),
                    n_parameters=_i("n_parameters"),
                    runtime_seconds=float(rd["runtime_seconds"] or 0.0),
                    convergence_history=[],
                    optimal_parameters=None,
                    final_gradient_norm=None,
                    converged=_b("converged"),
                    metadata={},
                    backend_type=str(rd.get("backend_type") or "statevector"),
                    noise_model=noise_model,
                    noise_strength=float(n_strength),
                    hf_energy=hf,
                )
            )

        logger.info("Loaded %s rows from CSV %s", len(results), filepath)
        return results

    def get_results_by_molecule(self, molecule_abbrev: str) -> List[VQEResult]:
        """Get all results for a specific molecule"""
        return [r for r in self.results if r.molecule_abbrev == molecule_abbrev]
    
    def get_results_by_algorithm(self, algorithm_name: str) -> List[VQEResult]:
        """Get all results for a specific algorithm"""
        return [r for r in self.results if r.algorithm_name == algorithm_name]
    
    def get_best_result(self, molecule_abbrev: str) -> Optional[VQEResult]:
        """Get the best (lowest energy) result for a molecule"""
        mol_results = self.get_results_by_molecule(molecule_abbrev)
        if not mol_results:
            return None
        return min(mol_results, key=lambda r: r.calculated_energy)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to a pandas DataFrame"""
        rows = [r.to_dict() for r in self.results]
        df = pd.DataFrame(rows)
        if not df.empty and "iterations" not in df.columns and "n_iterations" in df.columns:
            df["iterations"] = df["n_iterations"]
        return df
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of all results"""
        if not self.results:
            return {}
        
        df = self.to_dataframe()
        
        return {
            "n_molecules": df["molecule_abbrev"].nunique(),
            "n_algorithms": df["algorithm_name"].nunique(),
            "n_total_runs": len(df),
            "mean_error": df["error"].mean(),
            "std_error": df["error"].std(),
            "mean_runtime": df["runtime_seconds"].mean(),
            "mean_iterations": df["n_iterations"].mean(),
            "convergence_rate": df["converged"].mean(),
            "best_algorithm_by_error": df.groupby("algorithm_name")["error"].mean().abs().idxmin(),
            "best_algorithm_by_runtime": df.groupby("algorithm_name")["runtime_seconds"].mean().idxmin(),
            "best_algorithm_by_iterations": df.groupby("algorithm_name")["n_iterations"].mean().idxmin(),
        }
    
    def clear(self):
        """Clear all results from memory"""
        self.results = []
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __repr__(self) -> str:
        return f"ResultsManager(n_results={len(self.results)}, dir={self.results_dir})"
