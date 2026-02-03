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
        
        # Extract summary data
        rows = []
        for r in self.results:
            rows.append({
                "molecule_abbrev": r.molecule_abbrev,
                "molecule_name": r.molecule_name,
                "algorithm": r.algorithm_name,
                "calculated_energy": r.calculated_energy,
                "reference_energy": r.reference_energy,
                "error": r.error,
                "relative_error": r.relative_error,
                "n_iterations": r.n_iterations,
                "n_qubits": r.n_qubits,
                "n_parameters": r.n_parameters,
                "runtime_seconds": r.runtime_seconds,
                "converged": r.converged,
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved CSV summary to {filepath}")
        return filepath
    
    def load_results(self, filepath: Union[str, Path]) -> List[VQEResult]:
        """
        Load results from a JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of VQEResult objects
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
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
            )
            results.append(result)
        
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
        return pd.DataFrame(rows)
    
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
            "convergence_rate": df["converged"].mean(),
            "best_algorithm_by_error": df.groupby("algorithm_name")["error"].mean().abs().idxmin(),
            "best_algorithm_by_runtime": df.groupby("algorithm_name")["runtime_seconds"].mean().idxmin(),
        }
    
    def clear(self):
        """Clear all results from memory"""
        self.results = []
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __repr__(self) -> str:
        return f"ResultsManager(n_results={len(self.results)}, dir={self.results_dir})"
