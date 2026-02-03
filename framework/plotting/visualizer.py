"""
VQE Visualization Module

Comprehensive plotting functions for VQE results comparison.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

import sys
sys.path.append('..')
from core.base_vqe import VQEResult
from core.results_manager import ResultsManager

logger = logging.getLogger(__name__)

# Default color palette for algorithms
DEFAULT_COLORS = {
    "vanilla_vqe": "#1f77b4",
    "adapt_vqe": "#ff7f0e",
    "hardware_efficient_vqe": "#2ca02c",
    "qaoa_inspired_vqe": "#d62728",
    "reference": "#7f7f7f",
}


class VQEVisualizer:
    """
    Visualization class for VQE results.
    
    Generates various plots comparing VQE algorithm performance.
    """
    
    def __init__(self,
                 results_manager: ResultsManager,
                 output_dir: Optional[Union[str, Path]] = None,
                 color_map: Optional[Dict[str, str]] = None,
                 style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize the visualizer.
        
        Args:
            results_manager: ResultsManager with loaded results
            output_dir: Directory to save plots
            color_map: Custom color mapping for algorithms
            style: Matplotlib style
        """
        self.results_manager = results_manager
        self.output_dir = Path(output_dir) if output_dir else Path("./plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = color_map or DEFAULT_COLORS
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')
    
    def _get_color(self, algorithm: str) -> str:
        """Get color for algorithm"""
        return self.colors.get(algorithm, "#333333")
    
    def plot_molecule_comparison(self,
                                  molecule_abbrev: str,
                                  show_reference: bool = True,
                                  save: bool = True,
                                  figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot energy comparison for a single molecule across all algorithms.
        
        Args:
            molecule_abbrev: Molecule abbreviation
            show_reference: Whether to show reference energy line
            save: Whether to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        results = self.results_manager.get_results_by_molecule(molecule_abbrev)
        
        if not results:
            logger.warning(f"No results found for molecule: {molecule_abbrev}")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        algorithms = []
        energies = []
        colors = []
        
        for r in results:
            algorithms.append(r.algorithm_name)
            energies.append(r.calculated_energy)
            colors.append(self._get_color(r.algorithm_name))
        
        # Create bar plot
        bars = ax.bar(algorithms, energies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add reference line
        if show_reference and results:
            ref_energy = results[0].reference_energy
            ax.axhline(y=ref_energy, color=self.colors.get("reference", "gray"),
                      linestyle='--', linewidth=2, label=f'Reference: {ref_energy:.4f} Ha')
        
        # Add value labels on bars
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax.annotate(f'{energy:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('VQE Algorithm', fontsize=12)
        ax.set_ylabel('Energy (Hartree)', fontsize=12)
        ax.set_title(f'VQE Energy Comparison: {results[0].molecule_name.title()}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"molecule_{molecule_abbrev}_comparison.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_algorithm_comparison(self,
                                   algorithm_name: str,
                                   save: bool = True,
                                   figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot results for a single algorithm across all molecules.
        
        Args:
            algorithm_name: Name of the algorithm
            save: Whether to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        results = self.results_manager.get_results_by_algorithm(algorithm_name)
        
        if not results:
            logger.warning(f"No results found for algorithm: {algorithm_name}")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        molecules = [r.molecule_abbrev for r in results]
        calc_energies = [r.calculated_energy for r in results]
        ref_energies = [r.reference_energy for r in results]
        errors = [r.error for r in results]
        
        color = self._get_color(algorithm_name)
        
        # Plot 1: Calculated vs Reference
        x = np.arange(len(molecules))
        width = 0.35
        
        ax1.bar(x - width/2, calc_energies, width, label='Calculated', color=color, alpha=0.8)
        ax1.bar(x + width/2, ref_energies, width, label='Reference', color='gray', alpha=0.6)
        
        ax1.set_xlabel('Molecule', fontsize=12)
        ax1.set_ylabel('Energy (Hartree)', fontsize=12)
        ax1.set_title(f'{algorithm_name}: Calculated vs Reference', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(molecules, rotation=45, ha='right')
        ax1.legend()
        
        # Plot 2: Errors
        error_colors = ['red' if e > 0 else 'blue' for e in errors]
        ax2.bar(molecules, errors, color=error_colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax2.set_xlabel('Molecule', fontsize=12)
        ax2.set_ylabel('Error (Hartree)', fontsize=12)
        ax2.set_title(f'{algorithm_name}: Energy Errors', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"algorithm_{algorithm_name}_comparison.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_convergence(self,
                         results: Optional[List[VQEResult]] = None,
                         molecule_abbrev: Optional[str] = None,
                         save: bool = True,
                         figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot convergence history for VQE runs.
        
        Args:
            results: List of VQEResults (or uses all if None)
            molecule_abbrev: Filter by molecule
            save: Whether to save plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if results is None:
            if molecule_abbrev:
                results = self.results_manager.get_results_by_molecule(molecule_abbrev)
            else:
                results = self.results_manager.results
        
        if not results:
            logger.warning("No results to plot")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for r in results:
            if r.convergence_history:
                color = self._get_color(r.algorithm_name)
                label = f"{r.algorithm_name} ({r.molecule_abbrev})"
                ax.plot(r.convergence_history, color=color, label=label, linewidth=1.5)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Energy (Hartree)', fontsize=12)
        ax.set_title('VQE Convergence History', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            suffix = f"_{molecule_abbrev}" if molecule_abbrev else ""
            filepath = self.output_dir / f"convergence{suffix}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_heatmap(self,
                     metric: str = "error",
                     save: bool = True,
                     figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot heatmap of algorithm performance across molecules.
        
        Args:
            metric: Metric to display (error, relative_error, runtime_seconds)
            save: Whether to save plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        df = self.results_manager.to_dataframe()
        
        if df.empty:
            logger.warning("No results to plot")
            return None
        
        # Pivot to create heatmap data
        pivot_df = df.pivot_table(
            index='molecule_abbrev',
            columns='algorithm_name',
            values=metric,
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose colormap based on metric
        if metric in ['error', 'relative_error']:
            cmap = 'RdYlGn_r'  # Red=bad, Green=good (reversed for errors)
            center = 0
        else:
            cmap = 'viridis'
            center = None
        
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap=cmap,
                   center=center, ax=ax, cbar_kws={'label': metric})
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Molecule', fontsize=12)
        ax.set_title(f'VQE Performance Heatmap: {metric}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"heatmap_{metric}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_all_molecules(self, save: bool = True, figsize: tuple = (14, 8)) -> plt.Figure:
        """
        Plot comprehensive comparison of all molecules and algorithms.
        
        Args:
            save: Whether to save plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        df = self.results_manager.to_dataframe()
        
        if df.empty:
            logger.warning("No results to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Energy by molecule and algorithm
        ax1 = axes[0, 0]
        molecules = df['molecule_abbrev'].unique()
        algorithms = df['algorithm_name'].unique()
        
        x = np.arange(len(molecules))
        width = 0.8 / len(algorithms)
        
        for i, alg in enumerate(algorithms):
            alg_data = df[df['algorithm_name'] == alg]
            energies = [alg_data[alg_data['molecule_abbrev'] == m]['calculated_energy'].values[0]
                       if m in alg_data['molecule_abbrev'].values else np.nan
                       for m in molecules]
            ax1.bar(x + i * width, energies, width, label=alg, color=self._get_color(alg))
        
        ax1.set_xlabel('Molecule')
        ax1.set_ylabel('Energy (Hartree)')
        ax1.set_title('Calculated Energies')
        ax1.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax1.set_xticklabels(molecules, rotation=45, ha='right')
        ax1.legend(fontsize=8)
        
        # Plot 2: Error distribution
        ax2 = axes[0, 1]
        for alg in algorithms:
            alg_data = df[df['algorithm_name'] == alg]
            ax2.scatter(alg_data['molecule_abbrev'], alg_data['error'],
                       label=alg, color=self._get_color(alg), s=100, alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Molecule')
        ax2.set_ylabel('Error (Hartree)')
        ax2.set_title('Energy Errors')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(fontsize=8)
        
        # Plot 3: Runtime comparison
        ax3 = axes[1, 0]
        for alg in algorithms:
            alg_data = df[df['algorithm_name'] == alg]
            ax3.bar(alg_data['molecule_abbrev'], alg_data['runtime_seconds'],
                   label=alg, color=self._get_color(alg), alpha=0.7)
        
        ax3.set_xlabel('Molecule')
        ax3.set_ylabel('Runtime (seconds)')
        ax3.set_title('Computation Time')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(fontsize=8)
        
        # Plot 4: Algorithm summary (mean error and runtime)
        ax4 = axes[1, 1]
        alg_summary = df.groupby('algorithm_name').agg({
            'error': lambda x: np.abs(x).mean(),
            'runtime_seconds': 'mean'
        }).reset_index()
        
        x = np.arange(len(alg_summary))
        
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x - 0.2, alg_summary['error'], 0.4,
                       label='Mean |Error|', color='coral', alpha=0.8)
        bars2 = ax4_twin.bar(x + 0.2, alg_summary['runtime_seconds'], 0.4,
                            label='Mean Runtime', color='steelblue', alpha=0.8)
        
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Mean |Error| (Hartree)', color='coral')
        ax4_twin.set_ylabel('Mean Runtime (s)', color='steelblue')
        ax4.set_title('Algorithm Performance Summary')
        ax4.set_xticks(x)
        ax4.set_xticklabels(alg_summary['algorithm_name'], rotation=15, ha='right')
        
        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "comprehensive_comparison.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def generate_all_plots(self):
        """Generate all available plots"""
        logger.info("Generating all plots...")
        
        # Get unique molecules and algorithms
        df = self.results_manager.to_dataframe()
        
        if df.empty:
            logger.warning("No results to plot")
            return
        
        molecules = df['molecule_abbrev'].unique()
        algorithms = df['algorithm_name'].unique()
        
        # Per-molecule plots
        for mol in molecules:
            self.plot_molecule_comparison(mol, save=True)
        
        # Per-algorithm plots
        for alg in algorithms:
            self.plot_algorithm_comparison(alg, save=True)
        
        # Convergence plots
        for mol in molecules:
            self.plot_convergence(molecule_abbrev=mol, save=True)
        
        # Heatmaps
        self.plot_heatmap(metric='error', save=True)
        self.plot_heatmap(metric='relative_error', save=True)
        self.plot_heatmap(metric='runtime_seconds', save=True)
        
        # Comprehensive plot
        self.plot_all_molecules(save=True)
        
        logger.info(f"All plots saved to {self.output_dir}")


# Convenience functions
def plot_molecule_comparison(results: List[VQEResult],
                              molecule_abbrev: str,
                              output_path: Optional[str] = None) -> plt.Figure:
    """Quick function to plot molecule comparison"""
    rm = ResultsManager("./temp_results")
    rm.add_results(results)
    viz = VQEVisualizer(rm)
    fig = viz.plot_molecule_comparison(molecule_abbrev, save=output_path is not None)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_algorithm_comparison(results: List[VQEResult],
                               algorithm_name: str,
                               output_path: Optional[str] = None) -> plt.Figure:
    """Quick function to plot algorithm comparison"""
    rm = ResultsManager("./temp_results")
    rm.add_results(results)
    viz = VQEVisualizer(rm)
    fig = viz.plot_algorithm_comparison(algorithm_name, save=output_path is not None)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig
