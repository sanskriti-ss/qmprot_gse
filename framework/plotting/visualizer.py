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
from datetime import datetime

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
        base_output_dir = Path(output_dir) if output_dir else Path("./plots")
        
        # Create timestamped subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = base_output_dir / timestamp
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
    
    def plot_molecular_complexity_heatmap(self, save: bool = True, figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot heatmap of molecular complexity (n_qubits, n_parameters) across molecules and algorithms.
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

        # Pivot for n_qubits
        qubits_pivot = df.pivot_table(index='molecule_abbrev', columns='algorithm_name', values='n_qubits', aggfunc='mean')
        # Pivot for n_parameters
        params_pivot = df.pivot_table(index='molecule_abbrev', columns='algorithm_name', values='n_parameters', aggfunc='mean')

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        sns.heatmap(qubits_pivot, annot=True, fmt='.0f', cmap='Blues', ax=axes[0], cbar_kws={'label': 'n_qubits'})
        axes[0].set_title('Qubit Complexity (n_qubits)')
        axes[0].set_xlabel('Algorithm')
        axes[0].set_ylabel('Molecule')

        sns.heatmap(params_pivot, annot=True, fmt='.0f', cmap='Purples', ax=axes[1], cbar_kws={'label': 'n_parameters'})
        axes[1].set_title('Ansatz Complexity (n_parameters)')
        axes[1].set_xlabel('Algorithm')
        axes[1].set_ylabel('Molecule')

        plt.tight_layout()
        if save:
            filepath = self.output_dir / "heatmap_molecular_complexity.png"
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

    def plot_selected_molecules(self,
                                molecules: List[str],
                                algorithms: Optional[List[str]] = None,
                                show_reference: bool = True,
                                save: bool = True,
                                figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot a single grouped bar chart for a selected list of molecules.

        Args:
            molecules: List of molecule abbreviations to include
            algorithms: Optional list of algorithms to include (all if None)
            show_reference: Whether to plot reference energies per molecule
            save: Whether to save the figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        df = self.results_manager.to_dataframe()

        if df.empty:
            logger.warning("No results to plot")
            return None

        # Filter molecules
        df = df[df['molecule_abbrev'].isin(molecules)]
        if df.empty:
            logger.warning("No matching molecules found in results")
            return None

        if algorithms is None:
            algorithms = df['algorithm_name'].unique().tolist()

        fig, ax = plt.subplots(figsize=figsize)

        # Color cycle for VQE algorithms: dark green, lemon green, dark purple, dark blue
        vqe_colors = ['#006400', '#9ACD32', '#483D8B', '#00008B']
        
        x = np.arange(len(molecules))
        n_algorithms = len(algorithms)
        bar_width = 0.8 / (n_algorithms + 1)  # +1 for reference bars

        # Get reference energies (choose non-zero if available)
        refs = []
        for m in molecules:
            vals = df[df['molecule_abbrev'] == m]['reference_energy'].dropna().values
            # Prefer a non-zero reference (some saved results may have 0.0 placeholders)
            nonzero = [v for v in vals if abs(v) > 1e-12]
            if len(nonzero):
                refs.append(float(nonzero[0]))
            elif len(vals):
                refs.append(float(vals[0]))
            else:
                refs.append(np.nan)

        # Plot reference bars first
        if show_reference:
            ax.bar(x - bar_width * n_algorithms / 2, refs, bar_width, 
                  label='Reference', color='purple', alpha=0.7)

        # Plot VQE algorithm bars
        for i, alg in enumerate(algorithms):
            alg_data = df[df['algorithm_name'] == alg]
            calc_energies = []
            
            for m in molecules:
                mol_data = alg_data[alg_data['molecule_abbrev'] == m]
                if len(mol_data) > 0:
                    # If multiple results, prefer one with non-zero reference energy
                    valid_results = mol_data[mol_data['reference_energy'].abs() > 1e-12]
                    if len(valid_results) > 0:
                        # Use the last (most recent) valid result
                        calc_energies.append(float(valid_results.iloc[-1]['calculated_energy']))
                    else:
                        # Fall back to last result even if reference is zero
                        calc_energies.append(float(mol_data.iloc[-1]['calculated_energy']))
                else:
                    calc_energies.append(np.nan)
            
            # Position bars: reference on left, then algorithms
            bar_position = x - bar_width * n_algorithms / 2 + bar_width * (i + 1)
            color = vqe_colors[i % len(vqe_colors)]
            ax.bar(bar_position, calc_energies, bar_width, 
                  label=f'{alg}', color=color, alpha=0.8)

        # Auto-adjust y-limits with top at 0
        try:
            all_energies = []
            if show_reference:
                all_energies.extend([r for r in refs if not np.isnan(r)])
            
            # Collect all VQE energies
            for alg in algorithms:
                alg_data = df[df['algorithm_name'] == alg]
                for m in molecules:
                    mol_data = alg_data[alg_data['molecule_abbrev'] == m]
                    if len(mol_data) > 0:
                        valid_results = mol_data[mol_data['reference_energy'].abs() > 1e-12]
                        if len(valid_results) > 0:
                            all_energies.append(float(valid_results.iloc[-1]['calculated_energy']))
                        else:
                            all_energies.append(float(mol_data.iloc[-1]['calculated_energy']))
            
            if all_energies:
                ymin = min(all_energies)
                span = abs(ymin)
                pad = span * 0.1 if span > 0 else 10.0
                ax.set_ylim(ymin - pad, 0)
        except Exception:
            pass

        ax.set_xlabel('Molecule')
        ax.set_ylabel('Energy (Hartree)')
        ax.set_title('Selected Molecules: Reference vs VQE Algorithm Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(molecules, rotation=0)
        ax.legend(fontsize=9)

        plt.tight_layout()

        if save:
            fname = "selected_molecules_" + "_".join(molecules) + ".png"
            filepath = self.output_dir / fname
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")

        return fig
    
    def plot_computation_vs_error(self,
                                  molecules: Optional[List[str]] = None,
                                  algorithms: Optional[List[str]] = None,
                                  save: bool = True,
                                  figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot computational time vs error for all molecule/algorithm combinations.
        
        Args:
            molecules: List of molecules to include (all if None)
            algorithms: List of algorithms to include (all if None)
            save: Whether to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        df = self.results_manager.to_dataframe()
        
        if df.empty:
            logger.warning("No results to plot")
            return None
        
        if molecules is not None:
            df = df[df['molecule_abbrev'].isin(molecules)]
        if algorithms is not None:
            df = df[df['algorithm_name'].isin(algorithms)]
            
        if df.empty:
            logger.warning("No matching results found")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color cycle for different algorithms
        colors = ['#006400', '#9ACD32', '#483D8B', '#00008B', '#FF6347', '#32CD32']
        algorithm_colors = {}
        
        # Shape cycle for different molecules
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        molecule_markers = {}
        
        unique_algorithms = df['algorithm_name'].unique()
        unique_molecules = df['molecule_abbrev'].unique()
        
        for i, alg in enumerate(unique_algorithms):
            algorithm_colors[alg] = colors[i % len(colors)]
            
        for i, mol in enumerate(unique_molecules):
            molecule_markers[mol] = markers[i % len(markers)]
        
        # Plot each combination (keep only latest result for each molecule/algorithm pair)
        plotted_combinations = set()
        
        # Group by molecule and algorithm, keep latest result
        unique_results = df.groupby(['molecule_abbrev', 'algorithm_name']).last().reset_index()
        
        for _, row in unique_results.iterrows():
            alg = row['algorithm_name']
            mol = row['molecule_abbrev']
            runtime = row['runtime_seconds']
            error = abs(row['error'])  # Use absolute error
            
            color = algorithm_colors[alg]
            marker = molecule_markers[mol]
            
            # Create label for legend
            label = f"{alg} ({mol})"
            
            ax.scatter(runtime, error, color=color, marker=marker, s=80, alpha=0.8,
                      edgecolors='black', linewidth=0.5, label=label)
        
        ax.set_xlabel('Runtime (seconds)', fontsize=12)
        ax.set_ylabel('Absolute Error (Hartree)', fontsize=12)
        ax.set_title('VQE Performance: Computational Time vs Error', fontsize=14, fontweight='bold')
        ax.set_yscale('log')  # Log scale for better visualization of errors
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / "computation_vs_error.png"
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
        self.plot_molecular_complexity_heatmap(save=True)
        
        # Comprehensive plot
        self.plot_all_molecules(save=True)
        
        # Selected molecules plot (if multiple molecules)
        if len(molecules) > 1:
            self.plot_selected_molecules(molecules.tolist(), save=True)
        
        # Computation vs Error plot
        self.plot_computation_vs_error(save=True)
        
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
