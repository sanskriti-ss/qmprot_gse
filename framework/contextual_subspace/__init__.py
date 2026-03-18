"""
Contextual Subspace Reduction

Optional preprocessing step that reduces the qubit count of a
QubitHamiltonian before running any VQE algorithm.
"""

from .cs_reduction import apply_contextual_subspace_reduction

__all__ = ["apply_contextual_subspace_reduction"]
