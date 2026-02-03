"""
VQE Algorithms Module

Contains all VQE algorithm implementations.
"""
from .vqe_vanilla import VanillaVQE
from .vqe_adapt import AdaptVQE
from .vqe_hardware_efficient import HardwareEfficientVQE
from .vqe_qaoa_inspired import QAOAInspiredVQE

# Registry of all available algorithms
ALGORITHMS = {
    "vanilla_vqe": VanillaVQE,
    "adapt_vqe": AdaptVQE,
    "hardware_efficient_vqe": HardwareEfficientVQE,
    "qaoa_inspired_vqe": QAOAInspiredVQE,
}

def get_algorithm(name: str):
    """Get algorithm class by name"""
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[name]

def list_algorithms():
    """List all available algorithm names"""
    return list(ALGORITHMS.keys())

__all__ = [
    "VanillaVQE",
    "AdaptVQE", 
    "HardwareEfficientVQE",
    "QAOAInspiredVQE",
    "ALGORITHMS",
    "get_algorithm",
    "list_algorithms",
]
