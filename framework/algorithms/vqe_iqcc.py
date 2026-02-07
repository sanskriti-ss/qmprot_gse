'''
Iterative Qubit Coupled Cluster (iQCC) Implementation

Iteratively transforms Hamiltonian and aims to minimize circuit depth

'''

import numpy as np
from typing import Optional, Any
import logging

import sys
sys.path.append('..')
from core.base_vqe import BaseVQE, VQEResult
from core.hamiltonian_loader import QubitHamiltonian

logger = logging.getLogger(__name__)