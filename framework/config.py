"""
VQE Framework Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
FRAMEWORK_DIR = Path(__file__).parent.absolute()
DATA_DIR = FRAMEWORK_DIR / "data"
# Default to datasets/ for .h5 files, use data/hamiltonians for legacy .txt files
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", FRAMEWORK_DIR / "datasets"))
HAMILTONIANS_DIR = Path(os.getenv("HAMILTONIANS_DIR", DATA_DIR / "hamiltonians"))
MOLECULES_JSON = Path(os.getenv("MOLECULES_JSON", DATA_DIR / "qmprot.json"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", FRAMEWORK_DIR / "results"))
PLOTS_DIR = Path(os.getenv("PLOTS_DIR", FRAMEWORK_DIR / "plots"))
LOGS_DIR = FRAMEWORK_DIR / "logs"

# Hamiltonian mode: 'h5' (default) or 'legacy' (txt files)
HAMILTONIAN_MODE = os.getenv("HAMILTONIAN_MODE", "h5")

# Create directories if they don't exist
for directory in [DATA_DIR, DATASETS_DIR, HAMILTONIANS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# VQE Settings
DEFAULT_OPTIMIZER = os.getenv("DEFAULT_OPTIMIZER", "COBYLA")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 1000))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", 1e-6))
N_SHOTS = int(os.getenv("N_SHOTS", 0))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Hamiltonian Truncation Settings (for large .h5 datasets)
HAMILTONIAN_MAX_TERMS = int(os.getenv("HAMILTONIAN_MAX_TERMS", 1000))
HAMILTONIAN_TARGET_QUBITS = int(os.getenv("HAMILTONIAN_TARGET_QUBITS", 8))

# Backend Settings
BACKEND = os.getenv("BACKEND", "pennylane")
PENNYLANE_DEVICE = os.getenv("PENNYLANE_DEVICE", "default.qubit")
QISKIT_BACKEND = os.getenv("QISKIT_BACKEND", "aer_simulator_statevector")

# Noise Settings
BACKEND_TYPE = os.getenv("BACKEND_TYPE", "statevector")  # "statevector" or "noisy"
NOISE_MODEL = os.getenv("NOISE_MODEL", "depolarizing")  # depolarizing, bitflip, phaseflip, amplitude_damping, phase_damping
NOISE_STRENGTH = float(os.getenv("NOISE_STRENGTH", 0.01))

# Plotting Settings
PLOT_FORMAT = os.getenv("PLOT_FORMAT", "png")
PLOT_DPI = int(os.getenv("PLOT_DPI", 150))
COLOR_SCHEME = os.getenv("COLOR_SCHEME", "default")

# Algorithm color mapping for plots
ALGORITHM_COLORS = {
    "vanilla_vqe": "#1f77b4",      # Blue
    "adapt_vqe": "#ff7f0e",         # Orange
    "hardware_efficient_vqe": "#2ca02c",  # Green
    "qaoa_inspired_vqe": "#d62728",  # Red
    "cb_vqe": "#9467bd",             # Purple
    "reference": "#7f7f7f",          # Gray
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "vqe_framework.log"))

# Parallel Execution
N_WORKERS = int(os.getenv("N_WORKERS", 1))
ENABLE_MULTIPROCESSING = os.getenv("ENABLE_MULTIPROCESSING", "false").lower() == "true"

# Supported optimizers
SUPPORTED_OPTIMIZERS = [
    "COBYLA",
    "L-BFGS-B", 
    "SLSQP",
    "SPSA",
    "ADAM",
    "GradientDescent",
    "NelderMead",
]

# Supported backends
SUPPORTED_BACKENDS = ["pennylane", "qiskit"]
