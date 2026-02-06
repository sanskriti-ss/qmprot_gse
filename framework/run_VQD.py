# From the framework directory
from core import HamiltonianLoader, VQD, SSVQE, compute_first_excited_state
from config import DATASETS_DIR, MOLECULES_JSON

# Load a Hamiltonian (using datasets/ directory where H5 files are stored)
loader = HamiltonianLoader(DATASETS_DIR, MOLECULES_JSON)
H = loader.load_hamiltonian(molecule_abbrev="ala")  # or any molecule

# Run VQD (finds ground + first excited state)
# Increase max_iterations and relax convergence for better optimization
vqd = VQD(
    H, 
    n_states=2, 
    beta=20.0, 
    n_layers=2, 
    max_iterations=200, 
    convergence_threshold=1e-5
)
results = vqd.run()

# Access results
print(f"\n{'='*60}")
print(f"VQD Results for Alanine (ala)")
print(f"{'='*60}")
print(f"Ground state:     {results.ground_state.energy:.6f} Hartree")
print(f"1st excited:      {results.first_excited_energy:.6f} Hartree")
print(f"Energy gap:       {results.first_gap:.6f} Hartree")
print(f"\n NOTE: Gap should be POSITIVE!")
print(f"   If negative, increase beta parameter (try beta=20.0)")
print(f"{'='*60}")

# Additional diagnostics
print(f"\nOptimization Details:")
print(f"  Ground state iterations:  {results.ground_state.n_iterations}")
print(f"  Excited state iterations: {results.excited_states[0].n_iterations}")
print(f"  Ground state converged:   {results.ground_state.converged}")
print(f"  Excited state converged:  {results.excited_states[0].converged}")


SSVQE = SSVQE(
    H,
    n_states=2, 
    n_layers=2,
    max_iterations=200, 
    convergence_threshold=1e-5
)

ssvqe_results = SSVQE.run()
print(f"\n{'='*60}")
print(f"SSVQE Results for Alanine (ala)")
print(f"{'='*60}")
print(f"Ground state:     {ssvqe_results.ground_state.energy:.6f} Hartree")
print(f"1st excited:      {ssvqe_results.first_excited_energy:.6f} Hartree")
print(f"Energy gap:       {ssvqe_results.first_gap:.6f} Hartree")
print(f"\n NOTE: Gap should be POSITIVE!")
print(f"   If negative, increase max_iterations or relax convergence threshold")
print(f"{'='*60}")
