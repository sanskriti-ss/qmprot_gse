#!/usr/bin/env bash
# Train the ten non-HVA solvers on the 19 pinned whole amino acids in WSL.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir/framework"
python="$PWD/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  printf 'WSL Python not found at %s. Set up framework/.venv first.\n' "$python" >&2
  exit 1
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
run="$PWD/results/amino19_14q_no_hva_$(date +%Y%m%d_%H%M%S)"

# Seven spatial orbitals give 14 qubits. Override the cap in this process only.
exec "$python" -c '
from active_space_truncation import step1_orbitals
from main import main

step1_orbitals.MAX_ACTIVE_ORBITALS = 7
main()
' --all \
  --hamiltonians-dir "$PWD/smoke/geometries" \
  --molecules ala arg asp cys gln glu gly his ile leu lys met phe pro ser thr trp tyr val \
  --algorithms pennylane_vqe adapt_vqe qubit_adapt_vqe hardware_efficient_vqe \
    qaoa_inspired_vqe iqcc_inspired_vqe iqcc_true_vqe cb_vqe ucc_vqe nn_ae_vqe \
  --truncation-mode active_space --active-space-basis sto-3g \
  --backend-type statevector --n-shots 0 \
  --optimizer COBYLA --n-layers 2 --max-iterations 200 --seed 42 \
  --results-dir "$run/results" --plots-dir "$run/plots" --no-plot
