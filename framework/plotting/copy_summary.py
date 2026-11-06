import shutil
from pathlib import Path

def copy_vqe_energies_summary_to_output(output_dir: str):
    """
    Copy the main vqe_energies_summary.csv to the current output_dir (timestamped folder).
    """
    src = Path(__file__).parent.parent / "plots" / "vqe_energies_summary.csv"
    dst = Path(output_dir) / "vqe_energies_summary.csv"
    if src.exists():
        shutil.copyfile(src, dst)
        print(f"Copied vqe_energies_summary.csv to {dst}")
    else:
        print(f"Source vqe_energies_summary.csv not found at {src}")
