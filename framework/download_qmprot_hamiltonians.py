import pennylane as qml
from tqdm import tqdm
import os

def download_hamiltonian(max_terms=1000, save_path="data/hamiltonians/hamiltonian_gln.txt"):
    coefficients = []
    operators = []
    hamiltonian_chunks = []

    ds = qml.data.load('other', name='gln')
    print(f"Dataset type: {type(ds)}")
    print(f"Dataset content: {ds}")

    # If ds is a list, iterate through it to find the actual dataset object
    if isinstance(ds, list):
        dataset = ds[0] if len(ds) > 0 else None
        if dataset is not None:
            print(f"First item type: {type(dataset)}")
            if hasattr(dataset, 'list_attributes'):
                for key in dataset.list_attributes():
                    if "hamiltonian" in key:
                        hamiltonian_chunks.append(getattr(dataset, key))
            elif hasattr(dataset, '__dict__'):
                for key in dir(dataset):
                    if "hamiltonian" in key and not key.startswith('_'):
                        hamiltonian_chunks.append(getattr(dataset, key))
    else:
        for key in ds.list_attributes():
            if "hamiltonian" in key:
                hamiltonian_chunks.append(getattr(ds, key))

    if hamiltonian_chunks:
        full_hamiltonian = "".join(hamiltonian_chunks)
        print('successfully combined')

        lines = full_hamiltonian.split("\n")
        valid_lines = [line.strip() for line in lines if line.strip() and "Coefficient" not in line and "Operators" not in line]

        if len(valid_lines) > max_terms:
            print(f'Found {len(valid_lines)} terms, limiting to first {max_terms} for faster processing')
            valid_lines = valid_lines[:max_terms]
        else:
            print(f'Processing all {len(valid_lines)} Hamiltonian terms...')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("Coefficient\tOperators\n")
            for line in tqdm(valid_lines, desc="Saving Hamiltonian", unit="terms"):
                parts = line.split()
                try:
                    coeff = float(parts[0])
                    op_string = " ".join(parts[1:])
                    f.write(f"{coeff}\t{op_string}\n")
                except ValueError:
                    continue
        print(f'Saved first {len(valid_lines)} terms of Alanine Hamiltonian to {save_path}')
    else:
        print("No hamiltonian chunks found. Please check the dataset structure.")

if __name__ == "__main__":
    download_hamiltonian()