# ABOUTME: Runs the frozen glycine clean and depolarizing study on a single Modal CPU worker.
# ABOUTME: Downloads plots, complete optimizer events and run settings to verify the actual worker and source.
import json
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import modal

from smoke.modal_representative import image, retrieve, volume

app = modal.App("qmprot-noise-trial")


@app.function(image=image, cpu=2, memory=4096, timeout=3600, retries=0, volumes={"/outputs": volume})
def calculate(run_id):
    root = Path("/outputs")
    directory = root / run_id
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg", PYTHONPATH="/root")
    try:
        with (root / (run_id + ".log")).open("x") as log:
            start = perf_counter()
            subprocess.run([sys.executable, "-m", "smoke.noise_trial", str(directory)],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            elapsed = perf_counter() - start
        shutil.copyfile(root / (run_id + ".log"), directory / "native.log")
        (directory / "process.json").write_text(json.dumps(dict(remote_process_seconds=elapsed)) + "\n")
        archive = shutil.make_archive(str(directory), "tar", root_dir=directory)
    finally:
        volume.commit()
    return Path(archive).name


@app.local_entrypoint()
def main():
    from experiments.provenance import framework_source
    from smoke.noise_trial import validate
    scientific_source = sha256(framework_source()).hexdigest()
    run_id = "noise-trial-" + uuid4().hex
    print(f"Volume qmprot-smoke-results/{run_id}", flush=True)
    directory = retrieve(run_id, calculate.remote(run_id))
    print(f"Verified {directory}\n{json.dumps(validate(directory, scientific_source), indent=2)}", flush=True)
