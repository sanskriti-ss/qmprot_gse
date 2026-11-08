# ABOUTME: Runs paired production Bayesian glycine fits on a single Modal worker.
# ABOUTME: Retrieves both scientific snapshots, the frozen input, forest predictions and complete optimizer records.
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import modal

from smoke.forest_trial import compare
from smoke.modal_representative import image, retrieve, volume

app = modal.App("qmprot-forest-trial")
if modal.is_local():
    baseline = Path(os.environ["QMPROT_BASELINE_FRAMEWORK"])
    image = image.add_local_file(baseline / "campaigns/run_campaign.py", "/baseline/campaigns/run_campaign.py")
    image = image.add_local_file(Path(os.environ["QMPROT_FOREST_INPUT"]), "/root/pinned-gly.npz")


@app.function(image=image, cpu=2, memory=4096, timeout=1200, retries=0, volumes={"/outputs": volume})
def calculate(run_id):
    directory = Path("/outputs") / run_id
    directory.mkdir(exist_ok=False)
    shutil.copyfile("/root/pinned-gly.npz", directory / "gly.npz")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg")
    try:
        for version, root in (("baseline", "/baseline"), ("candidate", "/root")):
            argv = [sys.executable, "/root/smoke/forest_trial.py", str(directory / version),
                    "--input", "/root/pinned-gly.npz"]
            with (directory / (version + ".log")).open("x") as log:
                subprocess.run(argv, cwd=directory, env={**env, "PYTHONPATH": root},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        archive = shutil.make_archive(str(directory), "tar", root_dir=directory)
    finally:
        volume.commit()
    return Path(archive).name


@app.local_entrypoint()
def main():
    run_id = "forest-trial-" + uuid4().hex
    print(f"Volume qmprot-smoke-results/{run_id}", flush=True)
    directory = retrieve(run_id, calculate.remote(run_id))
    print(f"Verified {directory}\n{json.dumps(compare(directory), indent=2)}", flush=True)
