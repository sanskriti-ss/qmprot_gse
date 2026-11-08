# ABOUTME: Times glycine training and hydrogen prefix, initialization and noise studies on one Modal worker.
# ABOUTME: Retains both versions' native outputs and verifies their scientific results after retrieval.
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import modal

from smoke.efficiency import prefixes, trainability
from campaigns.run_campaign import argv_from_args
from smoke.deep_studies import compare_readouts, study_files
from smoke.modal_representative import image as paired_image, retrieve, volume
from smoke.representative import specification
from smoke.solver_efficiency import solver_studies


app = modal.App("qmprot-exact-efficiency")
image = paired_image
if modal.is_local():
    source = Path(os.environ["QMPROT_BASELINE_FRAMEWORK"])
    image = image.add_local_file(source / "campaigns/run_campaign.py", "/baseline/campaigns/run_campaign.py")


@app.function(image=image, cpu=2, memory=4096, timeout=2400, retries=0, volumes={"/outputs": volume})
def calculate(run_id, deep=False):
    directory = Path("/outputs") / run_id
    directory.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg", QMPROT_GEOMETRIES="/root/smoke/geometries",
               QMPROT_WORKLOAD="gly-full-budget")
    try:
        for version, source_root in (("baseline", "/baseline"), ("candidate", "/root")):
            target = directory / f"{version}-trainability"
            command = [sys.executable, "/root/smoke/record.py", "experiments.trainability",
                       str(directory / f"{version}-events.jsonl"), "--molecules", "gly",
                       "--algorithms", "qubit_adapt_vqe", "hardware_efficient_vqe", "--param-targets",
                       *(("10", "50", "100") if deep else ("10",)),
                       "--eval-budget", "2000", "--max-iter", "500", "--cs-target-qubits", "0",
                       "--seed", "42", "--output-root", str(target)]
            prefix = [sys.executable, "-m", "experiments.experiment5_term_count_sweep", "--molecules", "hydrogen",
                      "--prepared-inputs", "/root/smoke/reproductions/layered_vqe", "--term-counts", "15",
                      "--seeds", "42", "43", "--n-layers", "1", "--max-operators", "2", "--max-iter", "24",
                      "--cs-target-qubits", "0", "--output-dir", str(directory / f"{version}-prefix")]
            solver = [sys.executable, "-m", "smoke.solver_studies", str(directory / f"{version}-solver")]
            studies = [("trainability", command), ("prefix", prefix), ("solver", solver)]
            if deep:
                _, _, noise_module, *noise_args = argv_from_args(specification("noise", "full"))
                studies.extend([
                    ("readouts", [sys.executable, "/root/smoke/deep_studies.py", str(directory / f"{version}-readouts")]),
                    ("noise", [sys.executable, "/root/smoke/record.py", noise_module,
                               str(directory / f"{version}-noise-events.jsonl"), *noise_args,
                               "--output-dir", str(directory / f"{version}-noise")]),
                ])
            for name, args in studies:
                with (directory / f"{version}-{name}.log").open("x") as log:
                    start = perf_counter()
                    subprocess.run(args, cwd=directory, env={**env, "PYTHONPATH": f"{source_root}:/root"},
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                    print(f"real {perf_counter() - start:.6f}", file=log)
        archive = shutil.make_archive(str(directory), "tar", root_dir=directory)
    finally:
        volume.commit()
    return Path(archive).name


@app.local_entrypoint()
def main(deep: bool = False):
    run_id = "exact-efficiency-" + uuid4().hex
    print(f"Modal Volume qmprot-smoke-results/{run_id}", flush=True)
    directory = retrieve(run_id, calculate.remote(run_id, deep))
    report = trainability(directory)
    (directory / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    matrix = dict(prefix=prefixes(directory), solver=solver_studies(directory))
    for study in matrix.values():
        if study["source_sha256"] != report["source_sha256"]:
            raise ValueError("Scientific source differs between paired study families")
    (directory / "matrix-comparison.json").write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n")
    if deep:
        controls = dict(readouts=compare_readouts(directory), noise=study_files(directory, "noise"))
        if controls["noise"]["source_sha256"] != report["source_sha256"]:
            raise ValueError("Noise study used different sources")
        (directory / "deep-comparison.json").write_text(json.dumps(controls, indent=2, allow_nan=False) + "\n")
    print(f"Verified {directory} {report} {matrix}", flush=True)
