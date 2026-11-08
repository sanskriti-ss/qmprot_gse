# ABOUTME: Runs the representative VQE workflows on a bounded Modal CPU worker.
# ABOUTME: Persists their outputs in a Volume and validates the downloaded artifacts.
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import modal

from smoke.batch import FRAMEWORK
from smoke.modal_batch import add_framework_source, image as smoke_image, volume

app = modal.App("qmprot-representative-smoke")
image = (smoke_image
         .add_local_file(FRAMEWORK / "campaigns/run_campaign.py", "/root/campaigns/run_campaign.py")
         .add_local_dir(FRAMEWORK / "campaigns/20261105", "/root/campaigns/20261105",
                        ignore=lambda path: len(path.parts) != 1 or path.suffix != ".json"
                        or path.name == "manifest.json"))
candidate = os.getenv("QMPROT_FRAMEWORK") if modal.is_local() else None
baseline_source = os.getenv("QMPROT_BASELINE_FRAMEWORK") if modal.is_local() else None
if candidate:
    image = add_framework_source(image, Path(candidate).resolve(), "/candidate")
if baseline_source:
    image = add_framework_source(image, Path(baseline_source).resolve(), "/baseline")


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0, volumes={"/outputs": volume})
def calculate(jobs: list[tuple[str, str]], profile: str):
    from smoke.representative import run

    outputs = []
    for run_id, source in jobs:
        os.environ["QMPROT_FRAMEWORK"] = source
        directory = Path("/outputs") / run_id
        start = time.monotonic()
        try:
            run(directory, profile=profile)
            elapsed = time.monotonic() - start
            archive = shutil.make_archive(str(directory), "tar", root_dir=directory)
        finally:
            volume.commit()
        outputs.append((run_id, Path(archive).name, elapsed))
    return outputs


def retrieve(run_id, archive):
    destination = FRAMEWORK / "results/smoke" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryFile() as stream:
        for chunk in volume.read_file(archive):
            stream.write(chunk)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as results:
            results.extractall(destination, filter="data")
    return destination


@app.local_entrypoint()
def main(reference: str = "", quick: bool = False, real_inputs: bool = False, tiny_full: bool = False,
         campaign: bool = False, full: bool = False, repeat: bool = False, pair: bool = False):
    from smoke.comparison import compare
    from smoke.representative import saved_profile, validate

    profiles = [name for name, enabled in (("tiny", quick), ("exact", real_inputs), ("tiny-full", tiny_full),
                                           ("campaign", campaign), ("full", full)) if enabled]
    if len(profiles) > 1:
        raise ValueError("Choose at most one of --quick, --real-inputs, --tiny-full, --campaign and --full")
    profile = profiles[0] if profiles else "tiny-grid"
    if pair and repeat:
        raise ValueError("Choose one of --pair and --repeat")
    if pair and not candidate:
        raise ValueError("--pair requires QMPROT_FRAMEWORK to select the candidate source")
    previous = Path(reference) if reference else None
    if previous:
        validate(previous)
        if saved_profile(previous) != profile:
            raise ValueError("Reference suite scope differs from the requested scope")
    source = "/candidate" if candidate else "/root"
    if pair:
        jobs = [(f"representative-baseline-{uuid4().hex}", "/baseline" if baseline_source else "/root"),
                (f"representative-candidate-{uuid4().hex}", source)]
    else:
        jobs = [(f"representative-{uuid4().hex}", source) for _ in range(2 if repeat else 1)]
    for run_id, _ in jobs:
        print(f"Modal Volume: qmprot-smoke-results/{run_id}", flush=True)
    start = time.monotonic()
    seconds = {"tiny": 180, "tiny-grid": 600, "exact": 600, "campaign": 21600, "full": 86400, "tiny-full": 86400}
    timeout = min(86400, seconds[profile] * len(jobs))
    outputs = calculate.with_options(timeout=timeout).remote(jobs, profile)
    if [item[0] for item in outputs] != [run_id for run_id, _ in jobs]:
        raise ValueError("Modal returned a different set of runs than requested")
    downloads = [(retrieve(run_id, archive), elapsed) for run_id, archive, elapsed in outputs]
    for destination, elapsed in downloads:
        validate(destination)
        print(f"Verified {destination}. Remote suite {elapsed:.1f} s.", flush=True)
    for destination, _ in downloads:
        if previous:
            print(compare(previous, destination), flush=True)
        previous = destination
    print(f"Call through local validation {time.monotonic() - start:.1f} s.", flush=True)
