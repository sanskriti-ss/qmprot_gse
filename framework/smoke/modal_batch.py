# ABOUTME: Runs the trainability smoke batch on a CPU in Modal.
# ABOUTME: Persists artifacts in a Volume and checks the retrieved files locally.
from pathlib import Path
from uuid import uuid4

import modal

from smoke.batch import FRAMEWORK, run, validate

app = modal.App("qmprot-trainability-smoke")
volume = modal.Volume.from_name("qmprot-smoke-results", create_if_missing=True)


def add_framework_source(image, source, remote):
    image = image.add_local_file(source / "config.py", f"{remote}/config.py")
    driver = source.parent / "run_barren_plateau_part_b_all_molecules.py"
    if driver.is_file():
        image = image.add_local_file(driver, f"{remote}/{driver.name}")
    for folder in ("active_space_truncation", "algorithms", "contextual_subspace", "core", "experiments",
                   "visualize_molecules"):
        image = image.add_local_dir(source / folder, f"{remote}/{folder}",
                                    ignore=lambda path: len(path.parts) != 1 or path.suffix != ".py")
    return image


image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgomp1")
         .pip_install_from_requirements(str(FRAMEWORK / "smoke/requirements.txt")))
image = add_framework_source(image, FRAMEWORK, "/root")
image = image.add_local_dir(FRAMEWORK / "smoke", "/root/smoke",
                            ignore=["**/__pycache__/**", "**/*.pyc"])


@app.function(image=image, cpu=2, memory=4096, timeout=900, volumes={"/outputs": volume})
def calculate(run_id: str):
    directory = Path("/outputs") / run_id
    try:
        run(directory)
    finally:
        volume.commit()
    return [p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()]


@app.local_entrypoint()
def main():
    run_id = uuid4().hex
    destination = FRAMEWORK / "results/smoke" / run_id
    print(f"Modal Volume: qmprot-smoke-results/{run_id}", flush=True)
    files = calculate.remote(run_id)
    destination.mkdir(parents=True, exist_ok=False)
    for name in files:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            for chunk in volume.read_file(f"{run_id}/{name}"):
                stream.write(chunk)
    validate(destination)
    print(f"PASS: four trainability results retrieved and verified in {destination}")
