# ABOUTME: Runs all five scientific studies through the native campaign dispatcher on real molecular inputs.
# ABOUTME: Checks each saved study against its case grid and the Hamiltonian it actually solved.
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from smoke.batch import FRAMEWORK


CONFIGURATION = FRAMEWORK / "campaigns/acceptance"
CONTROL_SOURCES = ("smoke/fermionic_adapt.py", "smoke/active_spaces.py", "smoke/physics_acceptance.py")


def run(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for spec in CONFIGURATION.glob("*.json"):
        shutil.copyfile(spec, destination / spec.name)
    env = dict(os.environ, PYTHONPATH=str(FRAMEWORK), QMPROT_GEOMETRIES=str(FRAMEWORK / "smoke/geometries"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    with (destination / "campaign.log").open("w") as log:
        process = subprocess.run([sys.executable, str(FRAMEWORK / "campaigns/run_campaign.py"), str(destination)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = (destination / "campaign.log").read_text().splitlines()[-30:]
        raise RuntimeError("Campaign execution failed\n" + "\n".join(tail))
    run_fermionic_control(destination, env)
    with (destination / "campaign.log").open("a") as log:
        subprocess.run([sys.executable, "-m", "smoke.iqcc_control", str(destination)], cwd=FRAMEWORK,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (destination / "solver-studies.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "smoke.solver_studies", str(destination / "solver-studies")],
                       cwd=FRAMEWORK, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (destination / "paired-studies.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "smoke.paired_studies", str(destination / "paired-studies")],
                       cwd=FRAMEWORK, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_fermionic_control(destination, env):
    start = perf_counter()
    with (destination / "campaign.log").open("a") as log:
        control = subprocess.run([sys.executable, "-m", "smoke.fermionic_adapt"], cwd=FRAMEWORK,
                                 env=env, stdout=subprocess.PIPE, stderr=log, check=True)
    result = json.loads(control.stdout)
    result["elapsed_seconds"] = perf_counter() - start
    archive = destination / "fermionic-control.zip"
    with ZipFile(archive, "w") as source:
        paths = [*CONTROL_SOURCES, *(f"smoke/geometries/{name}/{name}.h5" for name in ("hydrogen", "gly"))]
        for path in paths:
            source.writestr(ZipInfo(path), (FRAMEWORK / path).read_bytes(), compress_type=ZIP_DEFLATED)
    result["control_archive_sha256"] = sha256(archive.read_bytes()).hexdigest()
    (destination / "fermionic-control.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if not args.validate:
        run(args.output)
    from smoke.study_checks import validate
    print(json.dumps(validate(args.output), indent=2))


if __name__ == "__main__":
    main()
