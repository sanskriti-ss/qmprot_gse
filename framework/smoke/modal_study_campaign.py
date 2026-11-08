# ABOUTME: Executes the native five-study acceptance campaign on a Modal CPU.
# ABOUTME: Downloads the saved campaign from a volume and verifies its molecular results locally.
from pathlib import Path
import shutil
from uuid import uuid4

import modal

from smoke.modal_representative import image as representative_image, retrieve, volume
from smoke.study_campaign import CONFIGURATION, run


app = modal.App("qmprot-study-acceptance")
image = representative_image.add_local_dir(CONFIGURATION, "/root/campaigns/acceptance")


@app.function(image=image, cpu=2, memory=4096, timeout=1200, retries=0, volumes={"/outputs": volume})
def calculate(run_id):
    directory = Path("/outputs") / run_id
    try:
        run(directory)
        archive = shutil.make_archive(str(directory), "tar", root_dir=directory)
    finally:
        volume.commit()
    return Path(archive).name


@app.local_entrypoint()
def main():
    run_id = "study-campaign-" + uuid4().hex
    print(f"Modal Volume qmprot-smoke-results/{run_id}", flush=True)
    archive = calculate.remote(run_id)
    destination = retrieve(run_id, archive)
    from smoke.study_checks import validate
    print(f"Verified {destination} {validate(destination)}", flush=True)
