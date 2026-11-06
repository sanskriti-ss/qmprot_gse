"""Run every experiment JSON in one campaign folder.

A campaign is one version of the comparison. Copy the folder, then change
the code, when the next version should stay separate from this one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Flags that take --no-<name> when the JSON value is false.
OPTIONAL_BOOLS = {
    "he-hf-init",
    "he-stop-at-accuracy",
    "prepare-hf",
}


def campaign_dir(name: str) -> Path:
    root = Path(__file__).resolve().parent
    path = Path(name)
    if not path.is_absolute():
        path = root / name
    if not path.is_dir():
        raise SystemExit(f"campaign folder not found: {path}")
    return path


def experiment_files(folder: Path) -> list[Path]:
    skipped = {"manifest.json"}
    files = [
        path for path in sorted(folder.glob("*.json"))
        if path.name not in skipped and not path.name.startswith("_")
    ]
    if not files:
        raise SystemExit(f"no experiment JSON files in {folder}")
    return files


def argv_from_args(spec: dict) -> list[str]:
    module = spec.get("module")
    if not module:
        raise SystemExit("experiment JSON needs a 'module' field")
    argv = [sys.executable, "-m", module]
    for key, value in spec.get("args", {}).items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if key in OPTIONAL_BOOLS:
                argv.append(flag if value else f"--no-{key}")
            elif value:
                argv.append(flag)
            continue
        if value is None:
            continue
        argv.append(flag)
        if isinstance(value, list):
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))
    return argv


def git_state(cwd: Path) -> dict:
    def capture(command: list[str]) -> str | None:
        try:
            out = subprocess.run(
                command, cwd=cwd, check=False, capture_output=True, text=True,
            )
        except OSError:
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    commit = capture(["git", "rev-parse", "HEAD"])
    dirty = capture(["git", "status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_dirty": None if dirty is None else bool(dirty),
    }


def output_flag(spec: dict) -> str:
    args = spec.get("args", {})
    if args.get("all-molecules") or args.get("molecules"):
        return "--output-root"
    return "--output-dir"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", help="Folder name under framework/campaigns/")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands and do not run them or write manifest.json.",
    )
    args = parser.parse_args()

    folder = campaign_dir(args.campaign)
    framework = folder.parent.parent
    files = experiment_files(folder)
    started = datetime.now(timezone.utc).isoformat()
    commands = []
    for path in files:
        spec = json.loads(path.read_text())
        argv = argv_from_args(spec)
        results = folder / "results" / path.stem
        if "output-dir" not in spec.get("args", {}) and "output-root" not in spec.get("args", {}):
            argv.extend([output_flag(spec), str(results)])
        commands.append({"file": path.name, "argv": argv, "results": str(results)})
        print(" ".join(argv))
        if args.dry_run:
            continue
        results.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(argv, cwd=framework)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    if args.dry_run:
        return

    manifest_path = folder / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("campaign", folder.name)
    manifest.setdefault("created", started)
    manifest.setdefault("runs", [])
    manifest["runs"].append({
        "started": started,
        **git_state(framework),
        "experiments": commands,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"recorded {manifest_path}")


if __name__ == "__main__":
    main()
