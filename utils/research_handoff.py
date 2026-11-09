"""Build and verify an offline source/evidence handoff using only Python and Git.

Run from the repository to archive: python utils/research_handoff.py build RUN.zip
Verify anywhere: python utils/research_handoff.py verify RUN.zip
The archive contains a standalone Git bundle, the tracked working-tree files,
and native results/logs. Symlinks and earlier handoffs are explicitly excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED


EVIDENCE_ROOTS = ("framework/results", "framework/experiments/results", "framework/logs")
MANIFEST = "EVIDENCE_MANIFEST.json"
BUNDLE = "repository.bundle"


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


def build(destination: Path):
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    destination = destination.absolute()
    tracked = git("ls-files", "-z", cwd=root).decode("utf-8").split("\0")[:-1]
    candidates = set(tracked)
    for directory in EVIDENCE_ROOTS:
        parent = root / directory
        if parent.is_symlink():
            candidates.add(directory)
        elif parent.is_dir():
            candidates.update(p.relative_to(root).as_posix() for p in parent.rglob("*"))
    files, omitted, missing = {}, [], []
    for name in sorted(candidates):
        path = root / name
        if name == "framework/results/handoffs" or name.startswith("framework/results/handoffs/"):
            continue
        if path.absolute() == destination:
            continue
        if any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents):
            omitted.append(name)
        elif path.is_file():
            if name in (MANIFEST, BUNDLE, "HANDOFF_RESTORE.txt"):
                raise ValueError(f"Reserved archive name: {name}")
            files[name] = path
        elif name in tracked and not path.exists():
            missing.append(name)
    manifest = dict(version=1, commit=git("rev-parse", "HEAD", cwd=root).decode().strip(),
                    tracked_files=sorted(set(tracked) & files.keys()),
                    deleted_tracked_files=missing, omitted_links=omitted, files={})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / BUNDLE
        git("bundle", "create", str(bundle), "--all", "HEAD", cwd=root)
        files[BUNDLE] = bundle
        # Exclusive creation protects an existing handoff from accidental replacement.
        with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, name)
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                manifest["files"][name] = dict(sha256=digest.hexdigest(), size=archive.getinfo(name).file_size)
            instructions = (
                "Verify this archive before restoring it. Extract into a new directory.\n"
                "Clone repository.bundle into a separate empty directory with git clone.\n"
                f"Check out commit {manifest['commit']}, then overlay the archived working-tree files.\n"
                "Remove deleted_tracked_files listed in EVIDENCE_MANIFEST.json from that checkout.\n"
                "The manifest records omitted symlinks; their targets are not included.\n"
                "Hashes detect corruption, not authenticity of an untrusted archive.\n"
            ).encode()
            archive.writestr("HANDOFF_RESTORE.txt", instructions)
            manifest["files"]["HANDOFF_RESTORE.txt"] = dict(
                sha256=hashlib.sha256(instructions).hexdigest(), size=len(instructions))
            archive.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")
    return dict(archive=str(destination), tracked_files=len(manifest["tracked_files"]))


def verify(path: Path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name or ":" in name:
                raise ValueError(f"Unsafe archive member: {name}")
        manifest = json.loads(archive.read(MANIFEST))
        if manifest["version"] != 1:
            raise ValueError("Unsupported handoff version")
        if set(names) != set(manifest["files"]) | {MANIFEST}:
            raise ValueError("Archive members differ from manifest")
        if not set(manifest["tracked_files"]) <= set(manifest["files"]):
            raise ValueError("Tracked files missing from manifest")
        for name, expected in manifest["files"].items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected["sha256"] or archive.getinfo(name).file_size != expected["size"]:
                raise ValueError("Content differs from manifest for " + name)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = directory / BUNDLE
            bundle.write_bytes(archive.read(BUNDLE))
            git("init", "-q", str(directory / "check"))
            git("bundle", "verify", str(bundle), cwd=directory / "check")
    return dict(tracked_files=len(manifest["tracked_files"]), files=len(manifest["files"]),
                commit=manifest["commit"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps((build if args.command == "build" else verify)(args.archive)))


if __name__ == "__main__":
    main()
