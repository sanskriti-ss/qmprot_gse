#!/usr/bin/env python3
"""
Download QMProt Hamiltonians from PennyLane, truncate to the first ``max_terms``
Pauli lines (file order), and save either:

  * **h5** (default): ``<out_dir>/<id>/<id>.h5`` — same layout as ``HamiltonianLoader``.
  * **txt**: ``<out_dir>/hamiltonian_<id>.txt`` — use with ``main.py --legacy``.

**Bandwidth:** Full QMProt files are tens of GB. This script opens the remote
HDF5 over HTTPS and reads only enough ``hamiltonian*`` chunks (~32 MiB each)
until ``max_terms`` lines are available (often **one chunk** for small
``max_terms``), instead of ``qml.data.load`` without attributes.

Closed-shell only (no ``r-*`` radicals). Default list: 20 amino acids plus
amino-group, carboxy-group, hydrogen, methylidyne, water.

Use ``--verbose`` / ``-v`` for timestamped progress. ``--quiet`` disables tqdm
and verbose lines.

Requires ``aiohttp`` and ``requests`` (used by fsspec to read HTTPS HDF5).
Install: ``pip install aiohttp requests``
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pennylane as qml
from pennylane.data.data_manager.graphql import get_dataset_urls
from tqdm import tqdm


def _check_remote_hdf5_dependencies() -> None:
    """fsspec HTTP + h5py remote read needs aiohttp (and requests)."""
    missing: List[str] = []
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        missing.append("aiohttp")
    try:
        import requests  # noqa: F401
    except ImportError:
        missing.append("requests")
    if missing:
        print(
            "Missing packages for remote HDF5 streaming: "
            + ", ".join(missing)
            + "\nInstall with:  pip install aiohttp requests\n"
            "Or use the project venv and:  pip install -r framework/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _open_remote_hdf5(url: str) -> "h5py.File":
    _check_remote_hdf5_dependencies()
    from pennylane.data.base.hdf5 import open_hdf5_s3

    return open_hdf5_s3(url, block_size=8388608)

FRAMEWORK_DIR = Path(__file__).resolve().parent

# Closed-shell QMProt dataset IDs (no ``r-*`` radicals)
DEFAULT_MOLECULES = (
    "ala",
    "arg",
    "asn",
    "asp",
    "cys",
    "gln",
    "glu",
    "gly",
    "his",
    "ile",
    "leu",
    "lys",
    "met",
    "phe",
    "pro",
    "ser",
    "thr",
    "trp",
    "tyr",
    "val",
    "amino-group",
    "carboxy-group",
    "hydrogen",
    "methylidyne",
    "water",
)

Format = Literal["h5", "txt"]


def _vlog(message: str, *, verbose: bool) -> None:
    if verbose:
        stamp = datetime.now().isoformat(timespec="seconds")
        print(f"[{stamp}] {message}", flush=True)


def _safe_filename(molecule: str) -> str:
    return molecule.replace("/", "_")


def _valid_term_lines(hamiltonian_text: str) -> List[str]:
    return [
        line.strip()
        for line in hamiltonian_text.split("\n")
        if line.strip()
        and "Coefficient" not in line
        and "Operators" not in line
    ]


def _hamiltonian_chunk_keys(f: h5py.File) -> List[str]:
    return sorted(
        (k for k in f.keys() if k.startswith("hamiltonian")),
        key=lambda x: (len(x), x),
    )


def _remote_dataset_download_url(molecule: str) -> str:
    pairs = get_dataset_urls(molecule, [])
    if not pairs:
        raise RuntimeError(f"No PennyLane download URL for dataset {molecule!r}")
    return pairs[0][1]


def _read_truncated_lines_from_remote(
    remote: h5py.File,
    max_terms: int,
    max_hamiltonian_chunks: Optional[int],
    *,
    verbose: bool,
    molecule: str,
) -> Tuple[List[str], int, int]:
    """
    Stream ``hamiltonian*`` chunks until ``max_terms`` valid lines exist.

    Returns:
        (truncated_lines, chunks_read, total_hamiltonian_keys_in_file)
    """
    hk = _hamiltonian_chunk_keys(remote)
    n_keys = len(hk)
    parts: List[str] = []
    chunks_read = 0
    truncated: Optional[List[str]] = None

    for key in hk:
        if max_hamiltonian_chunks is not None and chunks_read >= max_hamiltonian_chunks:
            _vlog(
                f"{molecule!r}: hit --max-ham-chunks={max_hamiltonian_chunks}, "
                f"stopping early ({chunks_read}/{n_keys} chunks)",
                verbose=verbose,
            )
            break
        t_ck = time.perf_counter()
        raw = remote[key][()]
        chunks_read += 1
        parts.append(
            raw.decode("utf-8")
            if isinstance(raw, (bytes, bytearray))
            else str(raw)
        )
        lines = _valid_term_lines("".join(parts))
        _vlog(
            f"{molecule!r}: read {key} chunk {chunks_read}/{n_keys} "
            f"({time.perf_counter() - t_ck:.1f}s) → {len(lines)} term lines so far",
            verbose=verbose,
        )
        if len(lines) >= max_terms:
            truncated = lines[:max_terms] if max_terms > 0 else lines
            break

    if truncated is None:
        lines = _valid_term_lines("".join(parts))
        truncated = lines[:max_terms] if max_terms > 0 else lines

    return truncated, chunks_read, n_keys


def _decode_h5_key(key: Union[str, bytes]) -> str:
    if isinstance(key, bytes):
        return key.decode("ascii", errors="replace")
    return key


def _copy_h5_tree(src: h5py.Group, dst: h5py.Group) -> None:
    """Deep-copy datasets and groups from ``src`` into ``dst`` (e.g. geometry)."""
    for raw_key in src.keys():
        name = _decode_h5_key(raw_key)
        obj = src[raw_key]
        if isinstance(obj, h5py.Dataset):
            dst.create_dataset(name, data=obj[()])
        elif isinstance(obj, h5py.Group):
            _copy_h5_tree(obj, dst.create_group(name))


def _write_local_h5_from_remote(
    remote: h5py.File,
    dst_h5: Path,
    truncated_lines: List[str],
) -> int:
    """Copy metadata from an open remote file; write one ``hamiltonian`` blob."""
    body = "\n".join(truncated_lines) + ("\n" if truncated_lines else "")
    ham_arr = np.array(body.encode("utf-8"), dtype=object)
    n_written = len(truncated_lines)

    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst_h5, "w") as fout:
        for key in remote.keys():
            name = _decode_h5_key(key)
            if name.startswith("hamiltonian"):
                continue
            obj = remote[name]
            if isinstance(obj, h5py.Dataset):
                fout.create_dataset(name, data=obj[()])
            elif isinstance(obj, h5py.Group):
                _copy_h5_tree(obj, fout.create_group(name))
        if "n_coefficients" in fout:
            del fout["n_coefficients"]
        fout.create_dataset("n_coefficients", data=np.int64(n_written))
        fout.create_dataset("hamiltonian", data=ham_arr)
    return n_written


def download_hamiltonian_h5(
    molecule: str,
    max_terms: int,
    out_dir: Path,
    force: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    max_hamiltonian_chunks: Optional[int] = 32,
) -> Path:
    """Stream remote QMProt HDF5; only fetch enough ``hamiltonian*`` chunks for ``max_terms``."""
    out_dir = Path(out_dir)
    dst = out_dir / molecule / f"{molecule}.h5"
    if dst.is_file() and not force:
        _vlog(f"skip {molecule!r} (file exists) → {dst}", verbose=verbose)
        if not quiet:
            print(f"Skip existing {dst}")
        return dst

    t0 = time.perf_counter()
    _vlog(
        f"start {molecule!r}: remote HDF5 stream, keep first {max_terms} terms "
        f"(cap {max_hamiltonian_chunks} chunks)",
        verbose=verbose,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    url = _remote_dataset_download_url(molecule)
    _vlog(f"{molecule!r}: {url}", verbose=verbose)

    t_dl = time.perf_counter()
    remote = _open_remote_hdf5(url)
    try:
        truncated, chunks_used, n_ham_keys = _read_truncated_lines_from_remote(
            remote,
            max_terms,
            max_hamiltonian_chunks,
            verbose=verbose,
            molecule=molecule,
        )
        if len(truncated) < max_terms and max_terms > 0 and not quiet:
            print(
                f"Warning {molecule!r}: only {len(truncated)} terms after "
                f"{chunks_used}/{n_ham_keys} chunks "
                f"(raise --max-ham-chunks or lower --max-terms).",
                flush=True,
            )

        t_wr = time.perf_counter()
        n = _write_local_h5_from_remote(remote, dst, truncated)
        _vlog(
            f"{molecule!r}: wrote {n} terms, {chunks_used} chunks read, "
            f"local h5 in {time.perf_counter() - t_wr:.1f}s "
            f"(open+read total {time.perf_counter() - t_dl:.1f}s)",
            verbose=verbose,
        )
        if not quiet:
            print(f"Saved {n} terms for {molecule!r} → {dst}")
    finally:
        remote.close()

    _vlog(f"done  {molecule!r}: wall {time.perf_counter() - t0:.1f}s", verbose=verbose)
    return dst


def download_hamiltonian_txt(
    molecule: str,
    max_terms: int,
    out_dir: Path,
    quiet: bool = False,
    verbose: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    max_hamiltonian_chunks: Optional[int] = 32,
) -> Path:
    """Stream remote HDF5 and write legacy .txt (same chunk cap as h5)."""
    if save_path is not None:
        path = Path(save_path)
    else:
        path = out_dir / f"hamiltonian_{_safe_filename(molecule)}.txt"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _vlog(f"start {molecule!r} (txt): remote stream + save", verbose=verbose)
    if not quiet:
        print(f"Opening remote HDF5 for {molecule!r} …")

    url = _remote_dataset_download_url(molecule)
    remote = _open_remote_hdf5(url)
    try:
        valid_lines, chunks_used, n_ham = _read_truncated_lines_from_remote(
            remote,
            max_terms,
            max_hamiltonian_chunks,
            verbose=verbose,
            molecule=molecule,
        )
        if len(valid_lines) < max_terms and max_terms > 0 and not quiet:
            print(
                f"Warning {molecule!r}: only {len(valid_lines)} terms after "
                f"{chunks_used}/{n_ham} chunks.",
                flush=True,
            )
    finally:
        remote.close()

    if not quiet:
        print(f"Writing {len(valid_lines)} terms …")

    with open(path, "w", encoding="utf-8") as f:
        f.write("Coefficient\tOperators\n")
        iterator = (
            tqdm(valid_lines, desc=f"Save {molecule}", unit="terms")
            if not quiet
            else valid_lines
        )
        for line in iterator:
            parts = line.split()
            try:
                coeff = float(parts[0])
                op_string = " ".join(parts[1:])
                f.write(f"{coeff}\t{op_string}\n")
            except (ValueError, IndexError):
                continue

    if not quiet:
        print(f"Saved {len(valid_lines)} terms for {molecule!r} → {path}")
    _vlog(
        f"done  {molecule!r} (txt): total {time.perf_counter() - t0:.1f}s",
        verbose=verbose,
    )
    return path


def download_hamiltonian(
    molecule: str,
    max_terms: int,
    out_dir: Path,
    fmt: Format = "h5",
    force: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    max_hamiltonian_chunks: Optional[int] = 32,
) -> Path:
    if fmt == "h5":
        return download_hamiltonian_h5(
            molecule,
            max_terms,
            out_dir,
            force=force,
            quiet=quiet,
            verbose=verbose,
            max_hamiltonian_chunks=max_hamiltonian_chunks,
        )
    return download_hamiltonian_txt(
        molecule,
        max_terms,
        out_dir,
        quiet=quiet,
        verbose=verbose,
        save_path=save_path,
        max_hamiltonian_chunks=max_hamiltonian_chunks,
    )


def discover_default_molecules() -> List[str]:
    """Built-in list, verified against PennyLane catalog."""
    ids = set(qml.data.list_data_names())
    missing = [m for m in DEFAULT_MOLECULES if m not in ids]
    if missing:
        raise RuntimeError(
            f"These QMProt IDs are missing from PennyLane: {missing}. "
            "Upgrade pennylane and check cloud.pennylane.ai access."
        )
    return list(DEFAULT_MOLECULES)


def download_all(
    molecules: Sequence[str],
    max_terms: int,
    out_dir: Path,
    fmt: Format,
    force: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    max_hamiltonian_chunks: Optional[int] = 32,
) -> List[Path]:
    paths: List[Path] = []
    iterator: Sequence[str] | tqdm = molecules
    if not quiet:
        iterator = tqdm(
            molecules,
            desc="QMProt download",
            unit="mol",
            total=len(molecules),
        )
    for mol in iterator:
        paths.append(
            download_hamiltonian(
                mol,
                max_terms=max_terms,
                out_dir=out_dir,
                fmt=fmt,
                force=force,
                quiet=quiet,
                verbose=verbose,
                max_hamiltonian_chunks=max_hamiltonian_chunks,
            )
        )
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule",
        "-m",
        type=str,
        default=None,
        help="Single PennyLane id (e.g. gly). With no --all, default test is gly.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"All default closed-shell molecules ({len(DEFAULT_MOLECULES)}).",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=100,
        help="First N Pauli lines to keep (default: 100)",
    )
    parser.add_argument(
        "--max-ham-chunks",
        type=int,
        default=32,
        help="Max number of remote ~32 MiB Hamiltonian chunks to read per molecule "
        "(safety cap; default: 32). Use 0 for no cap (may download very large data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FRAMEWORK_DIR / "datasets2",
        help="Root directory: h5 → <out>/<id>/<id>.h5; txt → <out>/hamiltonian_<id>.txt",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("h5", "txt"),
        default="h5",
        help="Output format (default: h5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download from PennyLane and overwrite existing outputs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output (disables --verbose)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Timestamped log lines for each step (download / write / skip)",
    )
    args = parser.parse_args(argv)
    fmt: Format = args.format  # type: ignore[assignment]
    verbose = args.verbose and not args.quiet
    max_ham_chunks: Optional[int] = (
        None if args.max_ham_chunks == 0 else args.max_ham_chunks
    )

    if args.all:
        mols = discover_default_molecules()
        batch_t0 = time.perf_counter()
        _vlog(
            f"batch start: {len(mols)} molecules → {args.out_dir.resolve()} ({fmt})",
            verbose=verbose,
        )
        download_all(
            mols,
            args.max_terms,
            args.out_dir,
            fmt=fmt,
            force=args.force,
            quiet=args.quiet,
            verbose=verbose,
            max_hamiltonian_chunks=max_ham_chunks,
        )
        _vlog(
            f"batch done: {len(mols)} molecules in {time.perf_counter() - batch_t0:.1f}s",
            verbose=verbose,
        )
        print(f"Done: {len(mols)} molecules → {args.out_dir.resolve()} ({fmt})")
        return

    mol = args.molecule or "gly"
    download_hamiltonian(
        mol,
        max_terms=args.max_terms,
        out_dir=args.out_dir,
        fmt=fmt,
        force=args.force,
        quiet=args.quiet,
        verbose=verbose,
        max_hamiltonian_chunks=max_ham_chunks,
    )


if __name__ == "__main__":
    main()
