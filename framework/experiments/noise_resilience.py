"""
Experiment 1 -- Noise resilience of a fixed VQE ansatz
======================================================

Question
--------
If we optimise the *same* variational ansatz, starting from the *same*
initial parameters, on the *same* qubit Hamiltonian, but the simulator
applies a fixed amount of single-qubit noise after every layer, how much do
the optimised parameters drift -- and how much does the recovered energy
drift?

This is a deliberately controlled study: only the noise model and noise
strength change between runs.  Anything else that varied (the optimiser
seed, the initial parameter vector, the ansatz, the Hamiltonian) would
contaminate the measurement.

What we measure
---------------

For every (noise_model, noise_strength) pair we record:

* ``E_noisy`` -- the energy COBYLA converged to on the noisy backend
  (i.e. the value the noisy optimiser thinks it found).
* ``E_clean_at_noisy_params`` -- the energy of the noisy-optimum parameter
  vector evaluated on the noiseless statevector simulator.  This is the
  *true* (information-theoretic) performance of the noisy run.
* ``param_drift_l2`` -- ``||theta_noisy* - theta_clean*||_2``.
* ``param_drift_cosine`` -- cosine similarity between the noisy and
  noiseless optimum parameter vectors.
* The full convergence trace.

Outputs
-------
``framework/experiments/results/noise_resilience/<timestamp>_<mol>_<algo>/``

* ``run_config.json``        -- everything needed to reproduce.
* ``results.csv``            -- one row per (noise_model, noise_strength).
* ``convergence_traces.csv`` -- long-format convergence data.
* ``energy_vs_noise.png``    -- E_noisy and E_clean_at_noisy vs strength.
* ``param_drift_vs_noise.png`` -- L2 drift and cosine similarity.
* ``convergence.png``        -- convergence trajectories overlay.

Run as a script
---------------
::

    python -m experiments.noise_resilience \
        --molecule water --algorithm hardware_efficient_vqe \
        --n-layers 2 --max-iter 80 --cs-target-qubits 4

or import :func:`run_noise_resilience` from another script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow ``python experiments/noise_resilience.py`` directly
_FW_DIR = Path(__file__).resolve().parent.parent
if str(_FW_DIR) not in sys.path:
    sys.path.insert(0, str(_FW_DIR))

from experiments._common import (  # noqa: E402
    INIT_STRATEGIES,
    LoadedHamiltonian,
    build_vqe,
    load_small_hamiltonian,
    make_backend_config,
    make_run_dir,
    sample_initial_parameters,
)

logger = logging.getLogger(__name__)


# Default sweep covers two qualitatively different gate-error channels.
DEFAULT_NOISE_MODELS: Tuple[str, ...] = (
    "depolarizing",
    "phaseflip",
    "amplitude_damping",
)
DEFAULT_NOISE_STRENGTHS: Tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)

# Standard 20 amino acids present under ``framework/datasets2/`` (folder names).
DATASETS2_AMINO_ACIDS: Tuple[str, ...] = (
    "ala", "arg", "asn", "asp", "cys", "gln", "glu", "gly", "his",
    "ile", "leu", "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val",
)


# ──────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class NoiseRunResult:
    noise_model: str
    noise_strength: float
    n_iterations: int
    energy_noisy: float                      # what optimiser saw (noisy cost)
    energy_clean_at_noisy_params: float      # noiseless eval of noisy theta*
    energy_clean_baseline: float             # noiseless eval of clean theta*
    param_drift_l2: float
    param_drift_cosine: float
    convergence_history: List[float] = field(default_factory=list)


@dataclass
class NoiseResilienceConfig:
    molecule: str
    algorithm: str
    basis: str
    cs_target_qubits: Optional[int]
    n_layers: int
    max_iterations: int
    convergence_threshold: float
    optimizer: str
    init_strategy: str
    random_seed: int
    noise_models: Tuple[str, ...]
    noise_strengths: Tuple[float, ...]
    n_qubits_active: int
    n_qubits_final: int
    n_parameters: int
    casci_energy: Optional[float]
    hf_energy: Optional[float]


# ──────────────────────────────────────────────────────────────────────────
# Core single-run helper
# ──────────────────────────────────────────────────────────────────────────

def _optimize_one(
    loaded: LoadedHamiltonian,
    *,
    algorithm: str,
    n_layers: int,
    optimizer: str,
    max_iterations: int,
    convergence_threshold: float,
    random_seed: int,
    noise_model: Optional[str],
    noise_strength: float,
    initial_parameters: np.ndarray,
) -> Tuple[np.ndarray, List[float], int]:
    """Run one optimisation under a chosen noise channel.

    Returns ``(theta_star, convergence_history, n_iterations)``.
    """
    bc = make_backend_config(
        n_qubits=loaded.hamiltonian.n_qubits,
        noise_model=noise_model,
        noise_strength=noise_strength,
    )
    vqe = build_vqe(
        loaded.hamiltonian,
        algorithm=algorithm,
        n_layers=n_layers,
        backend_config=bc,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
    )
    theta_star, _ = vqe.optimize(initial_parameters=initial_parameters.copy())
    return np.asarray(theta_star), list(vqe.convergence_history), vqe.iteration_count


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def run_noise_resilience(
    molecule: str = "water",
    algorithm: str = "hardware_efficient_vqe",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = 4,
    n_layers: int = 2,
    max_iterations: int = 150,
    convergence_threshold: float = 1e-10,
    optimizer: str = "COBYLA",
    init_strategy: str = "small_random",
    random_seed: int = 42,
    noise_models: Tuple[str, ...] = DEFAULT_NOISE_MODELS,
    noise_strengths: Tuple[float, ...] = DEFAULT_NOISE_STRENGTHS,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> Dict[str, object]:
    """Run the full noise-resilience sweep.

    The sweep iterates ``noise_models x noise_strengths`` plus an explicit
    ``noise_strength == 0`` baseline run on the statevector simulator.  All
    runs share the *same* initial parameter vector so any drift is purely
    attributable to the noise channel.

    Returns a dict with keys:

    * ``"config"``    -- :class:`NoiseResilienceConfig`
    * ``"baseline"``  -- :class:`NoiseRunResult` for the noiseless reference
    * ``"results"``   -- ``List[NoiseRunResult]``
    * ``"output_dir"`` -- path of the run folder
    """
    if init_strategy not in INIT_STRATEGIES:
        raise ValueError(
            f"init_strategy must be one of {INIT_STRATEGIES}, got {init_strategy!r}"
        )

    # ── 1. Load a small Hamiltonian ─────────────────────────────────────
    loaded = load_small_hamiltonian(
        molecule=molecule, basis=basis, cs_target_qubits=cs_target_qubits,
    )

    # ── 2. Build the noiseless ansatz once to know n_parameters ────────
    bc_clean = make_backend_config(loaded.hamiltonian.n_qubits)
    probe = build_vqe(
        loaded.hamiltonian,
        algorithm=algorithm,
        n_layers=n_layers,
        backend_config=bc_clean,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
    )
    n_parameters = int(probe.n_parameters)
    clean_cost_fn = probe.cost_fn  # keep this around to evaluate noisy thetas

    # ── 3. Sample one shared initial parameter vector ───────────────────
    rng = np.random.default_rng(random_seed)
    theta_init = sample_initial_parameters(
        n_parameters, strategy=init_strategy, rng=rng,
    )

    cfg = NoiseResilienceConfig(
        molecule=molecule,
        algorithm=algorithm,
        basis=basis,
        cs_target_qubits=cs_target_qubits,
        n_layers=n_layers,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        optimizer=optimizer,
        init_strategy=init_strategy,
        random_seed=random_seed,
        noise_models=tuple(noise_models),
        noise_strengths=tuple(noise_strengths),
        n_qubits_active=loaded.n_qubits_active,
        n_qubits_final=loaded.n_qubits_final,
        n_parameters=n_parameters,
        casci_energy=loaded.casci_energy,
        hf_energy=loaded.hf_energy,
    )

    out_dir = output_dir or make_run_dir(
        "noise_resilience", molecule, algorithm,
    )
    logger.info("Output dir: %s", out_dir)

    # ── 4. Noiseless baseline run ───────────────────────────────────────
    logger.info(
        "[baseline] noiseless | n_qubits=%d, n_params=%d, init=%s",
        loaded.hamiltonian.n_qubits, n_parameters, init_strategy,
    )
    theta_clean, conv_clean, niter_clean = _optimize_one(
        loaded,
        algorithm=algorithm,
        n_layers=n_layers,
        optimizer=optimizer,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed,
        noise_model=None,
        noise_strength=0.0,
        initial_parameters=theta_init,
    )
    e_clean = float(clean_cost_fn(theta_clean))
    baseline = NoiseRunResult(
        noise_model="none",
        noise_strength=0.0,
        n_iterations=niter_clean,
        energy_noisy=e_clean,
        energy_clean_at_noisy_params=e_clean,
        energy_clean_baseline=e_clean,
        param_drift_l2=0.0,
        param_drift_cosine=1.0,
        convergence_history=conv_clean,
    )

    # ── 5. Noisy sweep ──────────────────────────────────────────────────
    results: List[NoiseRunResult] = []
    for noise_model in noise_models:
        for p in noise_strengths:
            if p == 0.0:
                # Treat as identical to baseline; record one entry per model
                # so plots have a clean p=0 anchor for every channel.
                results.append(NoiseRunResult(
                    noise_model=noise_model,
                    noise_strength=0.0,
                    n_iterations=niter_clean,
                    energy_noisy=e_clean,
                    energy_clean_at_noisy_params=e_clean,
                    energy_clean_baseline=e_clean,
                    param_drift_l2=0.0,
                    param_drift_cosine=1.0,
                    convergence_history=list(conv_clean),
                ))
                continue

            logger.info("[run] noise=%s p=%.4f", noise_model, p)
            theta_noisy, conv_noisy, niter_noisy = _optimize_one(
                loaded,
                algorithm=algorithm,
                n_layers=n_layers,
                optimizer=optimizer,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                random_seed=random_seed,
                noise_model=noise_model,
                noise_strength=p,
                initial_parameters=theta_init,
            )
            e_noisy = float(conv_noisy[-1]) if conv_noisy else float("nan")
            # Re-evaluate the noisy-optimum parameters on the *clean* simulator
            e_clean_at_noisy = float(clean_cost_fn(theta_noisy))
            drift_l2 = float(np.linalg.norm(theta_noisy - theta_clean))
            denom = float(np.linalg.norm(theta_noisy) * np.linalg.norm(theta_clean))
            cos = (
                float(np.dot(theta_noisy, theta_clean) / denom) if denom > 0 else 1.0
            )
            results.append(NoiseRunResult(
                noise_model=noise_model,
                noise_strength=p,
                n_iterations=niter_noisy,
                energy_noisy=e_noisy,
                energy_clean_at_noisy_params=e_clean_at_noisy,
                energy_clean_baseline=e_clean,
                param_drift_l2=drift_l2,
                param_drift_cosine=cos,
                convergence_history=conv_noisy,
            ))

    # ── 6. Persist ──────────────────────────────────────────────────────
    _save_outputs(cfg, baseline, results, out_dir, save_plots=save_plots)

    return {
        "config": cfg,
        "baseline": baseline,
        "results": results,
        "output_dir": out_dir,
    }


# ──────────────────────────────────────────────────────────────────────────
# Output: CSV / JSON / plots
# ──────────────────────────────────────────────────────────────────────────

def _save_outputs(
    cfg: NoiseResilienceConfig,
    baseline: NoiseRunResult,
    results: List[NoiseRunResult],
    out_dir: Path,
    save_plots: bool = True,
) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    # run_config.json
    cfg_dict = asdict(cfg)
    cfg_dict["noise_models"] = list(cfg.noise_models)
    cfg_dict["noise_strengths"] = list(cfg.noise_strengths)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # results.csv -- one row per noisy run
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "noise_model", "noise_strength", "n_iterations",
            "energy_noisy", "energy_clean_at_noisy_params",
            "energy_clean_baseline",
            "param_drift_l2", "param_drift_cosine",
        ])
        for r in [baseline] + results:
            w.writerow([
                r.noise_model, f"{r.noise_strength:.6g}", r.n_iterations,
                f"{r.energy_noisy:.10f}", f"{r.energy_clean_at_noisy_params:.10f}",
                f"{r.energy_clean_baseline:.10f}",
                f"{r.param_drift_l2:.10f}", f"{r.param_drift_cosine:.10f}",
            ])

    # convergence_traces.csv -- long format
    with open(out_dir / "convergence_traces.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise_model", "noise_strength", "iteration", "energy"])
        for r in [baseline] + results:
            for it, e in enumerate(r.convergence_history):
                w.writerow([
                    r.noise_model, f"{r.noise_strength:.6g}", it, f"{e:.10f}",
                ])

    if not save_plots:
        return

    # ── Plots ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib usually present
        logger.warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    by_model: Dict[str, List[NoiseRunResult]] = {}
    for r in results:
        by_model.setdefault(r.noise_model, []).append(r)

    # 1) Energy vs noise strength
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(baseline.energy_clean_baseline, color="black", linestyle="--",
               linewidth=1, label=f"noiseless baseline = {baseline.energy_clean_baseline:.5f}")
    if cfg.casci_energy is not None:
        ax.axhline(cfg.casci_energy, color="grey", linestyle=":",
                   linewidth=1, label=f"CASCI = {cfg.casci_energy:.5f}")
    for model, rs in by_model.items():
        rs_sorted = sorted(rs, key=lambda r: r.noise_strength)
        xs = [r.noise_strength for r in rs_sorted]
        ax.plot(xs, [r.energy_noisy for r in rs_sorted],
                marker="o", label=f"{model}: noisy cost")
        ax.plot(xs, [r.energy_clean_at_noisy_params for r in rs_sorted],
                marker="x", linestyle="--", label=f"{model}: clean@noisy_theta")
    ax.set_xlabel("Noise strength p")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(
        f"Noise resilience: {cfg.molecule} ({cfg.algorithm}, "
        f"{cfg.n_qubits_final} qubits, {cfg.n_layers} layers)"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_vs_noise.png", dpi=150)
    plt.close(fig)

    # 2) Parameter drift vs noise strength
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, rs in by_model.items():
        rs_sorted = sorted(rs, key=lambda r: r.noise_strength)
        xs = [r.noise_strength for r in rs_sorted]
        ax1.plot(xs, [r.param_drift_l2 for r in rs_sorted], marker="o", label=model)
        ax2.plot(xs, [r.param_drift_cosine for r in rs_sorted], marker="o", label=model)
    ax1.set_xlabel("Noise strength p")
    ax1.set_ylabel(r"$||\theta_{noisy}^* - \theta_{clean}^*||_2$")
    ax1.set_title("Optimum parameter drift (L2)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Noise strength p")
    ax2.set_ylabel(r"$\cos(\theta_{noisy}^*, \theta_{clean}^*)$")
    ax2.set_title("Optimum parameter cosine similarity")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "param_drift_vs_noise.png", dpi=150)
    plt.close(fig)

    # 3) Convergence trajectories
    fig, axes = plt.subplots(1, len(by_model), figsize=(5 * len(by_model), 4.5),
                             sharey=True, squeeze=False)
    for ax, (model, rs) in zip(axes[0], by_model.items()):
        ax.plot(baseline.convergence_history, color="black", linewidth=2,
                label="noiseless")
        rs_sorted = sorted(rs, key=lambda r: r.noise_strength)
        cmap = plt.get_cmap("viridis")
        nz_runs = [r for r in rs_sorted if r.noise_strength > 0]
        n = max(1, len(nz_runs))
        for i, r in enumerate(nz_runs):
            ax.plot(
                r.convergence_history,
                color=cmap(i / n),
                linewidth=1,
                label=f"p={r.noise_strength:g}",
            )
        ax.set_xlabel("COBYLA iteration")
        ax.set_title(model)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[0, 0].set_ylabel("Energy (Ha)")
    fig.suptitle(
        f"Convergence under noise: {cfg.molecule}, {cfg.algorithm}, "
        f"{cfg.n_qubits_final} qubits"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=150)
    plt.close(fig)

    logger.info("Wrote outputs to %s", out_dir)


# ──────────────────────────────────────────────────────────────────────────
# Multi-molecule aggregation plots + batch runner
# ──────────────────────────────────────────────────────────────────────────

def plot_noise_resilience_multi_molecule(
    combined_csv: Path,
    out_dir: Path,
    *,
    reference_p: float = 0.02,
    dpi: int = 150,
) -> None:
    """Visualise noise sweep results for many molecules in one folder.

    Reads a long-format CSV (see :func:`run_noise_resilience_batch`) with at
    least columns ``molecule``, ``noise_model``, ``noise_strength``,
    ``param_drift_l2``, ``param_drift_cosine``.

    Produces:

    * ``heatmap_param_drift_l2.png`` -- one heatmap row per noise channel;
      colour shows :math:`||\\Delta\\theta||_2` (molecules on the *y* axis,
      noise strength *p* on the *x* axis).
    * ``heatmap_param_drift_cosine.png`` -- same layout for cosine
      similarity (clipped to [-1, 1] for the colour scale).
    * ``heatmap_energy_gap_clean_noisy.png`` -- per-channel heatmap of
      ``energy_clean_at_noisy_params - energy_clean_baseline`` (Ha); shows
      how far the *noiseless* evaluation of the noisy optimum sits above the
      noiseless optimum.
    * ``bars_reference_strength.png`` -- for ``p = reference_p`` (default
      0.02), horizontal bars of L2 drift sorted within each noise channel,
      so outliers are easy to spot.

    This layout keeps ~20 curves readable: heatmaps give the global picture,
    bars highlight ranking at one physically meaningful noise strength.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "plot_noise_resilience_multi_molecule requires pandas; "
            "install with `pip install pandas`."
        ) from exc

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib/seaborn unavailable: %s", exc)
        return

    if not combined_csv.is_file():
        logger.warning("Combined CSV not found: %s", combined_csv)
        return

    df = pd.read_csv(combined_csv)
    required = {
        "molecule", "noise_model", "noise_strength",
        "param_drift_l2", "param_drift_cosine",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"combined CSV missing columns {sorted(missing)}")

    # Drop the synthetic noiseless baseline row when present.
    df = df[df["noise_model"].astype(str).str.lower() != "none"].copy()
    df["noise_strength"] = pd.to_numeric(df["noise_strength"], errors="coerce")

    if "energy_clean_at_noisy_params" in df.columns and "energy_clean_baseline" in df.columns:
        ec = pd.to_numeric(df["energy_clean_at_noisy_params"], errors="coerce")
        eb = pd.to_numeric(df["energy_clean_baseline"], errors="coerce")
        df["delta_e_clean_eval"] = ec - eb
    else:
        df["delta_e_clean_eval"] = float("nan")

    mol_order = sorted(df["molecule"].unique())
    p_order = sorted(df["noise_strength"].dropna().unique())
    models = sorted(df["noise_model"].unique())
    out_dir.mkdir(parents=True, exist_ok=True)

    def _one_heatmap_panel(
        value_col: str,
        title: str,
        cbar_label: str,
        cmap: str,
        filename: str,
        vmin=None,
        vmax=None,
        center=None,
    ) -> None:
        nmodels = max(1, len(models))
        # Enough vertical space per molecule row so y-labels do not overlap.
        row_h = max(0.38, min(0.55, 14.0 / max(len(mol_order), 1)))
        fig_h = 2.35 * nmodels + row_h * len(mol_order)
        # Extra horizontal space for full molecule names on the left; slightly
        # narrower colour bar (`shrink`) buys a bit more room for the grid.
        fig_w = 11.5 + 0.06 * max((len(str(m)) for m in mol_order), default=3)
        fig, axes = plt.subplots(
            nmodels, 1, figsize=(min(fig_w, 14.0), fig_h), squeeze=False,
            sharex=True,
        )
        # ``axes`` has shape (nmodels, 1).  ``axes[0]`` is only the *first row*
        # (one Axes), so ``zip(axes[0], models)`` plotted a single panel — the
        # rest stayed blank.  Iterate down the column explicitly.
        ax_col = np.atleast_1d(axes[:, 0]).ravel()
        for ax, model in zip(ax_col, models):
            sub = df[df["noise_model"] == model]
            if sub.empty:
                ax.set_visible(False)
                continue
            pivot = (
                sub.pivot_table(
                    index="molecule",
                    columns="noise_strength",
                    values=value_col,
                    aggfunc="first",
                )
                .reindex(index=mol_order, columns=p_order)
            )
            sns.heatmap(
                pivot,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                linewidths=0.5,
                linecolor="white",
                yticklabels=mol_order,
                cbar_kws={
                    "label": cbar_label,
                    "shrink": 0.68,
                    "aspect": 22,
                },
            )
            ax.set_title(f"{title} — {model}")
            ax.set_xlabel("Noise strength p")
            ax.set_ylabel("Molecule")
            ax.tick_params(axis="y", which="major", labelsize=7.5, length=0)
            # Right-align so names sit flush to the grid (readable at ~20 rows).
            plt.setp(
                ax.get_yticklabels(),
                rotation=0,
                ha="right",
                va="center",
            )
        fig.suptitle(
            f"{title} (all molecules)",
            fontsize=12, y=1.01,
        )
        fig.tight_layout()
        # Reserve left margin for long abbreviations (trp, phe, ...).
        fig.subplots_adjust(left=0.20)
        fig.savefig(out_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    _one_heatmap_panel(
        "param_drift_l2",
        r"$||\theta_{\mathrm{noisy}}^* - \theta_{\mathrm{clean}}^*||_2$",
        r"$||\Delta\theta||_2$",
        "rocket",
        "heatmap_param_drift_l2.png",
    )
    _one_heatmap_panel(
        "param_drift_cosine",
        r"$\cos(\theta_{\mathrm{noisy}}^*, \theta_{\mathrm{clean}}^*)$",
        "cosine similarity",
        "vlag",
        "heatmap_param_drift_cosine.png",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
    )
    if df["delta_e_clean_eval"].notna().any():
        _one_heatmap_panel(
            "delta_e_clean_eval",
            r"$E_{\mathrm{clean}}(\theta_{\mathrm{noisy}}^*) - "
            r"E_{\mathrm{clean}}(\theta_{\mathrm{clean}}^*)$ (Ha)",
            "ΔE (Ha)",
            "mako",
            "heatmap_energy_gap_clean_noisy.png",
        )

    # Reference-strength horizontal bars (closest tabulated p to reference_p).
    avail_p = sorted(df["noise_strength"].dropna().unique())
    if avail_p:
        closest_p = min(avail_p, key=lambda x: abs(float(x) - reference_p))
    else:
        closest_p = reference_p
    ns_arr = df["noise_strength"].to_numpy(dtype=float, copy=False)
    mask = np.isclose(ns_arr, float(closest_p), rtol=0.0, atol=1e-12)
    sub_ref = df.loc[mask]
    if sub_ref.empty:
        sub_ref = df[df["noise_strength"] == closest_p]

    if not sub_ref.empty:
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 8),
                                 squeeze=False)
        # Same pitfall as heatmaps: use ravel so every panel is addressed.
        for ax, model in zip(np.ravel(axes), models):
            chunk = sub_ref[sub_ref["noise_model"] == model].copy()
            if chunk.empty:
                ax.set_visible(False)
                continue
            chunk = chunk.sort_values("param_drift_l2", ascending=True)
            y_pos = np.arange(len(chunk))
            ax.barh(y_pos, chunk["param_drift_l2"].values, color="steelblue", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(chunk["molecule"].values, fontsize=8)
            ax.set_xlabel(r"$||\Delta\theta||_2$")
            ax.set_title(f"{model}\np ≈ {closest_p:g}")
            ax.grid(True, axis="x", alpha=0.3)
        fig.suptitle(
            f"Parameter L2 drift at reference noise (target p={reference_p:g}, "
            f"using tabulated p={closest_p:g})",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "bars_reference_strength.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Compact manifest for quick grep.
    manifest = out_dir / "multi_plot_manifest.txt"
    with open(manifest, "w") as f:
        f.write(
            "Generated multi-molecule noise-resilience figures.\n"
            f"source_csv: {combined_csv.resolve()}\n"
            f"molecules: {len(mol_order)}\n"
            f"noise_models: {models}\n"
            f"p values: {p_order}\n"
        )
    logger.info("Wrote multi-molecule plots to %s", out_dir)


def _rows_for_batch_csv(
    molecule: str,
    cfg: NoiseResilienceConfig,
    baseline: NoiseRunResult,
    results: List[NoiseRunResult],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in [baseline] + results:
        rows.append({
            "molecule": molecule,
            "noise_model": r.noise_model,
            "noise_strength": r.noise_strength,
            "n_iterations": r.n_iterations,
            "energy_noisy": r.energy_noisy,
            "energy_clean_at_noisy_params": r.energy_clean_at_noisy_params,
            "energy_clean_baseline": r.energy_clean_baseline,
            "param_drift_l2": r.param_drift_l2,
            "param_drift_cosine": r.param_drift_cosine,
            "n_qubits_final": cfg.n_qubits_final,
            "n_parameters": cfg.n_parameters,
        })
    return rows


def run_noise_resilience_batch(
    molecules: Sequence[str],
    *,
    algorithm: str = "hardware_efficient_vqe",
    basis: str = "sto-3g",
    cs_target_qubits: Optional[int] = 6,
    n_layers: int = 2,
    max_iterations: int = 150,
    convergence_threshold: float = 1e-10,
    optimizer: str = "COBYLA",
    init_strategy: str = "small_random",
    random_seed: int = 42,
    noise_models: Tuple[str, ...] = DEFAULT_NOISE_MODELS,
    noise_strengths: Tuple[float, ...] = DEFAULT_NOISE_STRENGTHS,
    output_dir: Optional[Path] = None,
    save_per_molecule_plots: bool = True,
    save_aggregate_plots: bool = True,
    reference_p_for_bars: float = 0.02,
) -> Dict[str, object]:
    """Run :func:`run_noise_resilience` for each molecule and aggregate CSV + plots.

    Writes ``batch_all_results.csv`` (long format, every molecule) under
    ``output_dir``, ``batch_run_config.json``, ``batch_failures.json`` on
    errors, and calls :func:`plot_noise_resilience_multi_molecule` when
    ``save_aggregate_plots`` is True.

    Returns keys ``output_dir``, ``batch_csv``, ``n_ok``, ``n_fail``,
    ``failures``, ``per_molecule_dirs``.
    """
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        tag = f"nq{cs_target_qubits}_{algorithm}_batch"
        output_dir = (
            _FW_DIR / "experiments" / "results" / "noise_resilience_batch"
            / f"{ts}_{tag}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_csv = output_dir / "batch_all_results.csv"
    failures: List[Dict[str, str]] = []
    per_molecule_dirs: Dict[str, str] = {}

    import csv

    header = [
        "molecule", "noise_model", "noise_strength", "n_iterations",
        "energy_noisy", "energy_clean_at_noisy_params", "energy_clean_baseline",
        "param_drift_l2", "param_drift_cosine",
        "n_qubits_final", "n_parameters",
    ]
    first = True

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None  # type: ignore[misc,assignment]

    mol_iter = molecules
    if _tqdm is not None:
        mol_iter = _tqdm(list(molecules), desc="noise_resilience batch")

    for mol in mol_iter:
        mol = mol.strip().lower()
        try:
            sub_out = output_dir / "per_molecule" / mol
            sub_out.mkdir(parents=True, exist_ok=True)
            out = run_noise_resilience(
                molecule=mol,
                algorithm=algorithm,
                basis=basis,
                cs_target_qubits=cs_target_qubits,
                n_layers=n_layers,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                optimizer=optimizer,
                init_strategy=init_strategy,
                random_seed=random_seed,
                noise_models=noise_models,
                noise_strengths=noise_strengths,
                output_dir=sub_out,
                save_plots=save_per_molecule_plots,
            )
            cfg: NoiseResilienceConfig = out["config"]
            baseline: NoiseRunResult = out["baseline"]
            results: List[NoiseRunResult] = out["results"]
            rows = _rows_for_batch_csv(mol, cfg, baseline, results)
            per_molecule_dirs[mol] = str(sub_out)

            with open(batch_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if first:
                    w.writeheader()
                    first = False
                for row in rows:
                    w.writerow({
                        k: row[k] if k in row else ""
                        for k in header
                    })
        except Exception as exc:
            logger.exception("Batch: molecule %s failed: %s", mol, exc)
            failures.append({"molecule": mol, "error": str(exc)})

    meta = {
        "timestamp": ts,
        "molecules_requested": list(molecules),
        "algorithm": algorithm,
        "basis": basis,
        "cs_target_qubits": cs_target_qubits,
        "n_layers": n_layers,
        "max_iterations": max_iterations,
        "noise_models": list(noise_models),
        "noise_strengths": list(noise_strengths),
        "n_ok": len(per_molecule_dirs),
        "n_fail": len(failures),
    }
    with open(output_dir / "batch_run_config.json", "w") as f:
        json.dump(meta, f, indent=2)
    if failures:
        with open(output_dir / "batch_failures.json", "w") as f:
            json.dump(failures, f, indent=2)

    if save_aggregate_plots and batch_csv.is_file() and not first:
        try:
            plot_noise_resilience_multi_molecule(
                batch_csv, output_dir, reference_p=reference_p_for_bars,
            )
        except Exception as exc:
            logger.warning("Aggregate plotting failed: %s", exc)

    return {
        "output_dir": output_dir,
        "batch_csv": batch_csv,
        "n_ok": len(per_molecule_dirs),
        "n_fail": len(failures),
        "failures": failures,
        "per_molecule_dirs": per_molecule_dirs,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Noise-resilience sweep for a fixed VQE ansatz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--molecule", "-m", default="water",
                   help="Molecule abbreviation in datasets2/")
    p.add_argument("--algorithm", "-a", default="hardware_efficient_vqe",
                   help="Algorithm name (hardware_efficient_vqe or pennylane_vqe)")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--cs-target-qubits", type=int, default=4,
                   help="CS reduction target qubits; <=0 to skip CS")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-iter", type=int, default=150,
                   help="COBYLA max iterations per run")
    p.add_argument("--convergence-threshold", type=float, default=1e-10,
                   help="Optimiser tolerance (COBYLA's rhoend); tight by default "
                        "so the optimiser does not exit at iter ~10")
    p.add_argument("--optimizer", default="COBYLA")
    p.add_argument("--init-strategy", default="small_random",
                   choices=list(INIT_STRATEGIES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-models", nargs="+",
                   default=list(DEFAULT_NOISE_MODELS))
    p.add_argument("--noise-strengths", nargs="+", type=float,
                   default=list(DEFAULT_NOISE_STRENGTHS))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    # Multi-molecule batch (datasets2 amino acids or explicit list)
    p.add_argument(
        "--batch-amino-acids", action="store_true",
        help="Run all 20 standard amino acids from datasets2/ "
             "(implies aggregated heatmaps + batch CSV).",
    )
    p.add_argument(
        "--batch-molecules", nargs="+", default=None,
        help="Run several molecules in one batch (folder names under datasets2/).",
    )
    p.add_argument(
        "--batch-output-dir", type=Path, default=None,
        help="Optional parent folder for batch outputs.",
    )
    p.add_argument(
        "--no-per-molecule-plots", action="store_true",
        help="In batch mode, skip per-molecule PNGs (faster, smaller disk).",
    )
    p.add_argument(
        "--regenerate-multi-plots-from", type=Path, default=None,
        help="Only rebuild multi-molecule figures from an existing "
             "batch_all_results.csv path.",
    )
    p.add_argument(
        "--reference-p-bars", type=float, default=0.02,
        help="Target noise strength for ranked bar summary "
             "(closest tabulated p is used).",
    )
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    # Quiet down very chatty submodules
    for noisy_logger in ("pyscf", "pennylane", "matplotlib"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    if args.regenerate_multi_plots_from is not None:
        csv_path = args.regenerate_multi_plots_from.expanduser().resolve()
        out_d = csv_path.parent
        plot_noise_resilience_multi_molecule(
            csv_path, out_d, reference_p=args.reference_p_bars,
        )
        print(f"Multi-molecule figures written next to {csv_path}")
        return

    cs_target = args.cs_target_qubits if args.cs_target_qubits > 0 else None

    batch_mols: Optional[List[str]] = None
    if args.batch_amino_acids:
        batch_mols = list(DATASETS2_AMINO_ACIDS)
    elif args.batch_molecules:
        batch_mols = [m.strip().lower() for m in args.batch_molecules]

    if batch_mols:
        if args.batch_amino_acids and args.batch_molecules:
            logger.warning("Both --batch-amino-acids and --batch-molecules set; "
                           "using amino-acid list.")
            batch_mols = list(DATASETS2_AMINO_ACIDS)

        summary = run_noise_resilience_batch(
            batch_mols,
            algorithm=args.algorithm,
            basis=args.basis,
            cs_target_qubits=cs_target,
            n_layers=args.n_layers,
            max_iterations=args.max_iter,
            convergence_threshold=args.convergence_threshold,
            optimizer=args.optimizer,
            init_strategy=args.init_strategy,
            random_seed=args.seed,
            noise_models=tuple(args.noise_models),
            noise_strengths=tuple(args.noise_strengths),
            output_dir=args.batch_output_dir,
            save_per_molecule_plots=not args.no_per_molecule_plots,
            save_aggregate_plots=True,
            reference_p_for_bars=args.reference_p_bars,
        )
        print("\n" + "=" * 60)
        print("BATCH NOISE RESILIENCE COMPLETE")
        print("=" * 60)
        print(f"OK: {summary['n_ok']}  failed: {summary['n_fail']}")
        print(f"Combined CSV: {summary['batch_csv']}")
        print(f"Output folder: {summary['output_dir']}")
        if summary["failures"]:
            print("Failures:")
            for f in summary["failures"]:
                print(f"  {f['molecule']}: {f['error']}")
        return

    out = run_noise_resilience(
        molecule=args.molecule,
        algorithm=args.algorithm,
        basis=args.basis,
        cs_target_qubits=cs_target,
        n_layers=args.n_layers,
        max_iterations=args.max_iter,
        convergence_threshold=args.convergence_threshold,
        optimizer=args.optimizer,
        init_strategy=args.init_strategy,
        random_seed=args.seed,
        noise_models=tuple(args.noise_models),
        noise_strengths=tuple(args.noise_strengths),
        save_plots=not args.no_plots,
    )

    cfg: NoiseResilienceConfig = out["config"]
    baseline: NoiseRunResult = out["baseline"]
    results: List[NoiseRunResult] = out["results"]

    print("\n" + "=" * 60)
    print(f"NOISE RESILIENCE SWEEP -- {cfg.molecule} / {cfg.algorithm}")
    print("=" * 60)
    print(f"Active-space qubits: {cfg.n_qubits_active} -> final {cfg.n_qubits_final} "
          f"(CS={cfg.cs_target_qubits})")
    print(f"Ansatz: {cfg.algorithm}, n_layers={cfg.n_layers}, "
          f"n_parameters={cfg.n_parameters}, init={cfg.init_strategy}")
    print(f"HF energy:    {cfg.hf_energy}")
    print(f"CASCI energy: {cfg.casci_energy}")
    print(f"Noiseless VQE baseline: {baseline.energy_clean_baseline:.6f} Ha")
    print()
    print(f"{'noise_model':<22} {'p':>8} {'E_noisy':>13} "
          f"{'E_clean@noisy_θ':>18} {'||Δθ||₂':>10} {'cos':>7}")
    for r in results:
        print(
            f"{r.noise_model:<22} {r.noise_strength:>8.4f} "
            f"{r.energy_noisy:>13.6f} {r.energy_clean_at_noisy_params:>18.6f} "
            f"{r.param_drift_l2:>10.4f} {r.param_drift_cosine:>7.3f}"
        )
    print(f"\nResults saved to: {out['output_dir']}")


if __name__ == "__main__":
    main()
