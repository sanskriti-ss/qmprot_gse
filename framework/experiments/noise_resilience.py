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
from typing import Dict, List, Optional, Tuple

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

    logger.info("Wrote %d files to %s", 6, out_dir)


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

    cs_target = args.cs_target_qubits if args.cs_target_qubits > 0 else None

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
