"""Plot saved parallel amino-acid checkpoints without importing training code."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from figure_style import style, finish, SUNSET

NAMES = {
    "pennylane_vqe": "PennyLane VQE", "adapt_vqe": "ADAPT (approx.)",
    "qubit_adapt_vqe": "Qubit-ADAPT", "hardware_efficient_vqe": "Hardware-efficient",
    "qaoa_inspired_vqe": "QAOA-inspired", "iqcc_inspired_vqe": "iQCC-inspired",
    "iqcc_true_vqe": "iQCC true", "cb_vqe": "CB-VQE", "ucc_vqe": "UCC-VQE",
    "nn_ae_vqe": "NN-AE-VQE",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text())
    molecules, algorithms = config["molecules"], config["algorithms"]
    records, missing, inputs, outputs = {}, [], {}, []
    inputs[str(config_path.relative_to(root))] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    repair_manifest = root / "iqcc_repair_manifest.json"
    if repair_manifest.exists():
        inputs[repair_manifest.name] = hashlib.sha256(repair_manifest.read_bytes()).hexdigest()
    for mol in molecules:
        for algo in algorithms:
            path = root / "molecules" / mol / f"{algo}.json"
            if not path.exists():
                missing.append(f"{mol}/{algo}")
                continue
            record = json.loads(path.read_text())
            if (record["molecule_abbrev"], record["algorithm_name"]) != (mol, algo):
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            if record["n_qubits"] != config["fit"]["expected_n_qubits"]:
                raise ValueError(f"Unexpected qubit count: {path}")
            records[mol, algo] = record
            inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not records:
        raise ValueError("No saved checkpoints")
    style()
    fit = config["fit"]
    caption = (f"Rerun | {fit['expected_n_qubits']} qubits | {fit['active_space_basis'].upper()} | "
               f"seed {fit['random_seed']} | {len(records)}/{len(molecules)*len(algorithms)} saved fits")

    def save(fig, name, title, footnote):
        fig.suptitle(title, fontsize=16, fontweight="bold", y=.985)
        fig.text(.5, .945, caption, ha="center", fontsize=9)
        fig.text(.5, .015, footnote, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, .925))
        output = root / f"prof_rerun_benchmark_{name}.png"
        finish(fig, output)
        outputs.extend([output.name, output.with_suffix('.pdf').name])

    specs = [
        ("reference_error", "Signed energy difference from saved reference", "E − E_reference (mHa)",
         lambda r: 1000*(r['calculated_energy']-r['reference_energy']), "signed"),
        ("hf_difference", "Energy difference from Hartree–Fock", "E − E_HF (mHa)",
         lambda r: 1000*(r['calculated_energy']-r['hf_energy']) if r.get('hf_energy') is not None else np.nan, "signed"),
        ("runtime", "Recorded fit runtime", "Runtime (seconds; logarithmic color scale)",
         lambda r: r['runtime_seconds'], "log"),
        ("parameters", "Final variational parameter count", "Parameters", lambda r: r['n_parameters'], "linear"),
    ]
    for name, heading, units, value, scale in specs:
        data = np.full((len(molecules), len(algorithms)), np.nan)
        for i, mol in enumerate(molecules):
            for j, algo in enumerate(algorithms):
                if (mol, algo) in records:
                    data[i, j] = value(records[mol, algo])
        finite = data[np.isfinite(data)]
        if not finite.size:
            continue
        limit = max(float(np.max(np.abs(finite))), 1.)
        norm = SymLogNorm(linthresh=1, vmin=-limit, vmax=limit) if scale == "signed" else (
            LogNorm(vmin=max(float(finite[finite > 0].min()), 1e-6), vmax=max(float(finite.max()), 1.))
            if scale == "log" else None)
        cmap = plt.get_cmap('RdBu_r').copy() if scale == 'signed' else SUNSET.copy()
        cmap.set_bad('#e7e7eb')
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(algorithms)), [NAMES.get(a, a) for a in algorithms], rotation=30, ha='right')
        ax.set_yticks(range(len(molecules)), [m.upper() for m in molecules])
        for i in range(len(molecules)):
            for j in range(len(algorithms)):
                val = data[i, j]
                label = '—' if not np.isfinite(val) else f'{val:.2g}'
                rgba = im.cmap(im.norm(val)) if np.isfinite(val) else (1, 1, 1, 1)
                brightness = .299*rgba[0]+.587*rgba[1]+.114*rgba[2]
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        color='white' if brightness < .48 else '#20152b')
        fig.colorbar(im, ax=ax, pad=.02).set_label(units)
        note = 'Gray / dash: missing result or metric. Single seed; no uncertainty intervals.'
        if scale == 'signed':
            note += ' Color uses a symmetric log scale; linear within ±1 mHa.'
        if name == 'runtime':
            note += ' Concurrent-worker timings; excludes interrupted attempts.'
        save(fig, name, heading, note)

    colors = plt.get_cmap('tab10')
    for mol in molecules:
        fig, axes = plt.subplots(3, 4, figsize=(14, 9))
        for j, algo in enumerate(algorithms):
            ax = axes.flat[j]
            ax.set_title(NAMES.get(algo, algo), fontsize=10)
            r = records.get((mol, algo))
            if r is None:
                ax.text(.5, .5, 'No saved result', ha='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            history = np.array(r.get('convergence_history') or [], dtype=float)
            if history.size:
                ax.plot(np.arange(1, len(history)+1), 1000*(history-r['reference_energy']), color=colors(j), lw=1.5)
            else:
                ax.text(.5, .6, 'No recorded trajectory', ha='center', transform=ax.transAxes, fontsize=8)
            ax.axhline(1000*(r['calculated_energy']-r['reference_energy']), color=colors(j), ls=':', lw=1.2, label='Saved final')
            ax.axhline(0, color='#888888', lw=.7)
            ax.set_xlabel('Recorded history index')
            ax.set_ylabel('E − E_reference (mHa)')
            ax.grid(alpha=.15)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(algorithms):]:
            ax.remove()
        save(fig, f'{mol}_convergence', f'{mol.upper()} | Saved optimization histories',
             'Panels have independent scales. History indices are algorithm-specific, not comparable evaluation budgets. Dotted line: saved final energy.')
    manifest = dict(run=str(root), saved_fits=len(records), missing=missing, inputs_sha256=inputs,
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), outputs=outputs,
                    notes=['Saved fits are not necessarily converged.', 'Histories are plotted as recorded; no interpolation or retraining.'])
    (root / 'prof_rerun_benchmark_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f"Generated {len(outputs)//2} PNG/PDF pairs; {len(missing)} missing fits.")


if __name__ == '__main__':
    main()
