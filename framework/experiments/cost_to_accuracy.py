"""
Cost-to-accuracy analysis for the trainability experiment
==========================================================

The fair way to compare algorithms is the *cost to reach a fixed accuracy*,
not the total evaluations a run happened to spend.  A run that stops at its
iteration cap while still ~1 Ha above CASCI looks "cheap" but never solved
the problem.  This module turns trainability rows into

* one record per (molecule, algorithm, target) with
  ``evals_to_<threshold>`` (blank = never reached -> *censored*),
* per-(algorithm, target) summaries (how many molecules reached each
  accuracy, median cost among those that did),
* figures and a plain-language markdown report,
* a re-analysis path for older result folders that only stored final
  energies (``analyze_existing``): there the cost is an UPPER bound (the
  run total, when the final error is under the threshold) or censored.

Threshold semantics: error = E - E_CASCI (Ha); chemical accuracy = 1.6 mHa.
Molecules where HF is already within 1.6 mHa of CASCI are flagged
(``hf_trivial``) and excluded from the headline statistics, because there
is nothing for an ansatz to demonstrate.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from experiments._common import (
    ACCURACY_LADDER_HA,
    CHEMICAL_ACCURACY_HA,
    threshold_label,
)

logger = logging.getLogger(__name__)

ALGO_STYLE = {
    "qubit_adapt_vqe": ("#7c3aed", "o", "Qubit-ADAPT"),
    "adapt_vqe": ("#2563eb", "D", "ADAPT"),
    "hardware_efficient_vqe": ("#f97316", "s", "HE-VQE"),
}

BASE_FIELDS = [
    "molecule", "algorithm", "target_n_params", "actual_n_params", "param_match",
    "n_outer_iters", "n_cost_evals", "n_callback_evals", "eval_budget",
    "runtime_seconds", "final_energy", "error_vs_casci", "best_error_vs_casci",
    "hf_error_vs_casci", "hf_trivial", "converged_below_threshold",
    "stopped_at_accuracy", "init_label", "bounds_only", "notes",
]


def label_list(thresholds: Sequence[float] = ACCURACY_LADDER_HA) -> List[str]:
    return [threshold_label(t) for t in thresholds]


def record_fields(thresholds: Sequence[float] = ACCURACY_LADDER_HA) -> List[str]:
    labels = label_list(thresholds)
    return (
        BASE_FIELDS[:-1]
        + [f"evals_to_{l}" for l in labels]
        + [f"reached_{l}" for l in labels]
        + ["notes"]
    )


def make_record(
    molecule: str,
    row,
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
) -> Dict[str, object]:
    """Flatten a TrainabilityRow (duck-typed) into a CSV-ready dict."""
    hf_err = row.hf_error_vs_casci
    rec: Dict[str, object] = {
        "molecule": molecule,
        "algorithm": row.algorithm,
        "target_n_params": row.target_n_params,
        "actual_n_params": row.actual_n_params,
        "param_match": row.param_match,
        "n_outer_iters": row.n_outer_iters,
        "n_cost_evals": row.n_cost_evals,
        "n_callback_evals": row.n_callback_evals,
        "eval_budget": row.eval_budget,
        "runtime_seconds": f"{row.runtime_seconds:.4f}",
        "final_energy": f"{row.final_energy:.10f}",
        "error_vs_casci": f"{row.error_vs_casci:.10f}",
        "best_error_vs_casci": f"{row.best_error_vs_casci:.10f}",
        "hf_error_vs_casci": f"{hf_err:.10f}",
        "hf_trivial": int(bool(np.isfinite(hf_err) and abs(hf_err) < CHEMICAL_ACCURACY_HA)),
        "converged_below_threshold": int(row.converged_below_threshold),
        "stopped_at_accuracy": int(row.stopped_at_accuracy),
        "init_label": row.init_label,
        "bounds_only": int(row.bounds_only),
        "notes": row.notes,
    }
    for t in thresholds:
        lab = threshold_label(t)
        v = row.evals_to_threshold.get(lab)
        rec[f"evals_to_{lab}"] = "" if v is None else int(v)
        rec[f"reached_{lab}"] = int(v is not None)
    return rec


def write_records_csv(path: Path, records: Sequence[Dict[str, object]],
                      thresholds: Sequence[float] = ACCURACY_LADDER_HA) -> None:
    fields = record_fields(thresholds)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


def write_eval_traces_csv(path: Path, traces: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "molecule", "algorithm", "target_n_params", "eval_index",
            "best_error_vs_casci",
        ])
        w.writeheader()
        for t in traces:
            w.writerow(t)


def traces_from_rows(molecule: str, rows) -> List[Dict[str, object]]:
    out = []
    for r in rows:
        for idx, err in r.error_trace:
            out.append({
                "molecule": molecule, "algorithm": r.algorithm,
                "target_n_params": r.target_n_params, "eval_index": idx,
                "best_error_vs_casci": f"{err:.10f}",
            })
    return out


# ──────────────────────────────────────────────────────────────────────────
# Summaries
# ──────────────────────────────────────────────────────────────────────────

def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def summarize(
    records: Sequence[Dict[str, object]],
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
    exclude_hf_trivial: bool = True,
) -> List[Dict[str, object]]:
    """Per (algorithm, target) statistics, one row each."""
    labels = label_list(thresholds)
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for r in records:
        groups.setdefault((str(r["algorithm"]), int(r["target_n_params"])), []).append(r)

    out = []
    for (algo, target), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n_all = len(rs)
        used = [r for r in rs if not (exclude_hf_trivial and int(r["hf_trivial"]))]
        row: Dict[str, object] = {
            "algorithm": algo,
            "target_n_params": target,
            "n_molecules": len(used),
            "n_excluded_hf_trivial": n_all - len(used),
            "n_param_exact": sum(1 for r in used if r["param_match"] == "exact"),
            "median_actual_n_params": float(np.median([_f(r["actual_n_params"]) for r in used])) if used else float("nan"),
            "median_evals_spent": float(np.median([_f(r["n_cost_evals"]) for r in used])) if used else float("nan"),
            "median_final_error_mha": float(np.median([_f(r["error_vs_casci"]) for r in used])) * 1e3 if used else float("nan"),
            "bounds_only": int(any(int(r.get("bounds_only", 0) or 0) for r in used)),
        }
        for lab in labels:
            hits = [_f(r[f"evals_to_{lab}"]) for r in used if str(r[f"evals_to_{lab}"]) != ""]
            row[f"n_reached_{lab}"] = len(hits)
            row[f"median_evals_to_{lab}"] = float(np.median(hits)) if hits else float("nan")
        out.append(row)
    return out


def write_summary_csv(path: Path, summary: Sequence[Dict[str, object]]) -> None:
    if not summary:
        return
    fields = list(summary[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow(r)


def _fmt_int(v: float) -> str:
    return "never" if not np.isfinite(v) else f"{v:,.0f}"


def render_console(summary: Sequence[Dict[str, object]],
                   thresholds: Sequence[float] = ACCURACY_LADDER_HA) -> str:
    labels = label_list(thresholds)
    chem = threshold_label(CHEMICAL_ACCURACY_HA)
    bounds = any(int(s["bounds_only"]) for s in summary)
    head = (f"{'algorithm':<24}{'target':>7}{'actual':>7}{'exact':>8}"
            f"{'evals spent':>13}{'final err mHa':>15}"
            + "".join(f"{'reached ' + l:>18}" for l in labels)
            + f"{'median evals->' + chem:>24}")
    lines = [head, "-" * len(head)]
    for s in summary:
        n = s["n_molecules"]
        lines.append(
            f"{s['algorithm']:<24}{s['target_n_params']:>7}{s['median_actual_n_params']:>7.0f}"
            f"{str(s['n_param_exact']) + '/' + str(n):>8}"
            f"{_fmt_int(s['median_evals_spent']):>13}{s['median_final_error_mha']:>15.2f}"
            + "".join(f"{str(s['n_reached_' + l]) + '/' + str(n):>18}" for l in labels)
            + f"{_fmt_int(s['median_evals_to_' + chem]):>24}"
        )
    note = ("\n(final-energy-only data: 'median evals' for reached runs is an UPPER BOUND = "
            "run total)" if bounds else "")
    return "\n".join(lines) + note


def render_markdown(
    summary: Sequence[Dict[str, object]],
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
    title: str = "Cost to reach a fixed accuracy",
) -> str:
    labels = label_list(thresholds)
    chem = threshold_label(CHEMICAL_ACCURACY_HA)
    bounds = any(int(s["bounds_only"]) for s in summary)
    excl = max((int(s["n_excluded_hf_trivial"]) for s in summary), default=0)
    out = [
        f"# {title}",
        "",
        "Error = E - E_CASCI. Chemical accuracy = 1.6 mHa. A run that never reaches a "
        "threshold is *censored*: we report how many molecules reached it, and the "
        "median cost only among those that did.",
        "",
        f"Molecules where HF is already within chemical accuracy are excluded ({excl}).",
    ]
    if bounds:
        out += ["", "> **Data note:** these results only stored final energies, so the cost "
                    "for a run that reached a threshold is an upper bound (the run total)."]
    out += ["", "| algorithm | target params | actual (median) | exact match | median evals spent | "
                "median final error (mHa) | "
            + " | ".join(f"reached {l}" for l in labels)
            + f" | median evals to {chem} |",
            "|" + "---|" * (7 + len(labels))]
    for s in summary:
        n = s["n_molecules"]
        out.append(
            f"| {s['algorithm']} | {s['target_n_params']} | {s['median_actual_n_params']:.0f} | "
            f"{s['n_param_exact']}/{n} | {_fmt_int(s['median_evals_spent'])} | "
            f"{s['median_final_error_mha']:.2f} | "
            + " | ".join(f"{s['n_reached_' + l]}/{n}" for l in labels)
            + f" | {_fmt_int(s['median_evals_to_' + chem])} |"
        )
    out += [
        "",
        "## How to read this",
        "",
        "* *median evals spent* is what the old comparison used; it is only meaningful "
        "next to the final error column.",
        f"* *reached {chem}* is the fair comparison: the fraction of molecules where the "
        "algorithm actually got to chemical accuracy.",
        f"* *median evals to {chem}* is the cost to get there (only over runs that did).",
        "",
    ]
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────

def _style(algo: str):
    return ALGO_STYLE.get(algo, ("#555555", "o", algo))


def plot_cost_to_accuracy_single(
    records: Sequence[Dict[str, object]],
    path: Path,
    molecule: str,
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr = [t for t in thresholds if t <= 1e-2] or list(thresholds)[-1:]
    algos = sorted({str(r["algorithm"]) for r in records})
    targets = sorted({int(r["target_n_params"]) for r in records})
    fig, axes = plt.subplots(1, len(thr), figsize=(5.8 * len(thr), 4.8), squeeze=False)
    w = 0.8 / max(len(algos), 1)
    for ax, t in zip(axes[0], thr):
        lab = threshold_label(t)
        for i, algo in enumerate(algos):
            color, _, name = _style(algo)
            for j, tgt in enumerate(targets):
                rec = next((r for r in records if r["algorithm"] == algo
                            and int(r["target_n_params"]) == tgt), None)
                if rec is None:
                    continue
                x = j + i * w
                hit = str(rec[f"evals_to_{lab}"]) != ""
                spent = _f(rec["n_cost_evals"])
                if hit:
                    ax.bar(x, _f(rec[f"evals_to_{lab}"]), w * 0.92, color=color,
                           label=name if j == 0 else None)
                else:
                    ax.bar(x, spent, w * 0.92, facecolor="white", edgecolor=color,
                           hatch="//", label=name if j == 0 else None)
                    ax.annotate(f"not reached\n> {spent:,.0f}\n(err {_f(rec['error_vs_casci']) * 1e3:.0f} mHa)",
                                (x, spent), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(np.arange(len(targets)) + w * (len(algos) - 1) / 2)
        ax.set_xticklabels([
            f"target {tgt}\n" + ", ".join(
                f"{_style(a)[2]}: {next(str(r['actual_n_params']) for r in records if r['algorithm'] == a and int(r['target_n_params']) == tgt)}"
                for a in algos
                if any(r["algorithm"] == a and int(r["target_n_params"]) == tgt for r in records)
            ) + " params"
            for tgt in targets
        ], fontsize=7.5)
        ax.set_yscale("log")
        ax.set_ylabel("cost-function evaluations")
        ax.set_title(f"Evals to reach error <= {lab}")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.margins(y=0.35)
        ax.legend(fontsize=8)
    fig.suptitle(f"{molecule}: cost to reach a fixed accuracy "
                 "(hatched = never reached; bar height = budget spent)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_vs_cost(
    traces: Sequence[Dict[str, object]],
    records: Sequence[Dict[str, object]],
    path: Path,
    molecule: str,
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not traces:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    keys = sorted({(str(t["algorithm"]), int(t["target_n_params"])) for t in traces},
                  key=lambda k: (k[0], k[1]))
    targets = sorted({k[1] for k in keys})
    alphas = {t: 0.45 + 0.55 * (i + 1) / len(targets) for i, t in enumerate(targets)}
    for algo, tgt in keys:
        pts = sorted(((int(t["eval_index"]), _f(t["best_error_vs_casci"]))
                      for t in traces
                      if t["algorithm"] == algo and int(t["target_n_params"]) == tgt))
        xs = [p[0] for p in pts]
        ys = [max(p[1], 1e-9) for p in pts]
        color, _, name = _style(algo)
        ax.step(xs, ys, where="post", color=color, alpha=alphas[tgt],
                label=f"{name}, {tgt} params")
    for t in thresholds:
        ax.axhline(t, color="black", linestyle="--" if t == CHEMICAL_ACCURACY_HA else ":",
                   linewidth=1.0)
        ax.text(ax.get_xlim()[0], t, f" {threshold_label(t)}", va="bottom", fontsize=7)
    hf = next((_f(r["hf_error_vs_casci"]) for r in records), float("nan"))
    if np.isfinite(hf):
        ax.axhline(abs(hf), color="grey", linestyle="-.", linewidth=1.0,
                   label=f"HF error ({abs(hf) * 1e3:.1f} mHa)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cost-function evaluations")
    ax.set_ylabel(r"best error so far  $E - E_{CASCI}$  (Ha)")
    ax.set_title(f"{molecule}: accuracy vs cost (curves that never cross the 1.6 mHa "
                 "line did not reach chemical accuracy)", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_to_accuracy_all(
    records: Sequence[Dict[str, object]],
    path: Path,
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
    exclude_hf_trivial: bool = True,
) -> None:
    """Strip plot: filled dot = evals to chemical accuracy, open triangle = never
    reached (placed at the budget spent, i.e. a lower bound)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = threshold_label(CHEMICAL_ACCURACY_HA)
    recs = [r for r in records if not (exclude_hf_trivial and int(r["hf_trivial"]))]
    if not recs:
        return
    algos = sorted({str(r["algorithm"]) for r in recs})
    targets = sorted({int(r["target_n_params"]) for r in recs})
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    offs = np.linspace(-0.2, 0.2, max(len(algos), 1)) if len(algos) > 1 else [0.0]
    for i, algo in enumerate(algos):
        color, marker, name = _style(algo)
        first_hit = first_miss = True
        for j, tgt in enumerate(targets):
            sub = [r for r in recs if r["algorithm"] == algo and int(r["target_n_params"]) == tgt]
            hits = [r for r in sub if str(r[f"evals_to_{lab}"]) != ""]
            miss = [r for r in sub if str(r[f"evals_to_{lab}"]) == ""]
            if hits:
                ax.scatter(j + offs[i] + rng.uniform(-0.04, 0.04, len(hits)),
                           [_f(r[f"evals_to_{lab}"]) for r in hits], s=28, color=color,
                           marker=marker, alpha=0.75,
                           label=f"{name}: reached" if first_hit else None)
                first_hit = False
            if miss:
                ax.scatter(j + offs[i] + rng.uniform(-0.04, 0.04, len(miss)),
                           [max(_f(r["n_cost_evals"]), 1) for r in miss], s=40,
                           facecolors="none", edgecolors=color, marker="^", linewidths=1.3,
                           label=f"{name}: NOT reached (budget spent, lower bound)"
                           if first_miss else None)
                first_miss = False
            ax.text(j + offs[i], 0.98, f"{len(hits)}/{len(sub)}", color=color, fontsize=8,
                    ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"{t} params" for t in targets])
    ax.set_yscale("log")
    ax.set_ylabel("cost-function evaluations")
    ax.set_title(f"Cost to reach chemical accuracy ({lab} vs CASCI) across "
                 f"{len({r['molecule'] for r in recs})} molecules\n"
                 "numbers on top = molecules that reached it / total; "
                 "molecules where HF is already within chemical accuracy are excluded",
                 fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# One-call bundle + legacy re-analysis
# ──────────────────────────────────────────────────────────────────────────

def save_cost_to_accuracy_outputs(
    out_dir: Path,
    records: Sequence[Dict[str, object]],
    traces: Sequence[Dict[str, object]],
    *,
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
    prefix: str = "",
    single_molecule: Optional[str] = None,
    plots: bool = True,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    summary = summarize(records, thresholds)
    if prefix:
        csv_p = out_dir / f"{prefix}cost_to_accuracy_summary.csv"
        write_summary_csv(csv_p, summary)
        paths["summary_csv"] = csv_p
    md = out_dir / "cost_to_accuracy_report.md"
    md.write_text(render_markdown(
        summary, thresholds,
        title=("Cost to reach a fixed accuracy" + (f" -- {single_molecule}" if single_molecule else "")),
    ), encoding="utf-8")
    paths["report"] = md
    if traces:
        tp = out_dir / ("all_molecules_eval_traces.csv" if prefix else "eval_traces.csv")
        write_eval_traces_csv(tp, traces)
        paths["eval_traces"] = tp
    if plots:
        try:
            if single_molecule:
                plot_cost_to_accuracy_single(records, out_dir / "cost_to_accuracy.png",
                                             single_molecule, thresholds)
                plot_error_vs_cost(traces, records, out_dir / "error_vs_cost.png",
                                   single_molecule, thresholds)
            else:
                plot_cost_to_accuracy_all(records, out_dir / "all_molecules_cost_to_accuracy.png",
                                          thresholds)
        except Exception as exc:  # plotting must never kill a long run
            logger.warning("cost-to-accuracy plotting failed: %s", exc)
    return paths


def analyze_existing(
    results_root: Path,
    out_dir: Optional[Path] = None,
    thresholds: Sequence[float] = ACCURACY_LADDER_HA,
) -> Dict[str, object]:
    """Re-analyse a finished batch folder (``<mol>/results.csv`` + ``run_config.json``).

    New-format folders (with ``evals_to_*`` columns) are used as-is.  Old folders
    only have final energies, so a run counts as *reached* iff its final error is
    under the threshold, and the cost reported for it is the run total (an upper
    bound).  Everything else is censored.
    """
    results_root = Path(results_root)
    out_dir = Path(out_dir) if out_dir else results_root / "reanalysis_cost_to_accuracy"
    labels = label_list(thresholds)
    records: List[Dict[str, object]] = []
    for mol_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        res, cfg_p = mol_dir / "results.csv", mol_dir / "run_config.json"
        if not res.is_file() or not cfg_p.is_file():
            continue
        cfg = json.loads(cfg_p.read_text())
        hf, casci = cfg.get("hf_energy"), cfg.get("casci_energy")
        hf_err = (hf - casci) if (hf is not None and casci is not None) else float("nan")
        with open(res, newline="") as fh:
            for rec in csv.DictReader(fh):
                err = _f(rec["error_vs_casci"])
                has_new = f"evals_to_{labels[-1]}" in rec
                actual, target = int(rec["actual_n_params"]), int(rec["target_n_params"])
                r: Dict[str, object] = {
                    "molecule": mol_dir.name,
                    "algorithm": rec["algorithm"],
                    "target_n_params": target,
                    "actual_n_params": actual,
                    "param_match": rec.get("param_match") or (
                        "exact" if actual == target else
                        ("stopped_early" if actual < target else "rounded_up")),
                    "n_outer_iters": rec.get("n_outer_iters", ""),
                    "n_cost_evals": int(rec["n_cost_evals"]),
                    "n_callback_evals": rec.get("n_callback_evals", ""),
                    "eval_budget": rec.get("eval_budget", ""),
                    "runtime_seconds": rec.get("runtime_seconds", ""),
                    "final_energy": rec.get("final_energy", ""),
                    "error_vs_casci": err,
                    "best_error_vs_casci": rec.get("best_error_vs_casci", err),
                    "hf_error_vs_casci": hf_err,
                    "hf_trivial": int(np.isfinite(hf_err) and abs(hf_err) < CHEMICAL_ACCURACY_HA),
                    "converged_below_threshold": rec.get("converged_below_threshold", ""),
                    "stopped_at_accuracy": rec.get("stopped_at_accuracy", ""),
                    "init_label": rec.get("init_label", ""),
                    "bounds_only": 0 if has_new else 1,
                    "notes": rec.get("notes", ""),
                }
                for t, lab in zip(thresholds, labels):
                    if has_new:
                        v = rec.get(f"evals_to_{lab}", "")
                    else:
                        v = int(rec["n_cost_evals"]) if err <= t else ""
                    r[f"evals_to_{lab}"] = v
                    r[f"reached_{lab}"] = int(v != "")
                records.append(r)
    if not records:
        raise FileNotFoundError(f"No <molecule>/results.csv found under {results_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(out_dir / "all_molecules_records.csv", records, thresholds)
    paths = save_cost_to_accuracy_outputs(out_dir, records, [], thresholds=thresholds,
                                          prefix="all_molecules_")
    summary = summarize(records, thresholds)
    return {"records": records, "summary": summary, "output_dir": out_dir, **paths}
