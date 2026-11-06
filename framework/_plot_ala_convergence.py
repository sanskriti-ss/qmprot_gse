from pathlib import Path
import csv
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = Path('/Users/sanskriti/Documents/GitHub/qmprot_gse/framework/experiments/results/trainability_batch/20260514_160311_batch_comparison/ala_comparison/convergence_traces.csv')
out_path = csv_path.with_name('ala_convergence_traces_plot.png')

rows = []
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        r['target_n_params'] = int(r['target_n_params'])
        r['iteration'] = int(r['iteration'])
        r['energy'] = float(r['energy'])
        rows.append(r)

if not rows:
    raise SystemExit('No rows found')

by_algo_target = defaultdict(list)
for r in rows:
    by_algo_target[(r['algorithm'], r['target_n_params'])].append(r)

algos = sorted({a for a, _ in by_algo_target})
targets = sorted({t for _, t in by_algo_target})
colors = plt.get_cmap('tab10')

fig, axes = plt.subplots(len(targets), 1, figsize=(10, 3.3 * len(targets)), sharex=False)
if len(targets) == 1:
    axes = [axes]

for ax, target in zip(axes, targets):
    for idx, algo in enumerate(algos):
        subset = sorted(by_algo_target.get((algo, target), []), key=lambda r: r['iteration'])
        if not subset:
            continue
        xs = [r['iteration'] for r in subset]
        ys = [r['energy'] for r in subset]
        ax.plot(xs, ys, marker='o', markersize=3, linewidth=1.8, label=algo, color=colors(idx))
    ax.set_title(f'ALA convergence trace at target_n_params = {target}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy (Ha)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle('ALA Trainability: Convergence Traces by Algorithm and Parameter Target', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(out_path, dpi=200)
print(out_path)
