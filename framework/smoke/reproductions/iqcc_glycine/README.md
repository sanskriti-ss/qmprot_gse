# Glycine true-iQCC undershoot

The saved input contains the 27 ordered Pauli words and the normalized projected HF state from the published glycine geometry, STO-3G active space and a 12-to-4-qubit contextual reduction. The replay takes the first 26 words and preserves their order. Saving the projection also fixes the result of the time-limited contextual search.

Run from the repository root with its pinned scientific Python environment and upstream git history available locally.

```sh
framework/.venv/bin/python framework/smoke/reproductions/iqcc_glycine/replay.py
```

The script extracts each upstream revision into a temporary directory, loads its original solver and core modules, and independently diagonalizes the 16-by-16 Pauli sum. All four revisions produced −279.12260807171936 Ha after 20 iterations. The minimum eigenvalue of the same prefix was −279.11151473911696 Ha, an undershoot of 11.0933326024 mHa. Lightning emitted 3,240 complex-to-real casting warnings for each run. The recorded dependency versions, solver and input hashes, optimizer settings and shared 20-step history are in [observations.json](observations.json).

The snapshots cover the [May sweep implementation](https://github.com/sanskriti-ss/qmprot_gse/commit/2a304f2f72f8af2607d161356fdf0ed7c09c0449), the [October manuscript commit](https://github.com/sanskriti-ss/qmprot_gse/commit/4a23de80b29e4778f93fc754b198e3ffafb6b245), upstream [main](https://github.com/sanskriti-ss/qmprot_gse/commit/15e300abc82730e2aa57cf7e74b4200e1f02359d) and [experiments](https://github.com/sanskriti-ss/qmprot_gse/commit/6e4bea130ee81f2a1ffde8fe394e41670f69e715). The raw truncated commutator update entered the repository on [February 19](https://github.com/sanskriti-ss/qmprot_gse/commit/3755668c4013340df89872cd58a1b5c5351e39cb). The manuscript shows prefixes through ten terms. This 26-term case measures the historical implementations on one shared input with a common installed dependency stack.
