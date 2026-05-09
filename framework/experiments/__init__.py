"""
VQE Experiments Package
=======================

This package contains stand-alone, reproducible experiments built on top of
the framework's ``core``, ``algorithms``, ``active_space_truncation`` and
``contextual_subspace`` modules.

Experiments
-----------

* :mod:`experiments.noise_resilience`
    How much does the optimised ansatz drift when the same VQE problem is
    optimised under different PennyLane noise channels at varying strengths?
    Measures energy drift and L2/cosine drift of the optimal parameter
    vector relative to a noiseless baseline.

* :mod:`experiments.barren_plateau`
    Quantifies the barren-plateau effect for a hardware-efficient ansatz at
    several depths and several initialisation strategies, then runs the VQE
    optimisation to compare convergence speed/quality from each starting
    region.

* :mod:`experiments.trainability`
    Compares the trainability of an *adaptive* ansatz
    (``qubit_adapt_vqe``) against a *fixed* hardware-efficient ansatz at
    matched parameter counts (default 10/50/100).  Measures the *true*
    number of cost-function evaluations consumed by each algorithm
    (rather than its self-reported ``n_iterations``), plus wall time
    and final energy.

* :mod:`experiments.accuracy_vs_params`
    Sweeps the number of variational parameters via ``qubit_adapt_vqe``
    and reports the achieved energy at each requested ``k``.  A single
    ADAPT run with ``max_operators = max(k_list)`` is sufficient to
    recover the entire energy-vs-k curve, so the modular ``k_list`` is
    free.

All experiments share a small helper (:mod:`experiments._common`) that loads
a small qubit Hamiltonian for any molecule found in
``framework/datasets2``, attaches a cost-function-call counter to a VQE
instance, and creates the date- and experiment-stamped output directory
of the form
``framework/experiments/results/<experiment>/<timestamp>_<molecule>_<algo>/``.
Experiments 1-2 run in a couple of minutes on 4 qubits; experiments 3-4
default to a 12-qubit active-space Hamiltonian (no CS, since CS makes
adaptive ansatze exit at k=0) and take several minutes per ADAPT run.
"""
