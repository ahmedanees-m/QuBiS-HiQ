"""
topology_gated.py — Topology-Aware Circuit Builder
====================================================
Provides a drop-in replacement for build_circuit() that automatically
decides whether to apply the Watson-Crick entangling layer based on the
predicted secondary structure of the input sequence.

Design principle
----------------
The Watson-Crick layer (Layer 2) encodes intramolecular base-pairing
physics via CRZ gates between predicted stem pair positions. This layer
is only physically appropriate when:
  1. The sequence forms a stable intramolecular secondary structure
     (hairpin, stem-loop) in the experimental conditions of interest.
  2. The predicted stem pairs reflect the actual physics of the system.

For linear duplex sequences (e.g., PCR primers, siRNA guides, the
Oliveira 2020 benchmark duplexes), the sequence does NOT form an
intramolecular hairpin under hybridisation conditions. Applying the
WC layer in this context encodes phantom interactions, degrading
prediction accuracy (D3 ablation: +0.077 R² improvement when WC removed
for Oliveira linear duplexes).

Usage
-----
    from qubis_hiq.topology_gated import build_topology_gated_circuit

    # Automatically selects circuit topology based on predicted structure
    qc = build_topology_gated_circuit("GCGCTTTTGCGC")

    # Explicit override
    qc = build_topology_gated_circuit("ATGCATGC", force_skip_wc=True)

    # Topology classification only
    from qubis_hiq.topology_gated import classify_topology
    topo = classify_topology("GCGCTTTTGCGC")  # → {'is_hairpin': True, ...}

ViennaRNA vs heuristic
----------------------
If ViennaRNA is installed (conda install -c bioconda viennarna), MFE-based
thermodynamic classification is used. Otherwise, the built-in palindromic
heuristic is applied. The heuristic is exact for whether the WC layer
will apply CRZ gates (it uses the same predict_structure() function).
"""

from __future__ import annotations

import warnings
from typing import Optional, List, Tuple

import numpy as np
from qiskit import QuantumCircuit

from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.vienna_interface import predict_structure

# ── ViennaRNA import ──────────────────────────────────────────────────────────
try:
    import RNA as _RNA
    _VIENNA_AVAILABLE = True
except ImportError:
    _VIENNA_AVAILABLE = False

# Default MFE threshold (kcal/mol). Sequences with ViennaRNA MFE below this
# are considered hairpin-forming and receive the Watson-Crick layer.
DEFAULT_MFE_THRESHOLD: float = -2.0


def classify_topology(
    sequence: str,
    mfe_threshold: float = DEFAULT_MFE_THRESHOLD,
) -> dict:
    """
    Classify a sequence as hairpin-forming or linear.

    Parameters
    ----------
    sequence : str
        DNA or RNA sequence (ACGTU alphabet).
    mfe_threshold : float
        ViennaRNA MFE cutoff (kcal/mol). Only used when ViennaRNA is installed.
        Sequences with MFE < threshold are classified as hairpin-forming.

    Returns
    -------
    dict with keys:
        is_hairpin : bool
        stem_pairs : list of (i, j) tuples from predict_structure()
        n_stem_pairs : int
        mfe : float or None   (None if ViennaRNA not available)
        mfe_structure : str or None
        method : str          ('viennarna' or 'heuristic')
    """
    sequence = sequence.upper().replace("U", "T")
    dot_bracket, stem_pairs = predict_structure(sequence)

    mfe = None
    mfe_structure = None
    method = "heuristic"

    if _VIENNA_AVAILABLE:
        rna_seq = sequence.replace("T", "U")
        result = _RNA.fold(rna_seq)
        mfe = float(result[1])
        mfe_structure = result[0]
        method = "viennarna"
        is_hairpin = mfe < mfe_threshold
    else:
        # Heuristic: sequence is hairpin-like if predict_structure() finds
        # at least one stem pair — the same criterion that determines whether
        # the Watson-Crick layer will apply any CRZ gates at all.
        is_hairpin = len(stem_pairs) > 0

    return {
        "is_hairpin": is_hairpin,
        "stem_pairs": stem_pairs,
        "n_stem_pairs": len(stem_pairs),
        "mfe": mfe,
        "mfe_structure": mfe_structure,
        "method": method,
    }


def build_topology_gated_circuit(
    sequence: str,
    trainable_params: Optional[np.ndarray] = None,
    mfe_threshold: float = DEFAULT_MFE_THRESHOLD,
    force_skip_wc: Optional[bool] = None,
    include_measurement: bool = True,
    random_angles: bool = False,
    verbose: bool = False,
) -> QuantumCircuit:
    """
    Build a QuBiS-HiQ circuit with topology-appropriate WC layer gating.

    The Watson-Crick layer is applied only when the sequence is predicted
    to form a hairpin/stem-loop structure. For linear sequences, the WC
    layer is skipped, yielding the stacking-only circuit that performs
    optimally on linear duplex Tm prediction (Exp X2, R²=0.941 combined).

    Parameters
    ----------
    sequence : str
        DNA sequence (5'→3'), ACGT alphabet.
    trainable_params : np.ndarray, optional
        12-element array for trainable layer. Defaults to zeros.
    mfe_threshold : float
        MFE cutoff for hairpin classification (ViennaRNA only).
    force_skip_wc : bool, optional
        If provided, overrides topology-based gating:
          True  → always skip WC (stacking-only for all sequences)
          False → always apply WC (original behaviour)
    include_measurement : bool
        Add measurement gates to the circuit.
    random_angles : bool
        Use random gate angles (ablation mode).
    verbose : bool
        Print topology classification.

    Returns
    -------
    QuantumCircuit
        QuBiS-HiQ circuit with (2 × len(sequence)) qubits.
    """
    if trainable_params is None:
        trainable_params = np.zeros(12)

    topo = classify_topology(sequence, mfe_threshold)

    if force_skip_wc is not None:
        skip_wc = force_skip_wc
        reason = "forced"
    else:
        skip_wc = not topo["is_hairpin"]
        reason = topo["method"]

    if verbose:
        print(f"  {sequence}: is_hairpin={topo['is_hairpin']} "
              f"(method={reason}, stems={topo['n_stem_pairs']}) "
              f"→ skip_wc={skip_wc}")

    return build_circuit(
        sequence,
        stem_pairs=topo["stem_pairs"],
        trainable_params=trainable_params,
        include_measurement=include_measurement,
        skip_wc=skip_wc,
        random_angles=random_angles,
    )


def batch_classify(
    sequences: List[str],
    mfe_threshold: float = DEFAULT_MFE_THRESHOLD,
    n_jobs: int = 1,
) -> List[dict]:
    """
    Classify a list of sequences. Uses multiprocessing when n_jobs > 1.

    Parameters
    ----------
    sequences : list of str
    mfe_threshold : float
    n_jobs : int
        Number of parallel workers (-1 = all CPUs).

    Returns
    -------
    list of dict (same structure as classify_topology())
    """
    if n_jobs == 1:
        return [classify_topology(s, mfe_threshold) for s in sequences]

    from multiprocessing import Pool, cpu_count
    workers = cpu_count() if n_jobs == -1 else n_jobs

    def _classify(args):
        seq, thresh = args
        return classify_topology(seq, thresh)

    with Pool(workers) as pool:
        return pool.map(_classify, [(s, mfe_threshold) for s in sequences])
