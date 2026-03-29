"""Layer 2: Watson-Crick complementarity via CRZ gates.
Rotation angle proportional to H-bond count: A-T=2/3π, G-C=π."""
import numpy as np
from qiskit import QuantumCircuit
from typing import List, Tuple

WC_PAIRS = {("A","T"), ("T","A"), ("A","U"), ("U","A"), ("G","C"), ("C","G")}
HBOND_COUNT = {("A","T"): 2, ("T","A"): 2, ("A","U"): 2, ("U","A"): 2,
               ("G","C"): 3, ("C","G"): 3}

def get_wc_angle(nuc_i: str, nuc_j: str) -> float:
    pair = (nuc_i.upper(), nuc_j.upper())
    if pair not in HBOND_COUNT:
        raise ValueError(f"Not a Watson-Crick pair: {pair}")
    return (HBOND_COUNT[pair] / 3.0) * np.pi

def apply_watson_crick_layer(qc: QuantumCircuit, sequence: str,
                              pairs: List[Tuple[int, int]]):
    """Layer 2: Apply CRZ gates between stem-paired nucleotides.
    pairs: list of (i, j) nucleotide position indices from ViennaRNA."""
    seq = sequence.upper()
    for i, j in pairs:
        ni, nj = seq[i], seq[j]
        if (ni, nj) not in WC_PAIRS:
            continue
        theta = get_wc_angle(ni, nj)
        # CRZ on first qubits of each nucleotide pair
        qc.crz(theta, 2*i, 2*j)
        # CRZ on second qubits
        qc.crz(theta, 2*i+1, 2*j+1)
