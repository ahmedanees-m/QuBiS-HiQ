"""Layer 1: Deterministic orthogonal nucleotide encoding.
A=|00⟩, T/U=|01⟩, G=|10⟩, C=|11⟩ mapped to Ry rotations.


  bit=0  →  Ry(π/3):  |ψ⟩ = √3/2|0⟩ + 1/2|1⟩,  ⟨Z⟩ = +0.5
  bit=1  →  Ry(2π/3): |ψ⟩ = 1/2|0⟩ + √3/2|1⟩,  ⟨Z⟩ = −0.5

This gives all four nucleotides distinct (⟨Z₀⟩, ⟨Z₁⟩) signatures:
  A(00): (+0.5, +0.5)   T(01): (+0.5, −0.5)
  G(10): (−0.5, +0.5)   C(11): (−0.5, −0.5)
"""
import numpy as np
from qiskit import QuantumCircuit

NUC_MAP = {"A": (0, 0), "T": (0, 1), "U": (0, 1), "G": (1, 0), "C": (1, 1)}

# Ry rotation angles for each bit value
_RY_ANGLE = {0: np.pi / 3, 1: 2 * np.pi / 3}

def encode_nucleotide(qc: QuantumCircuit, nuc: str, q0: int, q1: int):
    """Apply Ry gates to encode a single nucleotide on 2 qubits."""
    bits = NUC_MAP[nuc.upper()]
    qc.ry(_RY_ANGLE[bits[0]], q0)
    qc.ry(_RY_ANGLE[bits[1]], q1)

def apply_encoding_layer(qc: QuantumCircuit, sequence: str):
    """Layer 1: Encode full sequence onto circuit. 2 qubits per nucleotide."""
    for i, nuc in enumerate(sequence.upper()):
        encode_nucleotide(qc, nuc, 2*i, 2*i+1)

def hamming_distance(nuc1: str, nuc2: str) -> int:
    """Compute Hamming distance between qubit encodings of two nucleotides."""
    b1, b2 = NUC_MAP[nuc1.upper()], NUC_MAP[nuc2.upper()]
    return sum(a != b for a, b in zip(b1, b2))

def is_transition(nuc1: str, nuc2: str) -> bool:
    purines, pyrimidines = {"A", "G"}, {"T", "C", "U"}
    n1, n2 = nuc1.upper(), nuc2.upper()
    return (n1 in purines and n2 in purines) or (n1 in pyrimidines and n2 in pyrimidines)
