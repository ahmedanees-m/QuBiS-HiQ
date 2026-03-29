"""Classical twin: compute same 49 features without quantum interference"""
import numpy as np
from typing import List, Tuple, Optional
from .encoding import NUC_MAP
from .santalucia import get_nn_dg, boltzmann_sigmoid, BETA
from .watson_crick import HBOND_COUNT

def classical_feature_vector(sequence: str,
                              stem_pairs: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """Compute the classical twin feature vector.
    
    Uses the same feature structure as quantum (local, NN, NNN, stem-pair)
    but computed from classical encodings without quantum interference.
    """
    seq = sequence.upper()
    N = len(seq)
    n_qubits = 2 * N
    stem_pairs = stem_pairs or []
    
    # Classical qubit values: deterministic from encoding
    z_values = np.zeros(n_qubits)
    for i, nuc in enumerate(seq):
        bits = NUC_MAP.get(nuc, (0, 0))
        z_values[2*i] = 1 - 2*bits[0]      # 0→+1, 1→-1
        z_values[2*i+1] = 1 - 2*bits[1]
    
    # Local magnetizations (trivially determined by encoding)
    z_local = z_values.copy()
    
    # NN correlators: product of adjacent qubit z-values,
    # weighted by stacking interaction strength
    zz_nn = np.zeros(n_qubits - 1)
    for i in range(n_qubits - 1):
        base_corr = z_values[i] * z_values[i+1]
        # Weight by stacking if crossing nucleotide boundary
        nuc_i, nuc_j = i // 2, (i+1) // 2
        if nuc_i != nuc_j and nuc_j < N:
            dinuc = seq[nuc_i] + seq[nuc_j]
            try:
                dg = get_nn_dg(dinuc)
                weight = boltzmann_sigmoid(dg) / np.pi  # Normalize to [0,1]
            except ValueError:
                weight = 0.5
            zz_nn[i] = base_corr * weight
        else:
            zz_nn[i] = base_corr
    
    # NNN correlators
    zz_nnn = np.zeros(n_qubits - 2)
    for i in range(n_qubits - 2):
        zz_nnn[i] = z_values[i] * z_values[i+2]
    
    # Stem-pair correlators
    zz_stem = np.zeros(len(stem_pairs))
    for idx, (a, b) in enumerate(stem_pairs):
        pair = (seq[a], seq[b])
        if pair in HBOND_COUNT:
            weight = HBOND_COUNT[pair] / 3.0
        else:
            weight = 0.0
        zz_stem[idx] = z_values[2*a] * z_values[2*b] * weight
    
    return np.concatenate([z_local, zz_nn, zz_nnn, zz_stem])
