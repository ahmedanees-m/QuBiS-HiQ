"""Full 6-layer quantum biomimetic circuit assembly."""
import numpy as np
from qiskit import QuantumCircuit
from typing import List, Tuple, Optional
from .encoding import apply_encoding_layer
from .watson_crick import apply_watson_crick_layer
from .stacking import apply_stacking_layer
from .trainable import apply_trainable_layer
from .interference import apply_hadamard_layer, apply_measurement

def build_circuit(sequence: str,
                  stem_pairs: Optional[List[Tuple[int, int]]] = None,
                  trainable_params: Optional[np.ndarray] = None,
                  include_measurement: bool = True,
                  skip_wc: bool = False,
                  skip_stacking: bool = False,
                  random_angles: bool = False) -> QuantumCircuit:
    """Build the complete 6-layer QuBiS-HiQ circuit.
    
    Args:
        sequence: Nucleotide sequence (5'→3')
        stem_pairs: List of (i,j) base pair indices from ViennaRNA
        trainable_params: Concrete parameter values for Layer 4 (None = parameterized)
        include_measurement: Whether to add measurement gates
        skip_wc: Ablation — skip Watson-Crick layer
        skip_stacking: Ablation — skip stacking layer
        random_angles: Ablation — use random gate angles
    
    Returns:
        QuantumCircuit
    """
    N = len(sequence)
    n_qubits = 2 * N
    qc = QuantumCircuit(n_qubits)
    stem_pairs = stem_pairs or []
    
    # Layer 1: Deterministic encoding
    apply_encoding_layer(qc, sequence)
    qc.barrier()
    
    # Layer 2: Watson-Crick complementarity
    if not skip_wc and stem_pairs:
        if random_angles:
            # Random CRZ angles for ablation
            for i, j in stem_pairs:
                theta = np.random.uniform(0, np.pi)
                qc.crz(theta, 2*i, 2*j)
                qc.crz(theta, 2*i+1, 2*j+1)
        else:
            apply_watson_crick_layer(qc, sequence, stem_pairs)
        qc.barrier()
    
    # Layer 3: Nearest-neighbor stacking
    if not skip_stacking:
        if random_angles:
            seq = sequence.upper()
            for k in range(N - 1):
                theta = np.random.uniform(0, np.pi)
                # FIX (v2): use CX+Ry instead of RZZ (matches apply_stacking_layer)
                qc.cx(2*k, 2*(k+1))
                qc.ry(theta, 2*(k+1))
                qc.cx(2*k+1, 2*(k+1)+1)
                qc.ry(theta, 2*(k+1)+1)
        else:
            apply_stacking_layer(qc, sequence)
        qc.barrier()

    # Layer 4: Trainable local rotations
    if trainable_params is not None:
        if random_angles:
            trainable_params = np.random.uniform(-np.pi, np.pi, len(trainable_params))
        apply_trainable_layer(qc, sequence, trainable_params)
    else:
        # Return ParameterVector for variational optimization (as documented)
        apply_trainable_layer(qc, sequence, None)
    qc.barrier()

    # Layer 5: Hadamard interference — REMOVED in v2.
    # With Ry encoding the qubits already have non-zero Z-axis components.
    # Applying H here would rotate them into the X-basis making ⟨Z⟩ → 0.
    # apply_hadamard_layer(qc, n_qubits)   # <-- intentionally disabled

    # Layer 6: Measurement
    if include_measurement:
        apply_measurement(qc)
    
    return qc

def build_ablation_variants(sequence: str,
                            stem_pairs: Optional[List[Tuple[int, int]]] = None,
                            params: Optional[np.ndarray] = None):
    """Build all 5 circuit variants for ablation study.
    
    Returns:
        Dict mapping variant name to QuantumCircuit
    """
    params = params if params is not None else np.zeros(12)
    return {
        "full": build_circuit(sequence, stem_pairs, params),
        "no_wc": build_circuit(sequence, stem_pairs, params, skip_wc=True),
        "no_stacking": build_circuit(sequence, stem_pairs, params, skip_stacking=True),
        "random": build_circuit(sequence, stem_pairs, params, random_angles=True),
    }
