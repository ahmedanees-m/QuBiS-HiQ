"""Extract biologically meaningful feature vectors from circuit measurement outcomes."""
import numpy as np
from typing import List, Tuple, Dict, Optional

def bitstring_to_array(bitstring: str) -> np.ndarray:
    """Convert bitstring to ±1 array (0→+1, 1→-1)."""
    return np.array([1 - 2*int(b) for b in bitstring])

def extract_feature_vector(counts: Dict[str, int], n_nucleotides: int,
                           stem_pairs: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """Extract the full feature vector from measurement counts.
    
    Features:
    - 2N local magnetizations ⟨Zᵢ⟩
    - (2N-1) nearest-neighbor correlators ⟨ZᵢZᵢ₊₁⟩
    - (2N-2) next-nearest-neighbor correlators ⟨ZᵢZᵢ₊₂⟩
    - S stem-pair correlators ⟨ZₐZᵦ⟩ for ViennaRNA-predicted pairs
    
    Args:
        counts: Dict of bitstring → count from measurement
        n_nucleotides: Number of nucleotides N
        stem_pairs: List of (i,j) nucleotide index pairs from ViennaRNA
    
    Returns:
        Feature vector as numpy array
    """
    n_qubits = 2 * n_nucleotides
    total_shots = sum(counts.values())
    
    # Initialize accumulators
    z_local = np.zeros(n_qubits)
    zz_nn = np.zeros(n_qubits - 1)
    zz_nnn = np.zeros(n_qubits - 2)
    
    stem_pairs = stem_pairs or []
    zz_stem = np.zeros(len(stem_pairs))
    
    for bitstring, count in counts.items():
        z = bitstring_to_array(bitstring)
        weight = count / total_shots
        
        # Local magnetizations
        z_local += weight * z
        
        # NN correlators
        for i in range(n_qubits - 1):
            zz_nn[i] += weight * z[i] * z[i+1]
        
        # NNN correlators
        for i in range(n_qubits - 2):
            zz_nnn[i] += weight * z[i] * z[i+2]
        
        # Stem-pair correlators
        for idx, (a, b) in enumerate(stem_pairs):
            # Use first qubit of each nucleotide
            qa, qb = 2*a, 2*b
            if qa < n_qubits and qb < n_qubits:
                zz_stem[idx] += weight * z[qa] * z[qb]
    
    return np.concatenate([z_local, zz_nn, zz_nnn, zz_stem])

def extract_from_statevector(statevector, n_nucleotides: int,
                              stem_pairs: Optional[List[Tuple[int, int]]] = None,
                              n_shots: int = 4096) -> np.ndarray:
    """Extract features by sampling from a statevector (for simulation).
    
    Args:
        statevector: Qiskit Statevector object
        n_nucleotides: Number of nucleotides
        stem_pairs: ViennaRNA stem pairs
        n_shots: Number of measurement shots to simulate
    
    Returns:
        Feature vector
    """
    # Sample measurement outcomes
    counts = statevector.sample_counts(n_shots)
    # Convert to string counts
    str_counts = {format(k, f'0{2*n_nucleotides}b'): v 
                  for k, v in counts.items()} if isinstance(list(counts.keys())[0], int) else counts
    return extract_feature_vector(str_counts, n_nucleotides, stem_pairs)

def feature_vector_dim(n_nucleotides: int, n_stem_pairs: int = 0) -> int:
    """Compute the dimension of the feature vector."""
    n_q = 2 * n_nucleotides
    return n_q + (n_q - 1) + (n_q - 2) + n_stem_pairs
