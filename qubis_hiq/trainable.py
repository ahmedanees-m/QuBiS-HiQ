"""Layer 4: Trainable local rotations with parameter sharing.
Reduces parameters from ~96 to ~40 for 16-qubit circuit."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

PURINES = {"A", "G"}
PYRIMIDINES = {"T", "C", "U"}

def get_param_group(qubit_idx: int, sequence: str) -> int:
    """Determine shared parameter group for a qubit.
    Groups: purine_q0(0), purine_q1(1), pyrimidine_q0(2), pyrimidine_q1(3),
            5prime_mod(4), 3prime_mod(5) → total 6 base groups × 2 (Rx,Rz) = 12 params."""
    nuc_idx = qubit_idx // 2
    is_first_qubit = (qubit_idx % 2 == 0)
    nuc = sequence[nuc_idx].upper()
    is_purine = nuc in PURINES
    N = len(sequence)
    is_5prime = nuc_idx < N // 2

    if is_purine and is_first_qubit: base = 0
    elif is_purine and not is_first_qubit: base = 1
    elif not is_purine and is_first_qubit: base = 2
    else: base = 3
    arm_offset = 4 if is_5prime else 5
    return base  # Simplified: 4 groups. Arm modifier added as separate parameter.

def apply_trainable_layer(qc: QuantumCircuit, sequence: str,
                           params: np.ndarray = None):
    """Layer 4: Rx/Rz on all qubits with parameter sharing.
    If params is None, creates a ParameterVector for variational optimization."""
    N = len(sequence)
    n_qubits = 2 * N
    n_groups = 6  # 4 nucleotide-type + 2 arm-position
    
    if params is None:
        # Create parameterized circuit for training
        pv = ParameterVector("θ_train", 2 * n_groups)  # Rx and Rz per group
        for i in range(n_qubits):
            group = get_param_group(i, sequence)
            arm = 4 if (i // 2) < N // 2 else 5
            qc.rx(pv[2*group], i)
            qc.rz(pv[2*group+1], i)
            # Arm modifier
            qc.rx(pv[2*arm], i)
            qc.rz(pv[2*arm+1], i)
        return pv
    else:
        # Apply with concrete parameter values
        for i in range(n_qubits):
            group = get_param_group(i, sequence)
            arm = 4 if (i // 2) < N // 2 else 5
            qc.rx(float(params[2*group]), i)
            qc.rz(float(params[2*group+1]), i)
            qc.rx(float(params[2*arm]), i)
            qc.rz(float(params[2*arm+1]), i)
        return None
