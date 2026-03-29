"""Layer 4: Trainable local rotations with parameter sharing."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

PURINES = {"A", "G"}
PYRIMIDINES = {"T", "C", "U"}


def get_param_group(qubit_idx: int, sequence: str) -> int:
        """Map a qubit index to one of four shared parameter groups.

            Groups are based on nucleotide type (purine/pyrimidine) and which of the
                two qubits within the 2-qubit nucleotide register the index refers to:
                        0: purine,     first qubit  (q0)
                                1: purine,     second qubit (q1)
                                        2: pyrimidine, first qubit  (q0)
                                                3: pyrimidine, second qubit (q1)
                                                    """
        nuc_idx = qubit_idx // 2
        is_first_qubit = (qubit_idx % 2 == 0)
        nuc = sequence[nuc_idx].upper()
        is_purine = nuc in PURINES

    if is_purine and is_first_qubit:
                return 0
elif is_purine and not is_first_qubit:
            return 1
elif not is_purine and is_first_qubit:
            return 2
else:
            return 3


def apply_trainable_layer(qc: QuantumCircuit, sequence: str, params: np.ndarray = None):
        """Layer 4: Rx/Rz on all qubits with parameter sharing.

            Parameters are shared across four nucleotide-type groups (0-3) plus two
                arm-position groups (4=5' arm, 5=3' arm).  Total: 6 groups x 2 rotations
                    (Rx, Rz) = 12 parameters.

                        If params is None, returns a ParameterVector for variational optimisation.
                            Otherwise applies concrete float values from the supplied array.
                                """
        N = len(sequence)
        n_qubits = 2 * N
        n_groups = 6  # 4 nucleotide-type + 2 arm-position

    if params is None:
                pv = ParameterVector("theta_train", 2 * n_groups)
                for i in range(n_qubits):
                                group = get_param_group(i, sequence)
                                arm = 4 if (i // 2) < N // 2 else 5
                                qc.rx(pv[2 * group], i)
                                qc.rz(pv[2 * group + 1], i)
                                qc.rx(pv[2 * arm], i)
                                qc.rz(pv[2 * arm + 1], i)
                            return pv
else:
            for i in range(n_qubits):
                            group = get_param_group(i, sequence)
                            arm = 4 if (i // 2) < N // 2 else 5
                            qc.rx(float(params[2 * group]), i)
                            qc.rz(float(params[2 * group + 1]), i)
                            qc.rx(float(params[2 * arm]), i)
                            qc.rz(float(params[2 * arm + 1]), i)
                        return None
