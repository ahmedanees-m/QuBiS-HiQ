"""Layer 5: Hadamard interference + Layer 6: Z-basis measurement."""
from qiskit import QuantumCircuit

def apply_hadamard_layer(qc: QuantumCircuit, n_qubits: int):
    """Layer 5: Apply H gates on all qubits."""
    for i in range(n_qubits):
        qc.h(i)

def apply_measurement(qc: QuantumCircuit):
    """Layer 6: Measure all qubits in Z-basis."""
    qc.measure_all()
