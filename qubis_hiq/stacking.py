"""Layer 3: Nearest-neighbor stacking via CX + Ry gates.
Angles from SantaLucia ΔG° through Boltzmann-scaled sigmoid.
"""
from qiskit import QuantumCircuit
from .santalucia import get_stacking_angle

def apply_stacking_layer(qc: QuantumCircuit, sequence: str):
    """Layer 3: Apply CX+Ry(θ) between adjacent nucleotides along the sequence."""
    seq = sequence.upper()
    for k in range(len(seq) - 1):
        dinuc = seq[k:k+2]
        theta = get_stacking_angle(dinuc)
        # First qubits of adjacent nucleotides
        qc.cx(2*k, 2*(k+1))
        qc.ry(theta, 2*(k+1))
        # Second qubits of adjacent nucleotides
        qc.cx(2*k+1, 2*(k+1)+1)
        qc.ry(theta, 2*(k+1)+1)
