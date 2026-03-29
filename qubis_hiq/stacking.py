"""Layer 3: Nearest-neighbor stacking via CX + Ry gates.
Angles from SantaLucia ΔG° through Boltzmann-scaled sigmoid.

FIX (v2): Replaced RZZ with CX→Ry(θ) pairs.

RZZ on a product state |ψ₁⟩⊗|ψ₂⟩ only adds relative phases; the Z-basis
probability amplitudes are unchanged, so ⟨Z⟩ and ⟨ZZ⟩ are unaffected.
CX creates genuine entanglement: ⟨Z₀Z₁⟩ ≠ ⟨Z₀⟩·⟨Z₁⟩ after the gate.
The subsequent Ry(θ) then rotates the target qubit by the SantaLucia angle,
embedding ΔG° information into the observable ⟨ZZ⟩ correlators.

For adjacent nucleotides k and k+1:
  CX(q_{2k}  → q_{2(k+1)})  then  Ry(θ, q_{2(k+1)})   [first qubits]
  CX(q_{2k+1}→ q_{2(k+1)+1}) then  Ry(θ, q_{2(k+1)+1}) [second qubits]
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
