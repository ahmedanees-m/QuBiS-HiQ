"""SantaLucia nearest-neighbor parameters and Boltzmann-sigmoid gate mapping.
Ref: SantaLucia & Hicks, Annu Rev Biophys 2004, 33:415-440; SantaLucia PNAS 1998 95:1460."""
import numpy as np
from typing import Dict, Tuple

# 10 unique NN ΔG° at 37°C, 1M NaCl (kcal/mol)
NN_DG: Dict[str, float] = {
    "AA/TT": -1.00, "AT/AT": -0.88, "TA/TA": -0.58,
    "CA/GT": -1.45, "GT/CA": -1.44, "CT/GA": -1.28,
    "GA/CT": -1.30, "CG/CG": -2.17, "GC/GC": -3.42, "GG/CC": -1.84,
}
NN_DH: Dict[str, float] = {
    "AA/TT": -7.9, "AT/AT": -7.2, "TA/TA": -7.2,
    "CA/GT": -8.5, "GT/CA": -8.4, "CT/GA": -7.8,
    "GA/CT": -8.2, "CG/CG": -10.6, "GC/GC": -9.8, "GG/CC": -8.0,
}
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "U": "A"}
BETA = 0.39  # mol/kcal — Boltzmann scaling at 310K
INIT_DG_AT = 1.03
INIT_DG_GC = 0.98

def _dinuc_to_key(dinuc: str) -> str:
    x, y = dinuc[0].upper().replace("U","T"), dinuc[1].upper().replace("U","T")
    xp, yp = COMPLEMENT[x], COMPLEMENT[y]
    for k in [f"{x}{y}/{xp}{yp}", f"{x}{y}/{yp}{xp}", f"{yp}{xp}/{y}{x}"]:
        if k in NN_DG: return k
    raise ValueError(f"Unknown dinucleotide: {dinuc}")

def get_nn_dg(dinuc: str) -> float:
    return NN_DG[_dinuc_to_key(dinuc)]

def compute_total_dg(seq: str) -> float:
    s = seq.upper().replace("U","T")
    dg = sum(get_nn_dg(s[i:i+2]) for i in range(len(s)-1))
    if s[0] in "AT" or s[-1] in "AT": dg += INIT_DG_AT
    if s[0] in "GC" or s[-1] in "GC": dg += INIT_DG_GC
    return dg

def boltzmann_sigmoid(dg: float, beta: float = BETA) -> float:
    """θ(ΔG°) = π · σ(-β·ΔG°). More stable (negative ΔG°) → larger θ."""
    return np.pi / (1.0 + np.exp(beta * dg))

def get_stacking_angle(dinuc: str) -> float:
    return boltzmann_sigmoid(get_nn_dg(dinuc))

def build_full_dinuc_table() -> Dict[str, Tuple[float, float]]:
    table = {}
    for x in "ATGC":
        for y in "ATGC":
            try:
                dg = get_nn_dg(x+y)
                table[x+y] = (dg, boltzmann_sigmoid(dg))
            except ValueError:
                pass
    return table
