#!/usr/bin/env python3
"""
Kernel Condition Number Analysis for QuBiS-HiQ
===============================================
Computes condition number κ(K) = σ_max / σ_min of two kernel matrices:
  1. Exact quantum kernel on 8-mers (K[i,j] = |<ψ(x_i)|ψ(x_j)>|²)
     via statevector inner products (16 qubits each — tractable)
  2. Feature-vector linear kernel on Oliveira 2020 19-nt sequences
     (statevector for 38 qubits is intractable; linear kernel over
      quantum feature vectors is a valid proxy)

High condition number (>1e6) → near-degeneracy, overfitting risk.
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from sklearn.preprocessing import normalize

from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.vienna_interface import predict_structure
from qubis_hiq.feature_extraction import extract_feature_vector
from qubis_hiq.santalucia import compute_total_dg

# ── Oliveira 2020 canonical duplexes (from exp1e_corrected.py) ───────────────
SCAFFOLD_LEFT  = "CGACGTGC"
SCAFFOLD_RIGHT = "ATGTGCTG"
NN_DG = {
    "AA": -1.00, "AT": -0.88, "AG": -1.28, "AC": -1.44,
    "TA": -0.58, "TT": -1.00, "TG": -1.45, "TC": -1.30,
    "GA": -1.30, "GT": -1.44, "GG": -1.84, "GC": -3.42,
    "CA": -1.45, "CT": -1.28, "CG": -2.17, "CC": -1.84,
}
BETA = 0.39

CANONICAL_DUPLEXES_RAW = [
    ("GCG/CGC", 69.3), ("CGC/GCG", 69.1), ("GGC/CCG", 68.9),
    ("GCC/CGG", 68.7), ("CGG/GCC", 68.2), ("CCG/GGC", 68.2),
    ("GGG/CCC", 67.7), ("CCC/GGG", 67.7), ("TGC/ACG", 67.5),
    ("GAC/CTG", 67.2), ("GCA/CGT", 67.1), ("TCG/AGC", 66.6),
    ("GGA/CCT", 66.6), ("CGA/GCT", 66.5), ("CAG/GTC", 66.5),
    ("GAG/CTC", 66.5), ("GCT/CGA", 66.4), ("CAC/GTG", 66.4),
    ("GTC/CAG", 66.4), ("GTG/CAC", 66.2), ("CGT/GCA", 66.2),
    ("AGC/TCG", 66.1), ("TCC/AGG", 66.1), ("CCA/GGT", 66.0),
    ("AGG/TCC", 66.0), ("GGT/CCA", 66.0), ("CTG/GAC", 65.9),
    ("CTC/GAG", 65.8), ("TGG/ACC", 65.5), ("ACG/TGC", 65.4),
    ("ACC/TGG", 65.2), ("TTG/AAC", 64.9), ("ATC/TAG", 64.8),
    ("GAA/CTT", 64.7), ("CAA/GTT", 64.5), ("AAC/TTG", 64.5),
    ("CCT/GGA", 64.4), ("TTC/AAG", 64.4), ("TGA/ACT", 64.4),
    ("TCA/AGT", 64.2), ("AGA/TCT", 64.2), ("GAT/CTA", 64.1),
    ("GTA/CAT", 63.9), ("AAG/TTC", 63.8), ("ATG/TAC", 63.6),
    ("GTT/CAA", 63.6), ("TAG/ATC", 63.6), ("TAC/ATG", 63.6),
    ("CTA/GAT", 63.6), ("TCT/AGA", 63.0), ("CAT/GTA", 63.0),
    ("ACA/TGT", 63.0), ("CTT/GAA", 62.9), ("AGT/TCA", 62.7),
    ("AAA/TTT", 62.4), ("TGT/ACA", 62.1), ("ATA/TAT", 61.8),
    ("TAA/ATT", 61.6), ("TAT/ATA", 61.3), ("AAT/TTA", 61.2),
    ("ATT/TAA", 60.7), ("ACT/TGA", 60.7), ("TTA/AAT", 61.1),
    ("TTT/AAA", 61.0),
]
seen = set()
CANONICAL = []
for entry in CANONICAL_DUPLEXES_RAW:
    if entry[0] not in seen:
        seen.add(entry[0])
        CANONICAL.append(entry)


def boltzmann_angle(dg):
    return np.pi / (1.0 + np.exp(BETA * dg))


def build_qubis_circuit_8mer(seq):
    """Minimal 8-mer circuit (same physics as exp1e, include_measurement=False)."""
    N = len(seq)
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2 * N)
    enc = {"A": (0,0), "T": (0,1), "G": (1,0), "C": (1,1)}
    ang = {0: np.pi/3, 1: 2*np.pi/3}
    for i, nuc in enumerate(seq.upper()):
        b = enc[nuc]
        qc.ry(ang[b[0]], 2*i)
        qc.ry(ang[b[1]], 2*i+1)
    for k in range(N - 1):
        dinuc = seq[k:k+2].upper()
        theta = boltzmann_angle(NN_DG.get(dinuc, -1.0))
        qc.cx(2*k, 2*(k+1))
        qc.ry(theta, 2*(k+1))
    return qc  # no measurement


def statevector_kernel(sv_list):
    """
    Build exact quantum kernel matrix K[i,j] = |<ψ_i|ψ_j>|² from statevectors.
    """
    n = len(sv_list)
    K = np.eye(n, dtype=np.float64)
    arr = [np.asarray(sv) for sv in sv_list]
    for i in range(n):
        for j in range(i + 1, n):
            ip = np.vdot(arr[i], arr[j])   # conjugate transpose inner product
            k_ij = float(np.abs(ip) ** 2)
            K[i, j] = k_ij
            K[j, i] = k_ij
    return K


def feature_vector_kernel(X):
    """
    Linear kernel on normalised quantum feature vectors:
        K[i,j] = φ(x_i)·φ(x_j)  (cosine similarity after ℓ2-normalisation)
    """
    X_norm = normalize(X, norm='l2')
    return X_norm @ X_norm.T


def analyze_kernel(K, label):
    """Compute condition number, effective rank, PSD check."""
    sv = np.linalg.svd(K, compute_uv=False)
    sv_pos = sv[sv > 1e-14]
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    cond_nonzero = float(sv[0] / sv_pos[-1]) if len(sv_pos) else np.inf
    eff_rank = int(np.sum(sv > 1e-10))
    eigenvalues = np.linalg.eigvalsh(K)
    min_eig = float(eigenvalues[0])
    is_psd = bool(min_eig >= -1e-8)

    print(f"\n  [{label}]")
    print(f"    Matrix size:           {K.shape[0]} × {K.shape[0]}")
    print(f"    Condition number κ:    {cond:.3e}  (log₁₀ = {np.log10(cond+1e-300):.2f})")
    print(f"    Effective rank:        {eff_rank} / {K.shape[0]}")
    print(f"    Min eigenvalue:        {min_eig:.3e}")
    print(f"    Is PSD:                {is_psd}")
    print(f"    Top-5 singular values: {sv[:5].tolist()}")
    print(f"    Bot-5 singular values: {sv[-5:].tolist()}")

    if cond > 1e12:
        msg = "CRITICAL: κ > 1e12 — numerically singular; SVM unstable"
    elif cond > 1e6:
        msg = "WARNING: κ > 1e6 — poorly conditioned; regularisation essential"
    elif cond > 1e3:
        msg = "MODERATE: κ > 1e3 — monitor stability"
    else:
        msg = "GOOD: κ < 1e3 — well-conditioned"
    print(f"    Assessment:            {msg}")

    return {
        "label": label,
        "n": K.shape[0],
        "condition_number": cond,
        "condition_number_log10": float(np.log10(cond + 1e-300)),
        "effective_rank": eff_rank,
        "min_eigenvalue": min_eig,
        "is_psd": is_psd,
        "top5_singular_values": sv[:5].tolist(),
        "bot5_singular_values": sv[-5:].tolist(),
        "assessment": msg,
    }


def scaling_analysis(sv_list, label_prefix, sizes=None):
    """Condition number vs subset size."""
    n_max = len(sv_list)
    if sizes is None:
        sizes = [s for s in [10, 20, 30, 40, 50] if s <= n_max]
    results = []
    print(f"\n  Scaling analysis — {label_prefix}:")
    print(f"    {'n':>4}  {'κ':>12}  {'log₁₀κ':>8}  {'eff_rank':>8}")
    for n in sizes:
        sv_sub = sv_list[:n]
        K_sub = statevector_kernel(sv_sub)
        svals = np.linalg.svd(K_sub, compute_uv=False)
        cond = float(svals[0] / svals[-1]) if svals[-1] > 0 else np.inf
        er = int(np.sum(svals > 1e-10))
        print(f"    {n:>4}  {cond:>12.3e}  {np.log10(cond+1e-300):>8.2f}  {er:>8}")
        results.append({"n": n, "condition_number": cond,
                        "log10_cond": float(np.log10(cond+1e-300)),
                        "effective_rank": er})
    return results


# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("KERNEL CONDITION NUMBER ANALYSIS")
    print("QuBiS-HiQ — Quantum Kernel Degeneracy & Conditioning")
    print("=" * 70)

    # ── Part A: Exact quantum kernel on 8-mers ───────────────────────────────
    print("\n[A] Exact quantum kernel on random 8-mers")
    print("    Circuit: encoding + stacking (16 qubits, statevector)")

    np.random.seed(42)
    N_8MER = 50
    seqs_8 = [''.join(np.random.choice(list('ATGC'), 8)) for _ in range(N_8MER)]

    print(f"    Generating statevectors for {N_8MER} sequences …")
    t0 = time.time()
    sv_list_8 = []
    for i, seq in enumerate(seqs_8):
        qc = build_qubis_circuit_8mer(seq)
        sv = Statevector.from_instruction(qc)
        sv_list_8.append(np.asarray(sv))
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{N_8MER}")
    t_sv = time.time() - t0
    print(f"    Done in {t_sv:.1f}s")

    print(f"\n    Building {N_8MER}×{N_8MER} quantum kernel matrix …")
    t0 = time.time()
    K_exact = statevector_kernel(sv_list_8)
    print(f"    Done in {time.time()-t0:.1f}s")

    res_exact = analyze_kernel(K_exact, "Exact quantum kernel (8-mers, n=50)")
    scale_res_8 = scaling_analysis(sv_list_8, "8-mer exact kernel")

    # ── Part B: Feature-vector linear kernel on Oliveira 19-nt sequences ─────
    print("\n[B] Feature-vector linear kernel on Oliveira 2020 (19-nt, n=64)")
    print("    (Statevector for 38-qubit circuits is intractable;")
    print("     linear kernel K[i,j]=φ(x_i)·φ(x_j) is the practical proxy.)")

    sim_mps = AerSimulator(method='matrix_product_state')
    n_shots = 8192
    oliveira_seqs = [SCAFFOLD_LEFT + c.split("/")[0] + SCAFFOLD_RIGHT
                     for c, _ in CANONICAL]
    oliveira_tms  = np.array([tm for _, tm in CANONICAL])

    N_OL = len(oliveira_seqs)
    SEQ_LEN = 19
    N_QUBITS = 38

    def build_and_extract_mps(seq):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2 * len(seq))
        enc = {"A": (0,0), "T": (0,1), "G": (1,0), "C": (1,1)}
        ang = {0: np.pi/3, 1: 2*np.pi/3}
        for i, nuc in enumerate(seq.upper()):
            b = enc[nuc]
            qc.ry(ang[b[0]], 2*i)
            qc.ry(ang[b[1]], 2*i+1)
        for k in range(len(seq) - 1):
            dinuc = seq[k:k+2].upper()
            theta = boltzmann_angle(NN_DG.get(dinuc, -1.0))
            qc.cx(2*k, 2*(k+1))
            qc.ry(theta, 2*(k+1))
        qc.measure_all()
        job = sim_mps.run(qc, shots=n_shots)
        counts = job.result().get_counts()
        # Extract features: Z marginals + ZZ adjacent + ZZ next-nearest
        NQ = 2 * len(seq)
        z_marg = np.zeros(NQ)
        zz_adj = np.zeros(NQ - 1)
        zz_nxt = np.zeros(NQ - 2)
        for bs, cnt in counts.items():
            bits = [int(b) for b in bs][::-1]
            zv = [1 - 2*b for b in bits[:NQ]]
            for k in range(NQ):
                z_marg[k] += zv[k] * cnt
            for k in range(NQ - 1):
                zz_adj[k] += zv[k] * zv[k+1] * cnt
            for k in range(NQ - 2):
                zz_nxt[k] += zv[k] * zv[k+2] * cnt
        total = sum(counts.values())
        return np.concatenate([z_marg/total, zz_adj/total, zz_nxt/total])

    print(f"    Running MPS simulations for {N_OL} Oliveira sequences …")
    t0 = time.time()
    features_ol = []
    for i, seq in enumerate(oliveira_seqs):
        fv = build_and_extract_mps(seq)
        features_ol.append(fv)
        if (i + 1) % 16 == 0:
            print(f"      {i+1}/{N_OL}")
    t_ol = time.time() - t0
    print(f"    Done in {t_ol:.1f}s")

    X_ol = np.array(features_ol)
    K_fv = feature_vector_kernel(X_ol)
    res_fv = analyze_kernel(K_fv, f"Feature-vector linear kernel (Oliveira, n={N_OL})")

    # ── Subset scaling for Oliveira feature kernel ────────────────────────────
    print(f"\n  Scaling analysis — Oliveira feature kernel:")
    print(f"    {'n':>4}  {'κ':>12}  {'log₁₀κ':>8}  {'eff_rank':>8}")
    X_norm_ol = normalize(X_ol, norm='l2')
    scale_res_ol = []
    for n in [10, 20, 30, 40, 50, N_OL]:
        if n > N_OL:
            continue
        K_sub = X_norm_ol[:n] @ X_norm_ol[:n].T
        svals = np.linalg.svd(K_sub, compute_uv=False)
        cond = float(svals[0] / svals[-1]) if svals[-1] > 0 else np.inf
        er = int(np.sum(svals > 1e-10))
        print(f"    {n:>4}  {cond:>12.3e}  {np.log10(cond+1e-300):>8.2f}  {er:>8}")
        scale_res_ol.append({"n": n, "condition_number": cond,
                             "log10_cond": float(np.log10(cond+1e-300)),
                             "effective_rank": er})

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "Kernel Condition Number Analysis",
        "exact_8mer_kernel": {
            **res_exact,
            "n_sequences": N_8MER,
            "circuit": "encoding + stacking (16 qubits)",
            "method": "statevector inner product",
            "scaling": scale_res_8,
        },
        "oliveira_feature_kernel": {
            **res_fv,
            "n_sequences": N_OL,
            "circuit": "encoding + stacking (38 qubits)",
            "method": "MPS feature vectors (linear kernel proxy)",
            "scaling": scale_res_ol,
        },
        "interpretation": {
            "threshold_critical": 1e12,
            "threshold_warning": 1e6,
            "threshold_moderate": 1e3,
        },
    }

    out_path = "results/kernel_condition_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Exact 8-mer kernel κ       = {res_exact['condition_number']:.3e}")
    print(f"  Oliveira feature kernel κ  = {res_fv['condition_number']:.3e}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
