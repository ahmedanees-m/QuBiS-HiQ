#!/usr/bin/env python3
"""
Best Configuration Synthesis — QuBiS-HiQ Diagnostic Suite
==========================================================
Consolidates all diagnostic findings to identify the optimal circuit-feature
configuration and test the combined quantum+classical feature set.

Addresses two open questions from the ablation study:
  Q1. The no-WC variant achieves R²=0.9122 on variable-region features — does
      combining quantum variable-region features with classical physics features
      improve beyond either alone (classical=0.9353)?
  Q2. Is the kernel condition problem solvable with Kernel Ridge Regression (KRR)?
      KRR analytically solves the dual problem with L2 regularisation, avoiding
      numerical issues that can affect kernel SVM.

Feature combinations tested (all LOO-CV Ridge, same protocol as Exp 1E):
  A.  Quantum var-region only       (21-d, best variant: no-WC)
  B.  Classical physics only        (17-d, variable-region features)
  C.  Quantum + Classical combined  (38-d)
  D.  Quantum full features         (111-d, no-WC variant)
  E.  KRR on feature-vector kernel  (linear kernel, no-WC, all 111-d)

Summary tables are printed and saved to results/best_configuration_results.json.
"""

import sys, os, json, time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr

# ── SantaLucia parameters ─────────────────────────────────────────────────────
NN_DG = {
    "AA": -1.00, "AT": -0.88, "AG": -1.28, "AC": -1.44,
    "TA": -0.58, "TT": -1.00, "TG": -1.45, "TC": -1.30,
    "GA": -1.30, "GT": -1.44, "GG": -1.84, "GC": -3.42,
    "CA": -1.45, "CT": -1.28, "CG": -2.17, "CC": -1.84,
}
BETA       = 0.39
INIT_AT    = 1.03
INIT_GC    = 0.98
SCAFFOLD_L = "CGACGTGC"
SCAFFOLD_R = "ATGTGCTG"

# ── Oliveira 2020 canonical duplexes ─────────────────────────────────────────
CANONICAL_RAW = [
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
CANONICAL = [(c, t) for c, t in CANONICAL_RAW if c not in seen and not seen.add(c)]


# ── Feature builders ──────────────────────────────────────────────────────────

def boltzmann_angle(dg):
    return np.pi / (1.0 + np.exp(BETA * dg))


def classical_variable_region(seq):
    """17-d classical physics features for the NNN centre (positions 8-10)."""
    s = seq.upper()
    centre = s[8:11]
    gc  = sum(1 for c in centre if c in "GC")
    dg1 = NN_DG.get(centre[0:2], -1.0)
    dg2 = NN_DG.get(centre[1:3], -1.0)
    nn_l = NN_DG.get(s[7] + centre[0], -1.0)
    nn_r = NN_DG.get(centre[2] + s[11], -1.0)
    enc = {"A": [1,0,0,0], "T": [0,1,0,0], "G": [0,0,1,0], "C": [0,0,0,1]}
    onehot = []
    for c in centre:
        onehot.extend(enc.get(c, [0,0,0,0]))
    return np.array([gc, dg1, dg2, nn_l, nn_r] + onehot, dtype=np.float64)


def extract_variable_region_slice(full_fv, n_qubits=38):
    """21-d quantum variable-region slice (qubits 16-21)."""
    NQ = n_qubits
    z_var  = full_fv[16:22]
    zz_adj = full_fv[NQ + 15: NQ + 22]
    zz_nxt = full_fv[NQ + NQ - 1 + 14: NQ + NQ - 1 + 22]
    return np.concatenate([z_var, zz_adj, zz_nxt])


# ── LOO-CV helpers ────────────────────────────────────────────────────────────

def loo_ridge(X, y, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    loo = LeaveOneOut()
    best_r2, best_alpha, best_preds = -np.inf, None, None
    for alpha in alphas:
        preds = np.zeros(len(y))
        for tr, te in loo.split(X):
            Xtr, Xte = X[tr], X[te]
            ytr = y[tr]
            mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
            Xtr_s = (Xtr - mu) / sd;  Xte_s = (Xte - mu) / sd
            preds[te] = Ridge(alpha=alpha).fit(Xtr_s, ytr).predict(Xte_s)
        r2 = r2_score(y, preds)
        if r2 > best_r2:
            best_r2, best_alpha, best_preds = r2, alpha, preds
    r, pval = pearsonr(y, best_preds)
    mae = mean_absolute_error(y, best_preds)
    return {"R2_loo": float(best_r2), "r": float(r), "p": float(pval),
            "MAE": float(mae), "alpha": float(best_alpha),
            "predictions": best_preds.tolist()}


def loo_krr(K, y, alphas=(1e-4, 1e-3, 0.01, 0.1, 1.0)):
    """LOO-CV Kernel Ridge Regression on precomputed kernel matrix."""
    loo = LeaveOneOut()
    best_r2, best_alpha, best_preds = -np.inf, None, None
    for alpha in alphas:
        preds = np.zeros(len(y))
        for tr, te in loo.split(K):
            K_tr = K[np.ix_(tr, tr)]
            K_te = K[np.ix_(te, tr)]
            krr = KernelRidge(alpha=alpha, kernel='precomputed')
            krr.fit(K_tr, y[tr])
            preds[te] = krr.predict(K_te)
        r2 = r2_score(y, preds)
        if r2 > best_r2:
            best_r2, best_alpha, best_preds = r2, alpha, preds
    r, pval = pearsonr(y, best_preds)
    mae = mean_absolute_error(y, best_preds)
    return {"R2_loo": float(best_r2), "r": float(r), "p": float(pval),
            "MAE": float(mae), "alpha": float(best_alpha),
            "predictions": best_preds.tolist()}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("BEST CONFIGURATION SYNTHESIS — QuBiS-HiQ")
    print("Identifying the optimal circuit-feature combination")
    print("=" * 70)

    full_seqs = [SCAFFOLD_L + c.split("/")[0] + SCAFFOLD_R for c, _ in CANONICAL]
    y = np.array([tm for _, tm in CANONICAL])
    N = len(full_seqs)
    print(f"\nDataset: {N} duplexes, Tm ∈ [{y.min():.1f}, {y.max():.1f}]°C")

    # ── Load pre-computed ablation features (no-WC variant, full + var-region) ─
    print("\n[1] Loading pre-computed features from ablation study …")
    abl_path = "results/entanglement_ablation_results.json"
    with open(abl_path) as f:
        abl = json.load(f)

    # Already computed during ablation — extract stored predictions for reuse
    no_wc_full_preds = np.array(abl["variant_results"]["no_wc"]["full_features"]["predictions"])
    no_wc_var_preds  = np.array(abl["variant_results"]["no_wc"]["variable_region_features"]["predictions"])

    # ── Re-run MPS simulation for no-WC to get feature matrices ──────────────
    print("[2] Re-running MPS simulation (no-WC variant) to build feature matrices …")
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qubis_hiq.circuit_builder import build_circuit
    from qubis_hiq.vienna_interface import predict_structure

    sim = AerSimulator(method='matrix_product_state')
    N_QUBITS = 38
    N_SHOTS  = 8192
    PARAMS   = np.zeros(12)

    stem_pairs_list = [predict_structure(seq)[1] for seq in full_seqs]

    t0 = time.time()
    feats_full_nowc = []
    feats_var_nowc  = []
    for i, (seq, sp) in enumerate(zip(full_seqs, stem_pairs_list)):
        qc   = build_circuit(seq, sp, PARAMS, include_measurement=True, skip_wc=True)
        qc_t = transpile(qc, sim)
        counts = sim.run(qc_t, shots=N_SHOTS).result().get_counts()
        # Extract Z marginals + ZZ correlators
        NQ = N_QUBITS
        z_m = np.zeros(NQ);  zz_a = np.zeros(NQ-1);  zz_n = np.zeros(NQ-2)
        tot = sum(counts.values())
        for bs, cnt in counts.items():
            bits = [int(b) for b in bs][::-1]
            zv   = [1 - 2*b for b in bits[:NQ]]
            for k in range(NQ):      z_m[k]   += zv[k] * cnt
            for k in range(NQ - 1): zz_a[k]  += zv[k] * zv[k+1] * cnt
            for k in range(NQ - 2): zz_n[k]  += zv[k] * zv[k+2] * cnt
        fv_full = np.concatenate([z_m/tot, zz_a/tot, zz_n/tot])
        fv_var  = extract_variable_region_slice(fv_full, NQ)
        feats_full_nowc.append(fv_full)
        feats_var_nowc.append(fv_var)
        if (i + 1) % 16 == 0:
            print(f"    [{i+1:3d}/{N}]")

    print(f"    Done in {time.time()-t0:.1f}s")
    X_q_full = np.array(feats_full_nowc)   # (64, 111) quantum full, no-WC
    X_q_var  = np.array(feats_var_nowc)    # (64,  21) quantum var-region, no-WC

    # ── Build classical variable-region features ──────────────────────────────
    print("[3] Building classical variable-region features …")
    X_cl_var = np.array([classical_variable_region(s) for s in full_seqs])  # (64, 17)

    # ── Build combined feature matrix ─────────────────────────────────────────
    X_combined = np.hstack([X_q_var, X_cl_var])   # (64, 38)

    # ── Feature-vector linear kernel (ℓ2-normalised dot product) ─────────────
    X_q_full_norm = normalize(X_q_full, norm='l2')
    K_fv = X_q_full_norm @ X_q_full_norm.T         # (64, 64) precomputed kernel

    # ── LOO-CV evaluations ────────────────────────────────────────────────────
    print("[4] Running LOO-CV evaluations …")

    configs = [
        ("Quantum var-region  (no-WC, 21-d)",    X_q_var,   "ridge"),
        ("Classical var-region          (17-d)",  X_cl_var,  "ridge"),
        ("Quantum + Classical combined  (38-d)",  X_combined,"ridge"),
        ("Quantum full features (no-WC, 111-d)",  X_q_full,  "ridge"),
        ("Kernel Ridge (linear kernel, no-WC)",   K_fv,      "krr"),
    ]

    results = {}
    print(f"\n{'Configuration':<44} {'LOO R²':>7} {'r':>6} {'MAE(°C)':>8} {'α':>8}")
    print("-" * 77)

    for name, X, method in configs:
        if method == "ridge":
            res = loo_ridge(X, y)
        else:
            res = loo_krr(X, y)
        results[name] = {"method": method, **res}
        print(f"{name:<44} {res['R2_loo']:>7.4f} {res['r']:>6.4f} "
              f"{res['MAE']:>8.3f} {res['alpha']:>8.1e}")

    # ── Reference lines ───────────────────────────────────────────────────────
    print("-" * 77)
    q_ref_path = "results/exp1e_corrected_results.json"
    if os.path.exists(q_ref_path):
        with open(q_ref_path) as f:
            qd = json.load(f)
        rr = qd["regression_results"]
        qv = rr["quantum_variable"]
        qf = rr["quantum_full"]
        cr = rr["classical_rich"]
        print(f"{'[Exp 1E ref] Quantum var-region (21-d)':<44} "
              f"{qv['R2_cv']:>7.4f} {qv['r']:>6.4f} {qv['MAE']:>8.3f}   —")
        print(f"{'[Exp 1E ref] Classical rich    (18-d)':<44} "
              f"{cr['R2_cv']:>7.4f} {cr['r']:>6.4f} {cr['MAE']:>8.3f}   —")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SYNTHESIS")
    print(f"{'='*70}")

    best_name = max(results, key=lambda k: results[k]["R2_loo"])
    best_r2   = results[best_name]["R2_loo"]

    q_var_r2  = results["Quantum var-region  (no-WC, 21-d)"]["R2_loo"]
    cl_var_r2 = results["Classical var-region          (17-d)"]["R2_loo"]
    comb_r2   = results["Quantum + Classical combined  (38-d)"]["R2_loo"]

    print(f"\n  Best configuration:             {best_name}")
    print(f"  Best R² (LOO-CV):               {best_r2:.4f}")
    print(f"\n  Quantum var-region (no-WC):     {q_var_r2:.4f}")
    print(f"  Classical var-region:           {cl_var_r2:.4f}")
    print(f"  Combined quantum + classical:   {comb_r2:.4f}")

    gap   = cl_var_r2 - q_var_r2
    synergy = comb_r2 - max(q_var_r2, cl_var_r2)
    print(f"\n  Classical - Quantum gap:        {gap:+.4f}")
    print(f"  Combination synergy:            {synergy:+.4f}")

    if synergy > 0.01:
        synergy_msg = ("Quantum and classical features are COMPLEMENTARY — "
                       "combination exceeds either alone by >{:.3f} R².".format(synergy))
    elif synergy > 0:
        synergy_msg = "Marginal synergy — combined is slightly better than best single source."
    else:
        synergy_msg = ("No synergy — combined does not improve on the best individual source. "
                       "Quantum and classical features encode similar information.")
    print(f"  Interpretation: {synergy_msg}")

    entanglement_note = (
        "At the variable-region level, entanglement contribution is "
        f"{results['Quantum var-region  (no-WC, 21-d)']['R2_loo'] - abl['contribution_analysis']['encoding_only_R2']:.4f} "
        f"(full-feature contribution was "
        f"{abl['contribution_analysis']['entanglement_contrib']:.4f}). "
        "Local Ry encoding captures most variable-region Tm information."
    )
    print(f"\n  Entanglement note: {entanglement_note}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "Best Configuration Synthesis",
        "dataset": {
            "source": "Oliveira et al. 2020", "n_sequences": N,
            "scaffold": f"5'-{SCAFFOLD_L}[NNN]{SCAFFOLD_R}-3'",
            "tm_range_C": [float(y.min()), float(y.max())],
        },
        "simulation": {
            "variant": "no-WC (skip_wc=True)",
            "method": "matrix_product_state",
            "n_shots": N_SHOTS, "n_qubits": N_QUBITS,
        },
        "configurations": results,
        "synthesis": {
            "best_configuration": best_name,
            "best_R2": float(best_r2),
            "quantum_var_R2": float(q_var_r2),
            "classical_var_R2": float(cl_var_r2),
            "combined_R2": float(comb_r2),
            "classical_quantum_gap": float(gap),
            "combination_synergy": float(synergy),
            "synergy_interpretation": synergy_msg,
            "entanglement_note": entanglement_note,
        },
    }

    out_path = "results/best_configuration_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
