#!/usr/bin/env python3
"""
Physics-Informed Classical Baseline for Oliveira 2020 Tm Prediction
====================================================================
Compares classical feature sets (SantaLucia thermodynamics + GC content)
against QuBiS-HiQ quantum kernel (R²=0.88, MAE=0.60°C from Exp 1E).

Feature sets tested:
  1. SantaLucia ΔG° only (1 feature: total ΔG°)
  2. Per-step ΔG° (18 dinucleotide steps for 19-nt)
  3. Thermodynamics + GC + NN diversity (rich physics)
  4. Same as 3 but with polynomial degree-2 interactions
  5. Classical twin (same architecture as quantum circuit, no interference)

Model: LOO-CV Ridge regression (same protocol as Exp 1E).

KEY QUESTION: Is quantum performance due to the physics encoding itself
(SantaLucia parameters) or the quantum computation (entanglement/interference)?
"""

import sys, os, json, time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import pearsonr

# ── SantaLucia nearest-neighbour parameters ───────────────────────────────────
NN_DG = {
    "AA": -1.00, "AT": -0.88, "AG": -1.28, "AC": -1.44,
    "TA": -0.58, "TT": -1.00, "TG": -1.45, "TC": -1.30,
    "GA": -1.30, "GT": -1.44, "GG": -1.84, "GC": -3.42,
    "CA": -1.45, "CT": -1.28, "CG": -2.17, "CC": -1.84,
}
NN_DH = {
    "AA": -7.9, "AT": -7.2, "AG": -7.8, "AC": -7.8,
    "TA": -7.2, "TT": -7.9, "TG": -8.5, "TC": -8.2,
    "GA": -8.2, "GT": -8.4, "GG": -8.0, "GC": -10.6,
    "CA": -8.5, "CT": -7.8, "CG": -9.8, "CC": -8.0,
}
BETA       = 0.39          # mol/kcal at 310K
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


# ── Feature extraction helpers ────────────────────────────────────────────────

def total_dg(seq):
    s = seq.upper()
    dg = sum(NN_DG.get(s[i:i+2], -1.0) for i in range(len(s)-1))
    if s[0] in "AT" or s[-1] in "AT":
        dg += INIT_AT
    if s[0] in "GC" or s[-1] in "GC":
        dg += INIT_GC
    return dg


def boltzmann_angle(dg):
    return np.pi / (1.0 + np.exp(BETA * dg))


def feat_dg_only(seq):
    """Feature set 1: single total ΔG° scalar."""
    return np.array([total_dg(seq)])


def feat_per_step_dg(seq):
    """Feature set 2: per-step ΔG° values (N-1 values for N-nt sequence)."""
    s = seq.upper()
    return np.array([NN_DG.get(s[i:i+2], -1.0) for i in range(len(s)-1)])


def feat_rich_physics(seq):
    """
    Feature set 3: rich physics features.
      - total ΔG°, ΔH° (sum)
      - per-step ΔG° (18 values)
      - per-step Boltzmann angles (18 values)
      - GC count (scalar)
      - AT count (scalar)
      - NN diversity: number of unique dinucleotides
      - variable-region ΔG°, ΔH° (positions 8-10)
    """
    s = seq.upper()
    N = len(s)
    dg_total = total_dg(s)
    dh_total = sum(NN_DH.get(s[i:i+2], -8.0) for i in range(N-1))
    dg_steps = [NN_DG.get(s[i:i+2], -1.0) for i in range(N-1)]
    angles   = [boltzmann_angle(dg) for dg in dg_steps]
    gc       = sum(1 for c in s if c in "GC")
    at       = N - gc
    diversity = len(set(s[i:i+2] for i in range(N-1)))
    # variable region (positions 8,9,10 = the NNN centre)
    var_centre = s[8:11]  # 3-nt centre
    var_dg = total_dg(var_centre) if len(var_centre) == 3 else 0.0
    var_gc = sum(1 for c in var_centre if c in "GC")
    return np.array(
        [dg_total, dh_total] + dg_steps + angles +
        [gc, at, diversity, var_dg, var_gc],
        dtype=np.float64
    )


def feat_variable_region(seq):
    """Feature set 4: only the variable NNN centre (positions 8–10)."""
    s = seq.upper()
    centre = s[8:11]
    gc = sum(1 for c in centre if c in "GC")
    dg1 = NN_DG.get(centre[0:2], -1.0)
    dg2 = NN_DG.get(centre[1:3], -1.0)
    # boundary dinucleotides
    nn_l = NN_DG.get(s[7] + centre[0], -1.0)
    nn_r = NN_DG.get(centre[2] + s[11], -1.0)
    enc = {"A": [1,0,0,0], "T": [0,1,0,0], "G": [0,0,1,0], "C": [0,0,0,1]}
    onehot = []
    for c in centre:
        onehot.extend(enc.get(c, [0,0,0,0]))
    return np.array([gc, dg1, dg2, nn_l, nn_r] + onehot, dtype=np.float64)


# ── LOO-CV Ridge regression (matches Exp 1E protocol) ────────────────────────

def loo_ridge(X, y, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    loo = LeaveOneOut()
    best_r2, best_alpha, best_preds = -np.inf, None, None
    for alpha in alphas:
        preds = np.zeros(len(y))
        for tr, te in loo.split(X):
            Xtr, Xte = X[tr], X[te]
            ytr = y[tr]
            mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0] = 1.0
            Xtr_s = (Xtr - mu) / sd;  Xte_s = (Xte - mu) / sd
            preds[te] = Ridge(alpha=alpha).fit(Xtr_s, ytr).predict(Xte_s)
        r2 = r2_score(y, preds)
        if r2 > best_r2:
            best_r2, best_alpha, best_preds = r2, alpha, preds
    r, pval = pearsonr(y, best_preds)
    mae = mean_absolute_error(y, best_preds)
    return {
        "R2_loo": float(best_r2), "r": float(r), "p": float(pval),
        "MAE": float(mae), "alpha": float(best_alpha),
        "predictions": best_preds.tolist()
    }


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("PHYSICS-INFORMED CLASSICAL BASELINE")
    print("Oliveira 2020 Tm prediction — classical vs QuBiS-HiQ")
    print("=" * 70)

    # Build data
    full_seqs  = [SCAFFOLD_L + c.split("/")[0] + SCAFFOLD_R for c, _ in CANONICAL]
    centres    = [c for c, _ in CANONICAL]
    y          = np.array([tm for _, tm in CANONICAL])

    print(f"\nDataset: {len(full_seqs)} duplexes, Tm ∈ [{y.min():.1f}, {y.max():.1f}]°C")

    # Build feature matrices
    X_dg_only  = np.array([feat_dg_only(s)        for s in full_seqs])
    X_perstep  = np.array([feat_per_step_dg(s)     for s in full_seqs])
    X_rich     = np.array([feat_rich_physics(s)    for s in full_seqs])
    X_varonly  = np.array([feat_variable_region(s) for s in full_seqs])

    # Polynomial degree-2 interactions on rich features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_rich)

    feature_sets = [
        ("Total ΔG° only (1-d)",        X_dg_only),
        ("Per-step ΔG° (18-d)",          X_perstep),
        ("Variable-region features (17-d)", X_varonly),
        ("Rich physics (41-d)",          X_rich),
        (f"Rich physics + poly-2 ({X_poly.shape[1]}-d)", X_poly),
    ]

    print(f"\n{'Model / Feature Set':<46} {'LOO R²':>7} {'r':>6} {'MAE':>7} {'α':>7}")
    print("-" * 70)

    results = {}
    for name, X in feature_sets:
        res = loo_ridge(X, y)
        results[name] = res
        print(f"{name:<46} {res['R2_loo']:>7.4f} {res['r']:>6.4f} "
              f"{res['MAE']:>7.3f} {res['alpha']:>7.1f}")

    # Reference: QuBiS-HiQ quantum result (from results/exp1e_corrected_results.json)
    quantum_ref_path = os.path.join(os.path.dirname(__file__), "..",
                                    "results", "exp1e_corrected_results.json")
    q_r2 = q_r  = q_mae = None
    q_r2_full = q_r_full = q_mae_full = None
    if os.path.exists(quantum_ref_path):
        with open(quantum_ref_path) as f:
            qdata = json.load(f)
        rr = qdata.get("regression_results", {})
        # Variable-region (21-d) features → the R²=0.88 headline result
        q_var  = rr.get("quantum_variable", {})
        q_r2   = q_var.get("R2_cv")
        q_r    = q_var.get("r")
        q_mae  = q_var.get("MAE")
        # Full 111-d features
        q_full = rr.get("quantum_full", {})
        q_r2_full  = q_full.get("R2_cv")
        q_r_full   = q_full.get("r")
        q_mae_full = q_full.get("MAE")

    print("-" * 70)
    if q_r2 is not None:
        print(f"{'QuBiS-HiQ quantum var-region (21-d)':<46} {q_r2:>7.4f} {q_r:>6.4f} {q_mae:>7.3f}   —")
    if q_r2_full is not None:
        print(f"{'QuBiS-HiQ quantum full (111-d)':<46} {q_r2_full:>7.4f} {q_r_full:>6.4f} {q_mae_full:>7.3f}   —")
    if q_r2 is None:
        print("(QuBiS-HiQ reference: R²=0.88, r=0.94, MAE=0.60 from Exp 1E)")

    # ── Analysis ─────────────────────────────────────────────────────────────
    best_name, best_res = max(results.items(), key=lambda x: x[1]["R2_loo"])
    best_r2_val = best_res["R2_loo"]

    print(f"\nBest classical: {best_name}")
    print(f"  LOO R² = {best_r2_val:.4f}")

    # Compare vs quantum variable-region (21-d) result — the headline R²=0.88
    if q_r2 is not None:
        delta_var  = q_r2      - best_r2_val
        delta_full = (q_r2_full or 0) - best_r2_val
        print(f"  Quantum var-region (21-d) - Classical ΔR² = {delta_var:+.4f}")
        print(f"  Quantum full (111-d)       - Classical ΔR² = {delta_full:+.4f}")
        delta = delta_var   # primary comparison
        if delta > 0.05:
            verdict = "quantum_advantage"
            msg = "Quantum circuit (variable-region) provides genuine improvement (>0.05 ΔR²)."
        elif delta < -0.05:
            verdict = "classical_wins"
            msg = ("Classical physics baseline outperforms quantum — quantum advantage is "
                   "NOT from computation; the SantaLucia encoding alone explains the performance.")
        else:
            verdict = "comparable"
            msg = "Comparable performance (ΔR² < 0.05): quantum advantage may be from physics encoding."
    else:
        delta, verdict, msg = None, "unknown", "Quantum reference results not found."

    print(f"  Verdict: {msg}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "Physics-Informed Classical Baseline",
        "dataset": {
            "source": "Oliveira et al. 2020", "n_sequences": len(y),
            "scaffold_top": f"5'-{SCAFFOLD_L}[NNN]{SCAFFOLD_R}-3'",
            "tm_range_C": [float(y.min()), float(y.max())],
        },
        "classical_results": {
            name: res for name, res in results.items()
        },
        "quantum_reference": {
            "source": "Exp 1E corrected results",
            "variable_region_21d": {"R2_loo": q_r2, "r": q_r, "MAE": q_mae},
            "full_111d": {"R2_loo": q_r2_full, "r": q_r_full, "MAE": q_mae_full},
            "method": "LOO-CV Ridge",
            "note": "R2=0.88 headline figure uses variable-region 21-d features",
        },
        "comparison": {
            "best_classical": best_name,
            "best_classical_R2": best_r2_val,
            "quantum_var_R2": q_r2,
            "quantum_full_R2": q_r2_full,
            "delta_R2_vs_quantum_var": float(q_r2 - best_r2_val) if q_r2 else None,
            "delta_R2_vs_quantum_full": float(q_r2_full - best_r2_val) if q_r2_full else None,
            "verdict": verdict,
            "interpretation": msg,
        },
    }

    out_path = "results/classical_physics_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
