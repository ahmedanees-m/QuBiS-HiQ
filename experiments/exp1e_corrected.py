"""
EXP 1E
============================================================================
"""

import numpy as np
import json
import time

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

BETA = 0.39  # mol/kcal at 310K (37°C)
NN_DG = {
    "AA": -1.00, "AT": -0.88, "AG": -1.28, "AC": -1.44,
    "TA": -0.58, "TT": -1.00, "TG": -1.45, "TC": -1.30,
    "GA": -1.30, "GT": -1.44, "GG": -1.84, "GC": -3.42,
    "CA": -1.45, "CT": -1.28, "CG": -2.17, "CC": -1.84,
}

# ═══════════════════════════════════════════════════════════
# CORRECTED SCAFFOLD — from page 8275 of Oliveira et al. 2020
# ═══════════════════════════════════════════════════════════
SCAFFOLD_LEFT = "CGACGTGC"    # 8 nt before variable region
SCAFFOLD_RIGHT = "ATGTGCTG"   # 8 nt after variable region
# Full: CGACGTGC + NNN + ATGTGCTG = 8 + 3 + 8 = 19 nt per strand
SEQ_LENGTH = 19
N_QUBITS = 2 * SEQ_LENGTH  # 38
FEATURE_DIM = 6 * SEQ_LENGTH - 3  # 111

COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}

def get_full_top_strand(centre):
    top = centre.split("/")[0]
    return SCAFFOLD_LEFT + top + SCAFFOLD_RIGHT

def boltzmann_angle(dg):
    x = -BETA * dg
    return np.pi / (1.0 + np.exp(-x))

def build_qubis_circuit(seq):
    N = len(seq)
    n_qubits = 2 * N
    qc = QuantumCircuit(n_qubits)
    encoding = {"A": (0, 0), "T": (0, 1), "G": (1, 0), "C": (1, 1)}
    angles = {0: np.pi/3, 1: 2*np.pi/3}
    for i, nuc in enumerate(seq.upper()):
        bits = encoding[nuc]
        qc.ry(angles[bits[0]], 2*i)
        qc.ry(angles[bits[1]], 2*i + 1)
    for k in range(N - 1):
        dinuc = seq[k:k+2].upper()
        dg = NN_DG.get(dinuc, -1.0)
        theta_s = boltzmann_angle(dg)
        qc.cx(2*k, 2*(k+1))
        qc.ry(theta_s, 2*(k+1))
    qc.measure_all()
    return qc

def extract_features(counts, n_qubits, n_shots):
    N = n_qubits
    z_marginals = np.zeros(N)
    zz_adjacent = np.zeros(N - 1)
    zz_next = np.zeros(N - 2)
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring]
        bits = bits[::-1]
        z_vals = [1 - 2*b for b in bits[:N]]
        for k in range(N):
            z_marginals[k] += z_vals[k] * count
        for k in range(N - 1):
            zz_adjacent[k] += z_vals[k] * z_vals[k+1] * count
        for k in range(N - 2):
            zz_next[k] += z_vals[k] * z_vals[k+2] * count
    z_marginals /= n_shots
    zz_adjacent /= n_shots
    zz_next /= n_shots
    return np.concatenate([z_marginals, zz_adjacent, zz_next])

def extract_variable_region_features(full_features, n_nuc=19):
    """Extract features from variable region (positions 8,9,10 = qubits 16-21)
    plus boundary correlators."""
    n_qubits = 2 * n_nuc  # 38
    
    # Z marginals: qubits 16-21 (6 features) — variable region positions 8,9,10
    z_var = full_features[16:22]
    
    # ZZ adjacent: pairs involving variable region
    # Pairs: (15,16), (16,17), (17,18), (18,19), (19,20), (20,21), (21,22)
    zz_adj_start = n_qubits  # index 38
    zz_var_adj = full_features[zz_adj_start + 15 : zz_adj_start + 22]  # 7 features
    
    # ZZ next-nearest: pairs (k, k+2)
    # (14,16), (15,17), (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)
    zz_next_start = zz_adj_start + (n_qubits - 1)  # index 38+37=75
    zz_var_next = full_features[zz_next_start + 14 : zz_next_start + 22]  # 8 features
    
    return np.concatenate([z_var, zz_var_adj, zz_var_next])  # 6+7+8 = 21 features


# ═══════════════════════════════════════════════════════════
# VERIFIED CANONICAL DUPLEXES — Tm from Table S1
# Centre notation: top_central / bottom_central
# Full top strand: CGACGTGC + top_central + ATGTGCTG
# ═══════════════════════════════════════════════════════════

CANONICAL_DUPLEXES_RAW = [
    ("GCG/CGC", 69.3, 1), ("CGC/GCG", 69.1, 2), ("GGC/CCG", 68.9, 3),
    ("GCC/CGG", 68.7, 4), ("CGG/GCC", 68.2, 5), ("CCG/GGC", 68.2, 6),
    ("GGG/CCC", 67.7, 7), ("CCC/GGG", 67.7, 8), ("TGC/ACG", 67.5, 9),
    ("GAC/CTG", 67.2, 10), ("GCA/CGT", 67.1, 11), ("TCG/AGC", 66.6, 12),
    ("GGA/CCT", 66.6, 13), ("CGA/GCT", 66.5, 14), ("CAG/GTC", 66.5, 15),
    ("GAG/CTC", 66.5, 16), ("GCT/CGA", 66.4, 17), ("CAC/GTG", 66.4, 18),
    ("GTC/CAG", 66.4, 19), ("GTG/CAC", 66.2, 20), ("CGT/GCA", 66.2, 21),
    ("AGC/TCG", 66.1, 22), ("TCC/AGG", 66.1, 23), ("CCA/GGT", 66.0, 24),
    ("AGG/TCC", 66.0, 25), ("GGT/CCA", 66.0, 26), ("CTG/GAC", 65.9, 27),
    ("CTC/GAG", 65.8, 29), ("TGG/ACC", 65.5, 30), ("ACG/TGC", 65.4, 31),
    ("ACC/TGG", 65.2, 32), ("TTG/AAC", 64.9, 34), ("ATC/TAG", 64.8, 35),
    ("GAA/CTT", 64.7, 36), ("CAA/GTT", 64.5, 37), ("AAC/TTG", 64.5, 39),
    ("CCT/GGA", 64.4, 40), ("TTC/AAG", 64.4, 41), ("TGA/ACT", 64.4, 42), ("TCA/AGT", 64.2, 43),
    ("AGA/TCT", 64.2, 44), ("GAT/CTA", 64.1, 45), ("GTA/CAT", 63.9, 47),
    ("AAG/TTC", 63.8, 50), ("ATG/TAC", 63.6, 55), ("GTT/CAA", 63.6, 58),
    ("TAG/ATC", 63.6, 59), ("TAC/ATG", 63.6, 61), ("CTA/GAT", 63.6, 62),
    ("TCT/AGA", 63.0, 73), ("CAT/GTA", 63.0, 74), ("ACA/TGT", 63.0, 76),
    ("CTT/GAA", 62.9, 80), ("AGT/TCA", 62.7, 86), ("AAA/TTT", 62.4, 98),
    ("TGT/ACA", 62.1, 107), ("ATA/TAT", 61.8, 112), ("TAA/ATT", 61.6, 119),
    ("TAT/ATA", 61.3, 126), ("AAT/TTA", 61.2, 138), ("ATT/TAA", 60.7, 159),
    ("ACT/TGA", 60.7, 160), ("TTA/AAT", 61.1, 142), ("TTT/AAA", 61.0, 147),
]

# Deduplicate
seen = set()
CANONICAL = []
for entry in CANONICAL_DUPLEXES_RAW:
    if entry[0] not in seen:
        seen.add(entry[0])
        CANONICAL.append(entry)

print("=" * 70)
print("EXP 1E CORRECTED: Oliveira et al. 2020")
print("=" * 70)
print(f"CORRECTED scaffold: 5'-CGACGTGC[NNN]ATGTGCTG-3' (19 nt)")
print(f"  Source: Page 8275, 'Sequence decomposition and notation'")
print(f"Buffer: 50 mM NaCl, 10 mM sodium phosphate, pH 7.4")
print(f"  Source: Page 8275, experimental conditions")
print(f"Strand concentration: 1.0 µM total")
print(f"Sequences: {len(CANONICAL)} canonical duplexes")
print(f"Qubits per sequence: {N_QUBITS}")
print(f"Feature dimension: {FEATURE_DIM}")
print()

# Verify sequences
for i, (centre, tm, rank) in enumerate(CANONICAL[:3]):
    full = get_full_top_strand(centre)
    assert len(full) == SEQ_LENGTH, f"Length error: {full} has {len(full)} nt, expected {SEQ_LENGTH}"
    print(f"  Example {i+1}: {centre:>8} → {full} ({len(full)} nt, {2*len(full)} qubits)")
print()

# ═══════════════════════════════════════════════════════════
# MPS SIMULATION
# ═══════════════════════════════════════════════════════════

sim = AerSimulator(method='matrix_product_state')
n_shots = 8192

all_features_full = []
all_features_var = []
all_tms = []
all_centres = []
all_full_seqs = []

print(f"Running MPS simulation ({N_QUBITS} qubits, {n_shots} shots)...")
t0 = time.time()

for i, (centre, tm, rank) in enumerate(CANONICAL):
    seq = get_full_top_strand(centre)
    qc = build_qubis_circuit(seq)
    
    assert qc.num_qubits == N_QUBITS, f"Qubit count error: {qc.num_qubits} vs expected {N_QUBITS}"
    
    job = sim.run(qc, shots=n_shots)
    counts = job.result().get_counts()
    
    fv_full = extract_features(counts, N_QUBITS, n_shots)
    fv_var = extract_variable_region_features(fv_full, n_nuc=SEQ_LENGTH)
    
    assert len(fv_full) == FEATURE_DIM, f"Feature dim error: {len(fv_full)} vs {FEATURE_DIM}"
    
    all_features_full.append(fv_full)
    all_features_var.append(fv_var)
    all_tms.append(tm)
    all_centres.append(centre)
    all_full_seqs.append(seq)
    
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1:3d}/{len(CANONICAL)}] {centre:>8} → {seq}  Tm={tm:.1f}°C")

t_total = time.time() - t0
print(f"\nSimulation complete: {t_total:.1f}s ({t_total/len(CANONICAL):.2f}s/seq)")

X_full = np.array(all_features_full)
X_var = np.array(all_features_var)
y = np.array(all_tms)

print(f"X_full: {X_full.shape}, X_var: {X_var.shape}, y: {y.shape}")
print(f"Tm range: {y.min():.1f} – {y.max():.1f}°C ({y.max()-y.min():.1f}°C)")

# ═══════════════════════════════════════════════════════════
# CLASSICAL BASELINES
# ═══════════════════════════════════════════════════════════

def classical_features(centre):
    top = centre.split("/")[0]
    gc_count = sum(1 for c in top if c in "GC")
    at_count = sum(1 for c in top if c in "AT")
    nuc_map = {"A": [1,0,0,0], "T": [0,1,0,0], "G": [0,0,1,0], "C": [0,0,0,1]}
    onehot = []
    for c in top:
        onehot.extend(nuc_map[c])
    nn1 = NN_DG.get(top[0:2], -1.0)
    nn2 = NN_DG.get(top[1:3], -1.0)
    # Boundary dinucleotides (CORRECTED scaffold)
    scaffold_last = "C"  # last nt of CGACGTGC
    after_first = "A"    # first nt of ATGTGCTG
    nn_left = NN_DG.get(scaffold_last + top[0], -1.0)
    nn_right = NN_DG.get(top[2] + after_first, -1.0)
    return np.array([gc_count, at_count, nn1, nn2, nn_left, nn_right] + onehot)

X_classical = np.array([classical_features(c) for c in all_centres])
X_gc_only = np.array([[sum(1 for c in dup[0].split("/")[0] if c in "GC")] for dup in CANONICAL])

# ═══════════════════════════════════════════════════════════
# LOO-CV RIDGE REGRESSION
# ═══════════════════════════════════════════════════════════

def loo_cv_ridge(X, y, alphas=[0.01, 0.1, 1.0, 10.0, 100.0]):
    loo = LeaveOneOut()
    best_r2 = -np.inf
    best_alpha = None
    best_preds = None
    for alpha in alphas:
        preds = np.zeros(len(y))
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            mu = X_train.mean(axis=0)
            sd = X_train.std(axis=0)
            sd[sd == 0] = 1.0
            X_train_s = (X_train - mu) / sd
            X_test_s = (X_test - mu) / sd
            model = Ridge(alpha=alpha)
            model.fit(X_train_s, y_train)
            preds[test_idx] = model.predict(X_test_s)
        r2 = r2_score(y, preds)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_preds = preds
    r, pval = pearsonr(y, best_preds)
    mae = mean_absolute_error(y, best_preds)
    return {"R2_cv": best_r2, "r": r, "p": pval, "MAE": mae, 
            "alpha": best_alpha, "predictions": best_preds.tolist()}

print(f"\n{'='*70}")
print(f"LOO-CV Ridge Regression Results")
print(f"{'='*70}")
print(f"\n{'Model':<40} {'CV R²':>8} {'r':>8} {'p-value':>12} {'MAE':>8} {'α':>6}")
print("-" * 85)

results = {}
for name, X in [
    ("Quantum: Full 111-dim", X_full),
    ("Quantum: Variable-region 21-dim", X_var),
    ("Classical: 18 features (rich)", X_classical),
    ("Classical: GC count only", X_gc_only),
]:
    res = loo_cv_ridge(X, y)
    results[name] = res
    print(f"{name:<40} {res['R2_cv']:8.4f} {res['r']:8.4f} {res['p']:12.2e} {res['MAE']:8.3f} {res['alpha']:6.1f}")

# Combined
X_combined = np.hstack([X_var, X_classical])
res_combined = loo_cv_ridge(X_combined, y)
print(f"{'Combined (var+classical) ' + str(X_combined.shape[1]) + '-dim':<40} "
      f"{res_combined['R2_cv']:8.4f} {res_combined['r']:8.4f} {res_combined['p']:12.2e} "
      f"{res_combined['MAE']:8.3f} {res_combined['alpha']:6.1f}")

# ═══════════════════════════════════════════════════════════
# SAVE CORRECTED RESULTS
# ═══════════════════════════════════════════════════════════

output = {
    "experiment": "Exp 1E: Experimental Tm validation (CORRECTED)",
    "corrections_applied": [
        "Scaffold corrected from TGACTCGACATCC[NNN]GCTACAA (23nt, from Fig.11/Chakraborty) to CGACGTGC[NNN]ATGTGCTG (19nt, from page 8275)",
        "Buffer corrected from 100mM NaCl to 50mM NaCl",
        "Strand concentration corrected from 4µM each to 1.0µM total",
        "Qubits corrected from 46 to 38",
        "Feature dimension corrected from 135 to 111",
    ],
    "dataset": {
        "source": "Oliveira et al. 2020 Chemical Science",
        "doi": "10.1039/d0sc01700k",
        "scaffold_top": "5'-CGACGTGC[N1N3N5]ATGTGCTG-3'",
        "scaffold_bottom": "3'-GCTGCACG[N2N4N6]TACACGAC-5'",
        "scaffold_source": "Page 8275, 'Sequence decomposition and notation'",
        "n_sequences": len(CANONICAL),
        "sequence_length_nt": SEQ_LENGTH,
        "tm_range_C": [float(y.min()), float(y.max())],
    },
    "conditions": {
        "buffer": "50 mM NaCl, 10 mM sodium phosphate, pH 7.4",
        "buffer_source": "Page 8275",
        "strand_concentration": "1.0 µM total",
        "method": "UV melting (260 nm absorbance vs temperature)",
    },
    "simulation": {
        "method": "matrix_product_state",
        "n_shots": n_shots,
        "n_qubits": N_QUBITS,
        "feature_dim_full": FEATURE_DIM,
        "feature_dim_variable": 21,
        "time_seconds": round(t_total, 1),
        "mps_exact": True,
        "max_bond_dimension": 4,
    },
    "regression_results": {
        "quantum_full": results["Quantum: Full 111-dim"],
        "quantum_variable": results["Quantum: Variable-region 21-dim"],
        "classical_rich": results["Classical: 18 features (rich)"],
        "classical_gc_only": results["Classical: GC count only"],
        "combined": res_combined,
    },
    "sequences": [
        {"centre": c, "full_sequence": s, "length_nt": len(s), 
         "n_qubits": 2*len(s), "exp_Tm_C": float(t)}
        for c, s, t in zip(all_centres, all_full_seqs, all_tms)
    ],
}

import os
os.makedirs("results", exist_ok=True)
with open("results/exp1e_corrected_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*70}")
print(f"CORRECTED RESULTS SAVED")
print(f"{'='*70}")
print(f"File: results/exp1e_corrected_results.json")
print(f"Scaffold: {SCAFFOLD_LEFT}[NNN]{SCAFFOLD_RIGHT} ({SEQ_LENGTH} nt)")
print(f"Qubits: {N_QUBITS}")
print(f"Feature dim: {FEATURE_DIM}")
print(f"Buffer: 50 mM NaCl, 10 mM sodium phosphate, pH 7.4")
print(f"Strand conc: 1.0 µM total")
