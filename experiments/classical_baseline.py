"""Classical ML Baseline Comparison for QuBiS-HiQ.

Compares classical ML models against the quantum circuit across two tasks:
  Task A: dG° regression (equivalent to Exp 1A)
    Task B: structural classification (equivalent to Exp 1C)
    """
import sys
import os
import json
import argparse
import numpy as np
from itertools import product as iproduct
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── SantaLucia Parameters (same as qubis_hiq/santalucia.py) ─────────────────
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
BETA = 1.0 / 0.593  # 1/kT at 298 K
NN_DG_TABLE = {
      "AA": -1.00, "AT": -0.88, "AG": -1.28, "AC": -1.44,
      "TA": -0.58, "TT": -1.00, "TG": -1.45, "TC": -1.30,
      "GA": -1.30, "GT": -1.44, "GG": -1.84, "GC": -3.42,
      "CA": -1.45, "CT": -1.28, "CG": -2.17, "CC": -1.84,
}


def get_dg(dinuc):
      return NN_DG_TABLE.get(dinuc, -1.0)


def compute_total_dg(seq):
      s = seq.upper()
      dg = sum(get_dg(s[i:i+2]) for i in range(len(s)-1))
      if s[0] in "AT" or s[-1] in "AT":
                dg += 1.03
            if s[0] in "GC" or s[-1] in "GC":
                      dg += 0.98
                  return dg


def boltzmann_sigmoid(dg):
      x = -BETA * dg
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return np.pi * sig


# ── Heuristic structure prediction (same as qubis_hiq/vienna_interface.py) ──
def predict_structure(seq):
      s = seq.upper()
    N = len(s)
    pairs = []
    left, right = 0, N - 1
    while left < right - 2:
              if COMPLEMENT.get(s[left]) == s[right]:
                            pairs.append((left, right))
                            left += 1
                            right -= 1
else:
            break
      return pairs


# ── Classical feature vectors ─────────────────────────────────────────────────
def build_classical_feature_vector(seq, stem_pairs=None):
      """Rich classical features: NN params, one-hot composition, cross-terms, structure."""
    s = seq.upper()
    N = len(s)
    stem_pairs = stem_pairs or []
    features = []

    # Per-step dG° values (N-1 steps)
    dg_steps = [get_dg(s[i:i+2]) for i in range(N - 1)]
    features.extend(dg_steps)

    # Per-step Boltzmann angles
    features.extend([boltzmann_sigmoid(dg) for dg in dg_steps])

    # One-hot per position (N x 4)
    nuc_to_idx = {"A": 0, "T": 1, "G": 2, "C": 3}
    for i in range(N):
        one_hot = [0, 0, 0, 0]
              one_hot[nuc_to_idx.get(s[i], 0)] = 1
        features.extend(one_hot)

    # GC content
    features.append(sum(1 for c in s if c in "GC") / N)

    # All pairwise dG° products (cross-terms that quantum gets "for free")
    for i in range(len(dg_steps)):
              for j in range(i + 1, len(dg_steps)):
                            features.append(dg_steps[i] * dg_steps[j])

          # Structure features
          features.append(len(stem_pairs))
    hbond_count = sum(
              3 if s[a]+s[b] in ("GC", "CG") else 2
              for a, b in stem_pairs
              if s[a]+s[b] in ("GC", "CG", "AT", "TA")
    )
    features.append(hbond_count)

    paired = {idx for pair in stem_pairs for idx in pair}
    features.extend([1 if i in paired else 0 for i in range(N)])

    return np.array(features, dtype=np.float64)


def build_classical_feature_vector_minimal(seq, stem_pairs=None):
      """Minimal classical features: only what the quantum circuit uses as input."""
    s = seq.upper()
    N = len(s)
    stem_pairs = stem_pairs or []
    features = []

    for i in range(N - 1):
              features.append(get_dg(s[i:i+2]))

    encoding = {"A": (0.5, 0.5), "T": (0.5, -0.5), "G": (-0.5, 0.5), "C": (-0.5, -0.5)}
    for i in range(N):
              z0, z1 = encoding.get(s[i], (0, 0))
              features.extend([z0, z1])

    paired = {idx for pair in stem_pairs for idx in pair}
    features.extend([1.0 if i in paired else 0.0 for i in range(N)])

    return np.array(features, dtype=np.float64)


# ── Task A: dG° regression ────────────────────────────────────────────────────
def run_regression_comparison(n_sequences=5000, seed=42):
      print(f"\n{'='*65}")
    print("TASK A: dG° Regression — Classical ML vs Quantum Features")
    print(f"{'='*65}")
    np.random.seed(seed)

    all_nucs = list("ATGC")
    seqs, dgs = [], []
    if n_sequences >= 65536:
              for combo in iproduct("ATGC", repeat=8):
                            seq = "".join(combo)
                            seqs.append(seq)
                            dgs.append(compute_total_dg(seq))
    else:
        while len(seqs) < n_sequences:
            seq = "".join(np.random.choice(all_nucs, 8))
                      try:
                dgs.append(compute_total_dg(seq))
                  seqs.append(seq)
except Exception:
                continue

    dgs = np.array(dgs)
    print(f"  Sequences: {len(seqs)}, dG° range: [{dgs.min():.2f}, {dgs.max():.2f}]")

    X_rich = np.array([build_classical_feature_vector(s, predict_structure(s)) for s in seqs])
    X_min = np.array([build_classical_feature_vector_minimal(s, predict_structure(s))
                                            for s in seqs])

    print(f"  Rich features: {X_rich.shape[1]}d | Minimal: {X_min.shape[1]}d | Quantum: 45d")

    models = {
                                "Ridge (a=1.0)": Ridge(alpha=1.0),
              "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "SVR-RBF": Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf", C=10))]),
}

                            results = {}
                            print(f"\n  {'Model':<25} {'Features':<14} {'CV R2 (mean±std)'}")
    print(f"  {'-'*58}")
                            for feat_name, X in [("Rich", X_rich), ("Minimal", X_min)]:
                                      for model_name, model in models.items():
                                    sc = cross_val_score(model, X, dgs, cv=5, scoring="r2")
            key = f"{model_name}_{feat_name}"
            results[key] = {"r2_mean": float(sc.mean()), "r2_std": float(sc.std())}
                                    print(f"  {model_name:<25} {feat_name:<14} {sc.mean():.4f} ± {sc.std():.4f}")

    print(f"\n  {'QuBiS-HiQ quantum':<25} {'45d':<14} 0.764 ± 0.055  (from Exp 1A)")
    return results


# ── Task B: structural classification ─────────────────────────────────────────
def run_classification_comparison(n_pairs=200, seed=42):
                            print(f"\n{'='*65}")
      print("TASK B: Structural Classification — Classical ML vs Quantum")
                            print(f"{'='*65}")
    np.random.seed(seed)

    pairs = []
    attempts = 0
          while len(pairs) < n_pairs and attempts < n_pairs * 200:
                    attempts += 1
        seq = "".join(np.random.choice(list("ATGC"), 8))
                                sp = predict_structure(seq)
        if not sp:
                      continue
        dg = compute_total_dg(seq)
        for _ in range(50):
            seq2 = "".join(np.random.choice(list("ATGC"), 8))
                                    sp2 = predict_structure(seq2)
            if not sp2 and abs(compute_total_dg(seq2) - dg) < 1.0:
                pairs.append((seq, sp, seq2, sp2))
                break

    print(f"  Found {len(pairs)} matched pairs")
    if len(pairs) < 20:
        return {"error": "insufficient_pairs", "n_pairs": len(pairs)}

    X_rich_s, X_rich_u = [], []
    X_min_s, X_min_u = [], []
    for seq1, sp1, seq2, sp2 in pairs:
        X_rich_s.append(build_classical_feature_vector(seq1, sp1))
        X_rich_u.append(build_classical_feature_vector(seq2, sp2))
        X_min_s.append(build_classical_feature_vector_minimal(seq1, sp1))
          X_min_u.append(build_classical_feature_vector_minimal(seq2, sp2))

    classifiers = {
        "SVM-RBF": Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=10))]),
        "SVM-Linear": Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="linear", C=1))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
      }

          for feat_name, X_s, X_u in [
        ("Rich classical", X_rich_s, X_rich_u),
        ("Minimal classical", X_min_s, X_min_u),
                ]:
          X = np.vstack([np.array(X_s), np.array(X_u)])
                                  y = np.concatenate([np.ones(len(X_s)), np.zeros(len(X_u))])
                                             print(f"\n  Feature set: {feat_name} ({X.shape[1]}d)")
        for clf_name, clf in classifiers.items():
                                                    sc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            mark = "v" if sc.mean() >= 0.99 else "~" if sc.mean() >= 0.70 else "x"
            print(f"    {clf_name:<25} {sc.mean():.4f} ± {sc.std():.4f}  [{mark}]")

    print(f"\n  QuBiS-HiQ quantum (45d):   1.000 ± 0.000  [v]")
    return {"n_pairs": len(pairs)}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
                                  parser = argparse.ArgumentParser(description="Classical ML baseline comparison.")
                    parser.add_argument("--n-seqs-regression", type=int, default=5000)
    parser.add_argument("--n-pairs", type=int, default=200)
    args = parser.parse_args()

    print(f"{'='*65}")
    print(f"Classical ML Baseline Comparison")
    print(f"Started: {datetime.now()}")
    print(f"{'='*65}")

    reg_results = run_regression_comparison(args.n_seqs_regression)
    cls_results = run_classification_comparison(args.n_pairs)

    output = {
        "timestamp": str(datetime.now()),
              "regression": reg_results,
              "classification": cls_results,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/classical_baseline_results.json", "w") as f:
              json.dump(output, f, indent=2)
    print(f"\nResults saved to results/classical_baseline_results.json")
