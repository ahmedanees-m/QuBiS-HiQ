#!/usr/bin/env python3
"""
Entanglement Ablation Study — Oliveira 2020 Tm Prediction
==========================================================
Tests whether stacking-layer CX gates contribute to Tm prediction performance.

Circuit variants (matching exp1b conventions):
  full        — Encoding + Stacking (complete circuit)
  no_stacking — Encoding only       (skip_stacking=True)
  random      — Encoding + random-angle CX (random_angles=True)

NOTE: The Watson-Crick layer uses ViennaRNA stem-pair predictions.
For 19-nt DNA scaffold sequences with fixed flanks, ViennaRNA typically
predicts a stem-loop hairpin. We include a no_wc variant where stem-pair
interactions are omitted (stem_pairs=[]).

Protocol: LOO-CV Ridge regression predicting Tm (same as Exp 1E).
Uses MPS simulator (38 qubits per sequence — statevector not tractable).

KEY QUESTION: Do entangling gates add predictive power, or do local rotations
(encoding alone) capture the same Tm information?
"""

import sys, os, json, time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.vienna_interface import predict_structure

# ── SantaLucia parameters (inlined for portability) ──────────────────────────
NN_DG = {
    "AA": -1.00, "AT": -0.88, "AG": -1.28, "AC": -1.44,
    "TA": -0.58, "TT": -1.00, "TG": -1.45, "TC": -1.30,
    "GA": -1.30, "GT": -1.44, "GG": -1.84, "GC": -3.42,
    "CA": -1.45, "CT": -1.28, "CG": -2.17, "CC": -1.84,
}
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

# ── Feature extraction from MPS measurement counts ───────────────────────────

def extract_features_from_counts(counts, n_qubits):
    """Z marginals + ZZ adjacent + ZZ next-nearest (matches exp1e protocol)."""
    z_marg = np.zeros(n_qubits)
    zz_adj = np.zeros(n_qubits - 1)
    zz_nxt = np.zeros(n_qubits - 2)
    total  = sum(counts.values())
    for bs, cnt in counts.items():
        bits = [int(b) for b in bs][::-1]
        zv   = [1 - 2*b for b in bits[:n_qubits]]
        for k in range(n_qubits):
            z_marg[k] += zv[k] * cnt
        for k in range(n_qubits - 1):
            zz_adj[k] += zv[k] * zv[k+1] * cnt
        for k in range(n_qubits - 2):
            zz_nxt[k] += zv[k] * zv[k+2] * cnt
    return np.concatenate([z_marg/total, zz_adj/total, zz_nxt/total])


def extract_variable_region(full_fv, n_qubits=38):
    """Same variable-region slice as exp1e (qubits 16–21, 21 features)."""
    NQ = n_qubits
    z_var    = full_fv[16:22]
    zz_start = NQ
    zz_var   = full_fv[zz_start + 15: zz_start + 22]
    zn_start = zz_start + NQ - 1
    zn_var   = full_fv[zn_start + 14: zn_start + 22]
    return np.concatenate([z_var, zz_var, zn_var])


# ── LOO-CV Ridge (matches Exp 1E) ─────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("ENTANGLEMENT ABLATION STUDY — Oliveira 2020 Tm Prediction")
    print("Testing whether CX gates (stacking layer) add predictive power")
    print("=" * 70)

    full_seqs = [SCAFFOLD_L + c.split("/")[0] + SCAFFOLD_R for c, _ in CANONICAL]
    y = np.array([tm for _, tm in CANONICAL])
    N = len(full_seqs)
    SEQ_LEN = 19
    N_QUBITS = 38
    PARAMS = np.zeros(12)   # trainable params fixed to zero (same as Exp 1E)
    N_SHOTS = 8192

    sim = AerSimulator(method='matrix_product_state')

    # Pre-compute stem pairs for all sequences
    print(f"\nPredicting secondary structure for {N} sequences …")
    stem_pairs_list = []
    for seq in full_seqs:
        _, sp = predict_structure(seq)
        stem_pairs_list.append(sp)
    n_with_stems = sum(1 for sp in stem_pairs_list if sp)
    print(f"  {n_with_stems}/{N} sequences have predicted stem pairs")

    # Circuit variants to test
    variants = {
        "full":        dict(skip_wc=False, skip_stacking=False, random_angles=False),
        "no_wc":       dict(skip_wc=True,  skip_stacking=False, random_angles=False),
        "no_stacking": dict(skip_wc=False, skip_stacking=True,  random_angles=False),
        "no_wc_no_stacking": dict(skip_wc=True, skip_stacking=True, random_angles=False),
        "random":      dict(skip_wc=False, skip_stacking=False, random_angles=True),
    }

    variant_descriptions = {
        "full":               "Full circuit (Encoding + WC + Stacking)",
        "no_wc":              "No Watson-Crick layer (skip WC CRZ)",
        "no_stacking":        "No Stacking layer (skip CX+Ry)",
        "no_wc_no_stacking":  "Encoding only (no entanglement)",
        "random":             "Random angles (physics-uninformed CX)",
    }

    all_features_full_var  = {}   # {variant: X_full_features}
    all_features_var_var   = {}   # {variant: X_variable_region_features}

    for vname, vkwargs in variants.items():
        print(f"\n  Running variant: {variant_descriptions[vname]} …")
        t0 = time.time()
        feats_full = []
        feats_var  = []

        for i, (seq, stem_pairs) in enumerate(zip(full_seqs, stem_pairs_list)):
            qc = build_circuit(seq, stem_pairs, PARAMS,
                               include_measurement=True, **vkwargs)
            assert qc.num_qubits == N_QUBITS, \
                f"Qubit mismatch: {qc.num_qubits} vs {N_QUBITS}"
            qc_t = transpile(qc, sim)
            job = sim.run(qc_t, shots=N_SHOTS)
            counts = job.result().get_counts()

            fv_full = extract_features_from_counts(counts, N_QUBITS)
            fv_var  = extract_variable_region(fv_full, N_QUBITS)
            feats_full.append(fv_full)
            feats_var.append(fv_var)

            if (i + 1) % 16 == 0 or i == 0:
                print(f"    [{i+1:3d}/{N}]")

        t_var = time.time() - t0
        print(f"    Done in {t_var:.1f}s  (feature dim: {len(feats_full[0])})")
        all_features_full_var[vname] = np.array(feats_full)
        all_features_var_var[vname]  = np.array(feats_var)

    # ── LOO-CV Ridge for each variant ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LOO-CV Ridge Regression Results (Tm prediction)")
    print(f"{'='*70}")
    print(f"\n{'Variant':<36} {'LOO R²':>7} {'r':>6} {'MAE(°C)':>8} {'α':>6}")
    print("-" * 70)

    results = {}
    for vname in variants:
        res_full = loo_ridge(all_features_full_var[vname], y)
        res_var  = loo_ridge(all_features_var_var[vname],  y)
        results[vname] = {
            "description": variant_descriptions[vname],
            "full_features": res_full,
            "variable_region_features": res_var,
        }
        desc_trunc = variant_descriptions[vname][:34]
        print(f"{desc_trunc:<36} {res_full['R2_loo']:>7.4f} {res_full['r']:>6.4f} "
              f"{res_full['MAE']:>8.3f} {res_full['alpha']:>6.1f}")

    # Reference
    qref_path = os.path.join(os.path.dirname(__file__), "..",
                             "results", "exp1e_corrected_results.json")
    q_r2 = q_r = q_mae = None
    if os.path.exists(qref_path):
        with open(qref_path) as f:
            qdata = json.load(f)
        q_res = qdata.get("regression_results", {}).get("quantum_full", {})
        q_r2  = q_res.get("R2_cv")
        q_r   = q_res.get("r")
        q_mae = q_res.get("MAE")

    print("-" * 70)
    if q_r2 is not None:
        print(f"{'Exp 1E reference (full circuit)':<36} {q_r2:>7.4f} {q_r:>6.4f} "
              f"{q_mae:>8.3f}   —")
    else:
        print("(Exp 1E reference: R²=0.88, r=0.94, MAE=0.60°C)")

    # ── Contribution analysis ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ENTANGLEMENT CONTRIBUTION ANALYSIS")
    print(f"{'='*70}")

    r2_full   = results["full"]["full_features"]["R2_loo"]
    r2_enc    = results["no_wc_no_stacking"]["full_features"]["R2_loo"]
    r2_no_wc  = results["no_wc"]["full_features"]["R2_loo"]
    r2_no_st  = results["no_stacking"]["full_features"]["R2_loo"]
    r2_random = results["random"]["full_features"]["R2_loo"]

    contrib_entanglement = r2_full - r2_enc
    contrib_wc           = r2_full - r2_no_wc
    contrib_stacking     = r2_full - r2_no_st
    contrib_physics_vs_random = r2_full - r2_random

    print(f"\n  Full circuit LOO R²:         {r2_full:.4f}")
    print(f"  Encoding-only LOO R²:        {r2_enc:.4f}")
    print(f"  Entanglement contribution:   {contrib_entanglement:+.4f}")
    print(f"  Watson-Crick contribution:   {contrib_wc:+.4f}")
    print(f"  Stacking contribution:       {contrib_stacking:+.4f}")
    print(f"  Physics vs random CX:        {contrib_physics_vs_random:+.4f}")

    threshold = 0.05
    ent_necessary      = contrib_entanglement > threshold
    physics_matters    = contrib_physics_vs_random > threshold
    wc_matters         = contrib_wc > threshold
    stacking_matters   = contrib_stacking > threshold

    print(f"\n  Entanglement necessary?      {'YES' if ent_necessary else 'NO'} "
          f"(threshold: >{threshold:.2f})")
    print(f"  Physics-informed CX better? {'YES' if physics_matters else 'NO'}")
    print(f"  Watson-Crick layer matters?  {'YES' if wc_matters else 'NO'}")
    print(f"  Stacking layer matters?      {'YES' if stacking_matters else 'NO'}")

    if ent_necessary and physics_matters:
        conclusion = "Physics-informed entanglement is NECESSARY and STRUCTURED — quantum approach fully validated."
    elif ent_necessary and not physics_matters:
        conclusion = "Entanglement adds value but physics structure doesn't matter — any CX pattern helps equally."
    elif not ent_necessary:
        conclusion = "Encoding-only circuit is sufficient — local Ry rotations capture Tm information without entanglement."
    else:
        conclusion = "Mixed result — entanglement effect is below significance threshold."

    print(f"\n  Conclusion: {conclusion}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "experiment": "Entanglement Ablation Study — Oliveira 2020 Tm",
        "dataset": {
            "source": "Oliveira et al. 2020", "n_sequences": N,
            "scaffold_top": f"5'-{SCAFFOLD_L}[NNN]{SCAFFOLD_R}-3'",
            "tm_range_C": [float(y.min()), float(y.max())],
        },
        "simulation": {
            "method": "matrix_product_state", "n_shots": N_SHOTS,
            "n_qubits": N_QUBITS, "trainable_params": "zeros (12)",
        },
        "variant_results": results,
        "contribution_analysis": {
            "full_R2":                float(r2_full),
            "encoding_only_R2":       float(r2_enc),
            "entanglement_contrib":   float(contrib_entanglement),
            "wc_layer_contrib":       float(contrib_wc),
            "stacking_layer_contrib": float(contrib_stacking),
            "physics_vs_random":      float(contrib_physics_vs_random),
            "entanglement_necessary": ent_necessary,
            "physics_informed_matters": physics_matters,
            "wc_matters": wc_matters,
            "stacking_matters": stacking_matters,
        },
        "quantum_reference": {
            "source": "Exp 1E corrected", "R2_loo": q_r2, "r": q_r, "MAE": q_mae,
        },
        "conclusion": conclusion,
    }

    out_path = "results/entanglement_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
