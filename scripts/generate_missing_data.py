"""Generate all missing raw data files for Zenodo deposition.

Produces:
  results/exp1e/quantum_features_111d.npy     — 64 × 111 MPS quantum features
  results/exp1e/quantum_features_21d.npy      — 64 × 21  variable-region slice
  results/exp1e/classical_features_17d.npy    — 64 × 17  classical NN features
  results/exp1e/combined_features_38d.npy     — 64 × 38  combined (quantum + classical)
  results/exp1d/sim_features_45d.npy          — 30 × 45  statevector features (8-mer)
  results/predictions/exp1e_predictions.csv   — LOO predictions, all model variants
  results/predictions/best_config_predictions.csv — best-config LOO predictions
  results/predictions/x2_predictions.csv      — X2 stacking-only LOO predictions
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import csv
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR     = os.path.join(ROOT, 'data')
EXP1E_OUT    = os.path.join(ROOT, 'results', 'exp1e')
EXP1D_OUT    = os.path.join(ROOT, 'results', 'exp1d')
PRED_OUT     = os.path.join(ROOT, 'results', 'predictions')
for d in [EXP1E_OUT, EXP1D_OUT, PRED_OUT]:
    os.makedirs(d, exist_ok=True)

# ── SantaLucia parameters (no-WC stacking-only, matching exp1e) ─────────────
BETA = 0.39
NN_DG = {
    "AA": -1.00, "AT": -0.88, "AG": -1.28, "AC": -1.44,
    "TA": -0.58, "TT": -1.00, "TG": -1.45, "TC": -1.30,
    "GA": -1.30, "GT": -1.44, "GG": -1.84, "GC": -3.42,
    "CA": -1.45, "CT": -1.28, "CG": -2.17, "CC": -1.84,
}
SCAFFOLD_LEFT  = "CGACGTGC"
SCAFFOLD_RIGHT = "ATGTGCTG"
N_SHOTS = 8192

def boltzmann_angle(dg):
    return np.pi / (1.0 + np.exp(BETA * dg))

# ── Circuit builder (matches exp1e_corrected.py exactly) ─────────────────────
def build_stacking_circuit(seq):
    """No-WC stacking-only circuit (skip_wc=True equivalent)."""
    N = len(seq)
    qc = QuantumCircuit(2 * N)
    enc = {"A": (0, 0), "T": (0, 1), "G": (1, 0), "C": (1, 1)}
    ang = {0: np.pi / 3, 1: 2 * np.pi / 3}
    for i, nuc in enumerate(seq.upper()):
        bits = enc[nuc]
        qc.ry(ang[bits[0]], 2 * i)
        qc.ry(ang[bits[1]], 2 * i + 1)
    for k in range(N - 1):
        dinuc = seq[k:k + 2].upper()
        dg = NN_DG.get(dinuc, -1.0)
        theta_s = boltzmann_angle(dg)
        qc.cx(2 * k, 2 * (k + 1))
        qc.ry(theta_s, 2 * (k + 1))
    qc.measure_all()
    return qc

# ── Feature extractor ─────────────────────────────────────────────────────────
def extract_features(counts, n_qubits, n_shots):
    N = n_qubits
    z  = np.zeros(N)
    zz = np.zeros(N - 1)
    zz2 = np.zeros(N - 2)
    for bitstring, cnt in counts.items():
        bits = [int(b) for b in bitstring][::-1]
        zv = [1 - 2 * b for b in bits[:N]]
        for k in range(N):      z[k]   += zv[k] * cnt
        for k in range(N - 1):  zz[k]  += zv[k] * zv[k + 1] * cnt
        for k in range(N - 2):  zz2[k] += zv[k] * zv[k + 2] * cnt
    return np.concatenate([z, zz, zz2]) / n_shots

# ── Classical 17-d feature vector (matches classical_physics_baseline.py) ─────
NN_DH = {
    "AA": -7.9, "AT": -7.2, "AG": -7.8, "AC": -7.8,
    "TA": -7.2, "TT": -7.9, "TG": -8.5, "TC": -8.2,
    "GA": -8.2, "GT": -8.4, "GG": -8.0, "GC": -9.8,
    "CA": -8.5, "CT": -7.8, "CG": -10.6,"CC": -8.0,
}
NN_DS = {k: (NN_DH[k] - NN_DG[k]) * 1000 / 310.15 for k in NN_DG}

def feat_classical_17d(seq):
    """17-d classical variable-region feature vector."""
    centre = seq[8:11]           # 3-nt variable region
    full   = seq                  # 19-nt full sequence
    gc = (full.count('G') + full.count('C')) / len(full)
    # 2 NN steps spanning the variable region boundary
    dg_7_8  = NN_DG.get(full[7:9].upper(),  -1.0)
    dg_10_11 = NN_DG.get(full[10:12].upper(), -1.0)
    # boundary NNs on both sides
    dg_left  = NN_DG.get(full[6:8].upper(),  -1.0)
    dg_right = NN_DG.get(full[11:13].upper(), -1.0)
    # one-hot NNN centre (4^3 = 64 → truncated to 12 for A/T/G/C × position)
    nucs = "ATGC"
    oh = np.zeros(12)
    for pos, nuc in enumerate(centre.upper()):
        if nuc in nucs:
            oh[pos * 4 + nucs.index(nuc)] = 1.0
    return np.array([gc, dg_7_8, dg_10_11, dg_left, dg_right] + oh.tolist())

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: 19-mer quantum features (MPS simulation, 64 Oliveira sequences)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Part 1: 19-mer quantum features via MPS (64 sequences)")
print("=" * 60)

with open(os.path.join(DATA_DIR, 'oliveira2020_corrected_dataset.json')) as f:
    oliv = json.load(f)

duplexes   = oliv['duplexes']
sequences  = [d['full_sequence'] for d in duplexes]
tm_values  = [d['exp_Tm_C'] for d in duplexes]
centres    = [d['centre'] for d in duplexes]
assert len(sequences) == 64

sim = AerSimulator(method='matrix_product_state')
N_Q = 38   # 2 × 19
FEAT_DIM = 6 * 19 - 3  # 111

features_111d = np.zeros((64, FEAT_DIM))
t0 = time.time()

for i, seq in enumerate(sequences):
    qc = build_stacking_circuit(seq)
    qc_t = transpile(qc, sim)
    job = sim.run(qc_t, shots=N_SHOTS)
    counts = job.result().get_counts()
    features_111d[i] = extract_features(counts, N_Q, N_SHOTS)
    if (i + 1) % 16 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/64 done  ({elapsed:.1f}s)")

print(f"  Complete in {time.time()-t0:.1f}s. Shape: {features_111d.shape}")

# Variable-region slice: Z[16:22] + ZZ_adj[NQ+15:NQ+22] + ZZ_nxt[NQ+NQ-1+14:NQ+NQ-1+22]
NQ = N_Q
z_var  = features_111d[:, 16:22]
zz_adj = features_111d[:, NQ + 15:NQ + 22]
zz_nxt = features_111d[:, NQ + NQ - 1 + 14:NQ + NQ - 1 + 22]
features_21d = np.concatenate([z_var, zz_adj, zz_nxt], axis=1)
print(f"  Variable-region 21-d slice: {features_21d.shape}")

# Classical 17-d features
features_17d = np.array([feat_classical_17d(s) for s in sequences])
print(f"  Classical 17-d features: {features_17d.shape}")

# Combined 38-d
features_38d = np.concatenate([features_21d, features_17d], axis=1)
print(f"  Combined 38-d features: {features_38d.shape}")

# Save
np.save(os.path.join(EXP1E_OUT, 'quantum_features_111d.npy'), features_111d)
np.save(os.path.join(EXP1E_OUT, 'quantum_features_21d.npy'),  features_21d)
np.save(os.path.join(EXP1E_OUT, 'classical_features_17d.npy'), features_17d)
np.save(os.path.join(EXP1E_OUT, 'combined_features_38d.npy'),  features_38d)
print("  Saved: quantum_features_111d.npy, quantum_features_21d.npy,")
print("         classical_features_17d.npy, combined_features_38d.npy")

# Also save sequence metadata as CSV for reference
meta_path = os.path.join(EXP1E_OUT, 'sequence_metadata.csv')
with open(meta_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'centre_NNN', 'full_sequence_19nt', 'Tm_C'])
    for i, (c, s, t) in enumerate(zip(centres, sequences, tm_values)):
        w.writerow([i, c, s, f"{t:.1f}"])
print(f"  Saved: sequence_metadata.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: 8-mer statevector features for exp1d sequences (30 sequences × 45-d)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 2: 8-mer statevector features (30 exp1d sequences)")
print("=" * 60)

with open(os.path.join(ROOT, 'results', 'exp1d_results.json')) as f:
    exp1d = json.load(f)

seqs_8mer  = exp1d['sequences']
labels_8mer = exp1d['seq_labels']
N_8 = 8
N_Q_8 = 16
FEAT_8 = 6 * N_8 - 3  # 45
N_SHOTS_SV = 8192

sim_features_45d = np.zeros((30, FEAT_8))
for i, seq in enumerate(seqs_8mer):
    qc = build_stacking_circuit(seq)
    sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
    counts = {(format(k, f'0{N_Q_8}b') if isinstance(k, int) else str(k)): v
              for k, v in sv.sample_counts(N_SHOTS_SV).items()}
    sim_features_45d[i] = extract_features(counts, N_Q_8, N_SHOTS_SV)

np.save(os.path.join(EXP1D_OUT, 'sim_features_45d.npy'), sim_features_45d)
print(f"  Saved: sim_features_45d.npy  shape={sim_features_45d.shape}")

# Also save 8-mer sequence metadata
meta_8mer_path = os.path.join(EXP1D_OUT, 'sequence_metadata.csv')
with open(meta_8mer_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'sequence', 'gc_group', 'gc_content'])
    for i, (s, l) in enumerate(zip(seqs_8mer, labels_8mer)):
        gc = (s.count('G') + s.count('C')) / len(s)
        w.writerow([i, s, l, f"{gc:.3f}"])
print(f"  Saved: sequence_metadata.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Regression predictions as CSV
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 3: Regression predictions as CSV")
print("=" * 60)

tm_true = np.array(tm_values)

# --- exp1e predictions ---
with open(os.path.join(ROOT, 'results', 'exp1e_corrected_results.json')) as f:
    exp1e = json.load(f)

rr = exp1e['regression_results']
pred_path = os.path.join(PRED_OUT, 'exp1e_predictions.csv')
with open(pred_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'centre_NNN', 'full_sequence', 'Tm_true_C',
                'pred_quantum_full_111d', 'pred_quantum_var_21d',
                'pred_classical_rich', 'pred_classical_gc_only', 'pred_combined'])
    for i in range(64):
        w.writerow([
            i, centres[i], sequences[i], f"{tm_true[i]:.2f}",
            f"{rr['quantum_full']['predictions'][i]:.4f}",
            f"{rr['quantum_variable']['predictions'][i]:.4f}",
            f"{rr['classical_rich']['predictions'][i]:.4f}",
            f"{rr['classical_gc_only']['predictions'][i]:.4f}",
            f"{rr['combined']['predictions'][i]:.4f}",
        ])
print(f"  Saved: exp1e_predictions.csv")

# --- best_configuration predictions ---
with open(os.path.join(ROOT, 'results', 'best_configuration_results.json')) as f:
    best = json.load(f)

cfg = best['configurations']
pred_path2 = os.path.join(PRED_OUT, 'best_config_predictions.csv')
with open(pred_path2, 'w', newline='') as f:
    w = csv.writer(f)
    col_names = list(cfg.keys())
    header = ['index', 'centre_NNN', 'full_sequence', 'Tm_true_C'] + \
             [f"pred_{c.strip().replace(' ', '_').replace('(','').replace(')','').replace(',','')}" for c in col_names]
    w.writerow(header)
    for i in range(64):
        row = [i, centres[i], sequences[i], f"{tm_true[i]:.2f}"]
        for c in col_names:
            row.append(f"{cfg[c]['predictions'][i]:.4f}")
        w.writerow(row)
print(f"  Saved: best_config_predictions.csv")

# --- x2 stacking-only predictions ---
with open(os.path.join(ROOT, 'results', 'x2_stacking_only_results.json')) as f:
    x2 = json.load(f)

x2_configs = x2.get('configurations', x2.get('results', {}))
pred_path3 = os.path.join(PRED_OUT, 'x2_predictions.csv')
with open(pred_path3, 'w', newline='') as f:
    w = csv.writer(f)
    x2_cols = list(x2_configs.keys())
    header = ['index', 'centre_NNN', 'full_sequence', 'Tm_true_C'] + \
             [f"pred_{c.strip().replace(' ', '_').replace('(','').replace(')','').replace(',','')}" for c in x2_cols]
    w.writerow(header)
    preds_available = all('predictions' in x2_configs[c] for c in x2_cols)
    if preds_available:
        for i in range(64):
            row = [i, centres[i], sequences[i], f"{tm_true[i]:.2f}"]
            for c in x2_cols:
                row.append(f"{x2_configs[c]['predictions'][i]:.4f}")
            w.writerow(row)
        print(f"  Saved: x2_predictions.csv")
    else:
        print(f"  x2 predictions not stored in JSON — skipping CSV export")
        os.remove(pred_path3)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPLETE. Files generated:")
for root_dir, dirs, files in os.walk(os.path.join(ROOT, 'results')):
    for fn in sorted(files):
        if fn.endswith(('.npy', '.csv')):
            rel = os.path.relpath(os.path.join(root_dir, fn), ROOT)
            size = os.path.getsize(os.path.join(root_dir, fn))
            print(f"  {rel}  ({size/1024:.1f} KB)")
