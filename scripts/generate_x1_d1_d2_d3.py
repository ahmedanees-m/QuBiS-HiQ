"""Generate missing raw data for X1, D1, D2, D3.

Outputs
-------
results/x1/
    hairpin_sequences.csv       — all 58 hairpin 8-mers with stem_pairs, dG, Tm
    linear_sequences.csv        — balanced 58 linear 8-mers with dG, Tm
    hairpin_features.npy        — 58 × 45 quantum features (full circuit)
    linear_features.npy         — 58 × 45 quantum features (full circuit)

results/d1/
    kernel_8mer_50x50.npy       — 50 × 50 exact quantum kernel matrix
    kernel_oliveira_64x64.npy   — 64 × 64 linear feature kernel matrix
    sequences_8mer.csv          — 50 random 8-mers used for kernel
    kernel_metadata.json        — condition numbers, effective ranks

results/d2/
    classical_baseline_predictions.csv

results/d3/
    ablation_predictions.csv
"""

import sys, os, json, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

ROOT = os.path.join(os.path.dirname(__file__), '..')

for d in ['results/x1', 'results/d1', 'results/d2', 'results/d3']:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

# ── Shared physics ────────────────────────────────────────────────────────────
NN_DG = {
    "AA": -1.00, "AT": -0.88, "AG": -1.28, "AC": -1.44,
    "TA": -0.58, "TT": -1.00, "TG": -1.45, "TC": -1.30,
    "GA": -1.30, "GT": -1.44, "GG": -1.84, "GC": -3.42,
    "CA": -1.45, "CT": -1.28, "CG": -2.17, "CC": -1.84,
}
NN_DH = {
    "AA": -7.9,  "AT": -7.2,  "AG": -7.8,  "AC": -7.8,
    "TA": -7.2,  "TT": -7.9,  "TG": -8.5,  "TC": -8.2,
    "GA": -8.2,  "GT": -8.4,  "GG": -8.0,  "GC": -9.8,
    "CA": -8.5,  "CT": -7.8,  "CG": -10.6, "CC": -8.0,
}
BETA = 0.39

def total_dg(seq):
    return sum(NN_DG.get(seq[i:i+2].upper(), -1.0) for i in range(len(seq)-1))

def santalucia_tm(seq, Na_mM=1000.0, strand_nM=250.0):
    dH = sum(NN_DH.get(seq[i:i+2].upper(), -8.0) for i in range(len(seq)-1)) * 1000
    dS = sum((NN_DH[seq[i:i+2].upper()] - NN_DG[seq[i:i+2].upper()]) * 1000 / 310.15
             for i in range(len(seq)-1) if seq[i:i+2].upper() in NN_DG)
    R = 1.987
    CT = strand_nM * 1e-9
    Na = Na_mM * 1e-3
    Tm_K = dH / (dS + R * np.log(CT / 4)) - 16.6 * np.log10(Na)
    return Tm_K - 273.15

def boltzmann_angle(dg):
    return np.pi / (1.0 + np.exp(BETA * dg))

def build_circuit_8mer(seq, skip_wc=False):
    from qiskit import QuantumCircuit
    N = len(seq)
    qc = QuantumCircuit(2 * N)
    enc = {"A": (0,0), "T": (0,1), "G": (1,0), "C": (1,1)}
    ang = {0: np.pi/3, 1: 2*np.pi/3}
    for i, nuc in enumerate(seq.upper()):
        bits = enc[nuc]; qc.ry(ang[bits[0]], 2*i); qc.ry(ang[bits[1]], 2*i+1)
    for k in range(N-1):
        qc.cx(2*k, 2*(k+1))
        qc.ry(boltzmann_angle(NN_DG.get(seq[k:k+2].upper(), -1.0)), 2*(k+1))
    return qc

def extract_features_8mer(sv_arr, n_q=16, n_shots=8192):
    from qiskit.quantum_info import Statevector
    sv = Statevector(sv_arr)
    counts = {(format(k, f'0{n_q}b') if isinstance(k, int) else str(k)): v
              for k, v in sv.sample_counts(n_shots).items()}
    z = np.zeros(n_q); zz = np.zeros(n_q-1); zz2 = np.zeros(n_q-2)
    for bs, cnt in counts.items():
        bits = [int(b) for b in bs][::-1]
        zv = [1-2*b for b in bits[:n_q]]
        for i in range(n_q):      z[i]   += zv[i]*cnt
        for i in range(n_q-1):    zz[i]  += zv[i]*zv[i+1]*cnt
        for i in range(n_q-2):    zz2[i] += zv[i]*zv[i+2]*cnt
    return np.concatenate([z, zz, zz2]) / n_shots

from qubis_hiq.vienna_interface import predict_structure as _predict_structure

def predict_structure_heuristic(seq):
    """Use the canonical predict_structure() — ViennaRNA if installed, heuristic otherwise."""
    _, pairs = _predict_structure(seq)
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# X1: Full hairpin / linear sequence lists + feature matrices
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("X1: Regenerating full hairpin/linear sequence sets")
print("=" * 60)

np.random.seed(42)
N_CANDIDATES = 65536
all_seqs = [''.join(np.random.choice(list('ATGC'), 8)) for _ in range(N_CANDIDATES)]

hairpin_seqs, hairpin_stems = [], []
linear_seqs = []
for seq in all_seqs:
    sp = predict_structure_heuristic(seq)
    if sp:
        hairpin_seqs.append(seq); hairpin_stems.append(sp)
    else:
        linear_seqs.append(seq)

print(f"  Hairpin: {len(hairpin_seqs)}, Linear: {len(linear_seqs)}")

# Balanced subset (same seed as X1 experiment)
N_EACH = min(len(hairpin_seqs), 58)
np.random.seed(42)
idx_h = np.random.choice(len(hairpin_seqs), N_EACH, replace=False)
idx_l = np.random.choice(len(linear_seqs),  N_EACH, replace=False)
sel_hairpin = [hairpin_seqs[i] for i in sorted(idx_h)]
sel_stems   = [hairpin_stems[i] for i in sorted(idx_h)]
sel_linear  = [linear_seqs[i]  for i in sorted(idx_l)]

# Save sequence CSVs
N_SHOTS = 8192
def write_seq_csv(path, seqs, stems_list=None):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['index','sequence','gc_content','total_dG_kcal_mol','Tm_approx_C']
        if stems_list is not None:
            header.append('stem_pairs')
        w.writerow(header)
        for i, seq in enumerate(seqs):
            gc = (seq.count('G')+seq.count('C'))/len(seq)
            dg = total_dg(seq)
            tm = santalucia_tm(seq)
            row = [i, seq, f"{gc:.3f}", f"{dg:.4f}", f"{tm:.2f}"]
            if stems_list is not None:
                row.append(str(stems_list[i]))
            w.writerow(row)

write_seq_csv(os.path.join(ROOT, 'results/x1/hairpin_sequences.csv'), sel_hairpin, sel_stems)
write_seq_csv(os.path.join(ROOT, 'results/x1/linear_sequences.csv'),  sel_linear)
print(f"  Saved: hairpin_sequences.csv ({len(sel_hairpin)} seqs), linear_sequences.csv ({len(sel_linear)} seqs)")

# Generate feature matrices (statevector, full circuit)
print(f"  Computing statevector features for {2*N_EACH} sequences …")
t0 = time.time()

def get_sv_features(seq, stem_pairs):
    qc = build_circuit_8mer(seq)
    sv = np.asarray(Statevector.from_instruction(qc))
    return extract_features_8mer(sv, n_q=16, n_shots=N_SHOTS)

feat_hairpin = np.array([get_sv_features(s, sp) for s, sp in zip(sel_hairpin, sel_stems)])
feat_linear  = np.array([get_sv_features(s, [])  for s in sel_linear])
print(f"  Done in {time.time()-t0:.1f}s. Shapes: {feat_hairpin.shape}, {feat_linear.shape}")

np.save(os.path.join(ROOT, 'results/x1/hairpin_features.npy'), feat_hairpin)
np.save(os.path.join(ROOT, 'results/x1/linear_features.npy'),  feat_linear)
print(f"  Saved: hairpin_features.npy, linear_features.npy")

# Also save dG targets
dg_hairpin = np.array([total_dg(s) for s in sel_hairpin])
dg_linear  = np.array([total_dg(s) for s in sel_linear])
np.save(os.path.join(ROOT, 'results/x1/hairpin_dg_targets.npy'), dg_hairpin)
np.save(os.path.join(ROOT, 'results/x1/linear_dg_targets.npy'),  dg_linear)
print(f"  Saved: hairpin_dg_targets.npy, linear_dg_targets.npy")


# ═══════════════════════════════════════════════════════════════════════
# D1: Kernel matrices
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("D1: Kernel matrices")
print("=" * 60)

# --- Part A: 50×50 exact quantum kernel on 8-mers ---
print("  [A] 50×50 exact quantum kernel (8-mers, statevector) …")
np.random.seed(42)
seqs_8 = [''.join(np.random.choice(list('ATGC'), 8)) for _ in range(50)]

t0 = time.time()
sv_list = []
for seq in seqs_8:
    qc = build_circuit_8mer(seq)
    sv_list.append(np.asarray(Statevector.from_instruction(qc)))

K_8 = np.zeros((50, 50))
for i in range(50):
    K_8[i, i] = 1.0
    for j in range(i+1, 50):
        v = abs(np.vdot(sv_list[i], sv_list[j]))**2
        K_8[i, j] = K_8[j, i] = v
print(f"  Done in {time.time()-t0:.1f}s")

np.save(os.path.join(ROOT, 'results/d1/kernel_8mer_50x50.npy'), K_8)
print(f"  Saved: kernel_8mer_50x50.npy  shape={K_8.shape}")

# Save 8-mer sequence list
with open(os.path.join(ROOT, 'results/d1/sequences_8mer.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'sequence', 'gc_content', 'total_dG_kcal_mol'])
    for i, seq in enumerate(seqs_8):
        gc = (seq.count('G')+seq.count('C'))/8
        w.writerow([i, seq, f"{gc:.3f}", f"{total_dg(seq):.4f}"])
print(f"  Saved: sequences_8mer.csv")

# --- Part B: 64×64 linear feature kernel on Oliveira sequences ---
print("  [B] 64×64 linear feature kernel (Oliveira, MPS features already computed) …")
feat_111 = np.load(os.path.join(ROOT, 'results/exp1e/quantum_features_111d.npy'))
# L2-normalise
norms = np.linalg.norm(feat_111, axis=1, keepdims=True)
norms[norms == 0] = 1.0
feat_norm = feat_111 / norms
K_oliv = feat_norm @ feat_norm.T
np.save(os.path.join(ROOT, 'results/d1/kernel_oliveira_64x64.npy'), K_oliv)
print(f"  Saved: kernel_oliveira_64x64.npy  shape={K_oliv.shape}")

# Kernel metadata
svd_8   = np.linalg.svd(K_8,    compute_uv=False)
svd_oliv = np.linalg.svd(K_oliv, compute_uv=False)
kernel_meta = {
    'kernel_8mer': {
        'shape': [50, 50], 'method': 'exact statevector inner product',
        'condition_number': float(svd_8[0]/max(svd_8[-1], 1e-15)),
        'effective_rank': int(np.sum(svd_8 > svd_8[0]*1e-6)),
        'is_psd': bool(np.all(svd_8 >= -1e-10)),
    },
    'kernel_oliveira': {
        'shape': [64, 64], 'method': 'linear (L2-normalised feature dot product)',
        'condition_number': float(svd_oliv[0]/max(svd_oliv[-1], 1e-15)),
        'effective_rank': int(np.sum(svd_oliv > svd_oliv[0]*1e-6)),
        'is_psd': bool(np.all(svd_oliv >= -1e-10)),
    }
}
with open(os.path.join(ROOT, 'results/d1/kernel_metadata.json'), 'w') as f:
    json.dump(kernel_meta, f, indent=2)
print(f"  Saved: kernel_metadata.json")
print(f"  κ(8-mer)={kernel_meta['kernel_8mer']['condition_number']:.1f}  "
      f"κ(Oliveira)={kernel_meta['kernel_oliveira']['condition_number']:.2e}")


# ═══════════════════════════════════════════════════════════════════════
# D2: Classical baseline predictions → CSV
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("D2: Classical baseline predictions → CSV")
print("=" * 60)

with open(os.path.join(ROOT, 'results/classical_physics_baseline_results.json')) as f:
    d2 = json.load(f)
with open(os.path.join(ROOT, 'data/oliveira2020_corrected_dataset.json')) as f:
    oliv = json.load(f)

centres   = [d['centre'] for d in oliv['duplexes']]
sequences = [d['full_sequence'] for d in oliv['duplexes']]
tm_true   = [d['exp_Tm_C'] for d in oliv['duplexes']]

cr = d2['classical_results']
model_keys = list(cr.keys())

with open(os.path.join(ROOT, 'results/d2/classical_baseline_predictions.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    safe = lambda s: s.replace(' ','_').replace('(','').replace(')','').replace('-','_').replace('+','')
    header = ['index','centre_NNN','full_sequence','Tm_true_C'] + \
             [f"pred_{safe(k)}" for k in model_keys]
    w.writerow(header)
    for i in range(64):
        row = [i, centres[i], sequences[i], f"{tm_true[i]:.2f}"]
        for k in model_keys:
            row.append(f"{cr[k]['predictions'][i]:.4f}")
        w.writerow(row)
print(f"  Saved: d2/classical_baseline_predictions.csv  ({len(model_keys)} models, 64 rows)")


# ═══════════════════════════════════════════════════════════════════════
# D3: Ablation predictions → CSV
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("D3: Ablation predictions → CSV")
print("=" * 60)

with open(os.path.join(ROOT, 'results/entanglement_ablation_results.json')) as f:
    d3 = json.load(f)

vr = d3['variant_results']
variants = list(vr.keys())

# Full-feature predictions
with open(os.path.join(ROOT, 'results/d3/ablation_full_features_predictions.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    header = ['index','centre_NNN','full_sequence','Tm_true_C'] + \
             [f"pred_{v}_111d" for v in variants]
    w.writerow(header)
    for i in range(64):
        row = [i, centres[i], sequences[i], f"{tm_true[i]:.2f}"]
        for v in variants:
            row.append(f"{vr[v]['full_features']['predictions'][i]:.4f}")
        w.writerow(row)
print(f"  Saved: d3/ablation_full_features_predictions.csv  ({len(variants)} variants, 64 rows)")

# Variable-region predictions
with open(os.path.join(ROOT, 'results/d3/ablation_variable_region_predictions.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    header = ['index','centre_NNN','full_sequence','Tm_true_C'] + \
             [f"pred_{v}_21d" for v in variants]
    w.writerow(header)
    for i in range(64):
        row = [i, centres[i], sequences[i], f"{tm_true[i]:.2f}"]
        for v in variants:
            row.append(f"{vr[v]['variable_region_features']['predictions'][i]:.4f}")
        w.writerow(row)
print(f"  Saved: d3/ablation_variable_region_predictions.csv  ({len(variants)} variants, 64 rows)")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL DONE. New files:")
for subdir in ['results/x1', 'results/d1', 'results/d2', 'results/d3']:
    full = os.path.join(ROOT, subdir)
    for fn in sorted(os.listdir(full)):
        path = os.path.join(full, fn)
        print(f"  {subdir}/{fn}  ({os.path.getsize(path)/1024:.1f} KB)")
