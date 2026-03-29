"""Experiment 1D-torino: Cross-Platform Portability on ibm_torino (Heron r1).

Proves the circuit output is NOT overfit to ibm_fez's (Heron r2) specific noise profile.
Runs the same 30 thermodynamically diverse 8-mer sequences on ibm_torino (Heron r1, 133q).

Comparison:
  ibm_fez   (Heron r2, 156q): cosine = 0.9970 +/- 0.0005  (from exp1d v2)
  ibm_torino (Heron r1, 133q): < this run >

Different topologies:
  ibm_fez   : Heron r2 -- new-gen, lower error rates, different coupling map
  ibm_torino : Heron r1 -- prev-gen, independent calibration/noise profile

Reuses:
  - sim_features.npy (statevector reference, backend-independent)
  - Same 30 sequences (seed=42, make_diverse_seqs) and params=zeros(12)
"""
import sys, os, json, time
sys.path.insert(0, 'D:/Qubis_HiQ/qubis-hiq-local/qubis-hiq')

import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from qubis.circuit_builder import build_circuit
from qubis.feature_extraction import extract_feature_vector

# ── Constants ────────────────────────────────────────────────────────
N_SEQS    = 30
N_SHOTS   = 4096
BASE      = 45
TOKEN     = '6dTJShdjSxCHyDai0J_FV5cjuDEWFktTYXk4IdjH8wgH'
OUT_DIR   = 'D:/Qubis_HiQ/results_exp1d_torino'
REF_DIR   = 'D:/Qubis_HiQ/results_exp1d'   # existing sim_features from exp1d v2

os.makedirs(OUT_DIR, exist_ok=True)

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def make_diverse_seqs(n_per_group=10, seed=42):
    """Same generation as exp1d v2 -- identical 30 sequences."""
    rng = np.random.RandomState(seed)
    groups = {
        'high_gc': ('GC',   0.75, 1.01),
        'low_gc':  ('AT',   0.00, 0.26),
        'mixed':   ('ATGC', 0.25, 0.76),
    }
    all_seqs, labels = [], []
    seen = set()
    for group_name, (alphabet, gc_lo, gc_hi) in groups.items():
        count, attempts = 0, 0
        while count < n_per_group and attempts < 100000:
            attempts += 1
            seq = ''.join(rng.choice(list(alphabet), 8))
            gc = gc_content(seq)
            if gc_lo <= gc < gc_hi and seq not in seen:
                seen.add(seq)
                all_seqs.append(seq)
                labels.append(group_name)
                count += 1
        if count < n_per_group:
            raise RuntimeError('Could not generate {} seqs for {}'.format(
                n_per_group, group_name))
    return all_seqs, labels

print('[{}] Exp 1D-torino: Cross-Platform Portability'.format(datetime.now()))
print('  Backend: ibm_torino (Heron r1, 133q) | N_SEQS={} | N_SHOTS={}'.format(
    N_SEQS, N_SHOTS))
sys.stdout.flush()

# ── Step 1: Load statevector reference + regenerate circuits ─────────
print('\n[Step 1/2] Loading statevector reference and rebuilding circuits...')
sys.stdout.flush()

sim_features = np.load('{}/sim_features.npy'.format(REF_DIR))
assert sim_features.shape == (N_SEQS, BASE), \
    'Expected ({}, {}), got {}'.format(N_SEQS, BASE, sim_features.shape)
print('  Loaded sim_features: {} x {}'.format(*sim_features.shape))

seqs, seq_labels = make_diverse_seqs(n_per_group=10, seed=42)
params = np.zeros(12)
gc_vals = [gc_content(s) for s in seqs]

hw_circuits = [
    build_circuit(seq, stem_pairs=[], trainable_params=params, include_measurement=True)
    for seq in seqs
]
print('  Rebuilt {} circuits.'.format(len(hw_circuits)))
sys.stdout.flush()

# ── Step 2: Transpile + Submit to ibm_torino ─────────────────────────
print('\n[Step 2/2] Transpiling and submitting to ibm_torino...')
sys.stdout.flush()
t0 = time.time()

service  = QiskitRuntimeService(channel='ibm_quantum_platform', token=TOKEN)
backend  = service.backend('ibm_torino')
print('  Backend: {} | Version: {} | Qubits: {}'.format(
    backend.name, backend.backend_version, backend.num_qubits))
sys.stdout.flush()

pm       = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circs = [pm.run(qc) for qc in hw_circuits]
depths   = [c.depth() for c in isa_circs]
print('  Transpiled in {:.1f}s. Mean depth: {:.1f} | Max: {} | Min: {}'.format(
    time.time()-t0, np.mean(depths), max(depths), min(depths)))
print('  (ibm_fez reference depth was 36 -- compare for topology effect)')
sys.stdout.flush()

sampler  = Sampler(mode=backend)
job      = sampler.run([(qc,) for qc in isa_circs], shots=N_SHOTS)
job_id   = job.job_id()
print('  Job ID: {}'.format(job_id))
print('  Submitted at: {}'.format(datetime.now()))
sys.stdout.flush()

while True:
    status  = job.status()
    elapsed = time.time() - t0
    print('  Status: {}  ({:.0f}s elapsed)'.format(status, elapsed))
    sys.stdout.flush()
    if str(status) in ('JobStatus.DONE', 'DONE', 'done'):
        break
    if str(status) in ('JobStatus.ERROR', 'ERROR', 'CANCELLED'):
        print('ERROR: Job failed: {}'.format(status))
        sys.exit(1)
    time.sleep(15)

exec_time = time.time() - t0
print('  Job completed in {:.1f}s'.format(exec_time))
sys.stdout.flush()

# ── Step 3: Extract features + cosine similarity ─────────────────────
results_hw = job.result()
hw_features = []
for idx in range(N_SEQS):
    counts = results_hw[idx].data.meas.get_counts()
    fv = extract_feature_vector(counts, 8, [])[:BASE]
    hw_features.append(fv)

hw_features = np.array(hw_features)
np.save('{}/hw_features_torino.npy'.format(OUT_DIR), hw_features)
np.save('{}/sim_features.npy'.format(OUT_DIR), sim_features)   # copy for reference

cos_torino      = [float(sk_cosine([sim_features[i]], [hw_features[i]])[0][0])
                   for i in range(N_SEQS)]
mean_cos_torino = float(np.mean(cos_torino))
std_cos_torino  = float(np.std(cos_torino))

# Per-group breakdown
cos_high = cos_torino[0:10]
cos_low  = cos_torino[10:20]
cos_mix  = cos_torino[20:30]

print('\n  ibm_torino cosine (n={}): {:.4f} +/- {:.4f}'.format(
    N_SEQS, mean_cos_torino, std_cos_torino))
print('    high-GC (n=10): {:.4f} +/- {:.4f}'.format(
    float(np.mean(cos_high)), float(np.std(cos_high))))
print('    low-GC  (n=10): {:.4f} +/- {:.4f}'.format(
    float(np.mean(cos_low)), float(np.std(cos_low))))
print('    mixed   (n=10): {:.4f} +/- {:.4f}'.format(
    float(np.mean(cos_mix)), float(np.std(cos_mix))))
print('  Target >0.85: {}'.format('MET' if mean_cos_torino > 0.85 else 'NOT MET'))
sys.stdout.flush()

# Cross-platform comparison
ibm_fez_mean = 0.9969584744558994   # from exp1d v2 results.json
ibm_fez_std  = 0.0005054656461104235
delta        = float(mean_cos_torino - ibm_fez_mean)

print('\n  Cross-platform comparison:')
print('    ibm_fez   (Heron r2): {:.4f} +/- {:.4f}'.format(ibm_fez_mean, ibm_fez_std))
print('    ibm_torino (Heron r1): {:.4f} +/- {:.4f}'.format(
    mean_cos_torino, std_cos_torino))
print('    Delta (torino - fez): {:+.4f}'.format(delta))
sys.stdout.flush()

# ── Save results ──────────────────────────────────────────────────────
output = {
    'experiment':        '1D_torino',
    'backend':           'ibm_torino',
    'backend_type':      'Heron r1',
    'n_qubits_backend':  133,
    'n_seqs':            N_SEQS,
    'n_shots':           N_SHOTS,
    'circuit_version':   'v2_ry_cx_ry',
    'job_id':            job_id,
    'exec_seconds':      exec_time,
    'transpiled_depths': depths,
    'sequences':         seqs,
    'seq_labels':        seq_labels,
    'gc_contents':       gc_vals,
    'ibm_torino': {
        'mean_cosine':   mean_cos_torino,
        'std_cosine':    std_cos_torino,
        'min_cosine':    float(np.min(cos_torino)),
        'max_cosine':    float(np.max(cos_torino)),
        'success_85':    bool(mean_cos_torino > 0.85),
        'by_group': {
            'high_gc': {'mean': float(np.mean(cos_high)),
                        'std':  float(np.std(cos_high))},
            'low_gc':  {'mean': float(np.mean(cos_low)),
                        'std':  float(np.std(cos_low))},
            'mixed':   {'mean': float(np.mean(cos_mix)),
                        'std':  float(np.std(cos_mix))},
        },
        'cosines': cos_torino
    },
    'cross_platform': {
        'ibm_fez_mean':    ibm_fez_mean,
        'ibm_fez_std':     ibm_fez_std,
        'ibm_torino_mean': mean_cos_torino,
        'ibm_torino_std':  std_cos_torino,
        'delta':           delta,
        'portable':        bool(abs(delta) < 0.01)
    }
}

with open('{}/exp1d_torino_results.json'.format(OUT_DIR), 'w') as f:
    json.dump(output, f, indent=2)

print('\n' + '='*60)
print('EXPERIMENT 1D-TORINO COMPLETE')
print('  ibm_torino (Heron r1): {:.4f} +/- {:.4f}  target >0.85: {}'.format(
    mean_cos_torino, std_cos_torino,
    'MET' if mean_cos_torino > 0.85 else 'NOT MET'))
print('  ibm_fez    (Heron r2): {:.4f} +/- {:.4f}  (reference)'.format(
    ibm_fez_mean, ibm_fez_std))
print('  Cross-platform delta: {:+.4f}  portable: {}'.format(
    delta, 'YES (|delta|<0.01)' if abs(delta) < 0.01 else 'MARGINAL'))
print('='*60)
sys.stdout.flush()
