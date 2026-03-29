"""Experiment 1D Extension: 12-mer Scalability + Error Mitigation on ibm_fez.

Two tests:
  1. SCALABILITY: 20 diverse 12-mer sequences (24 qubits vs 8-mer's 16 qubits)
     - Shows the circuit architecture scales beyond 8-mers
     - 12-mers have real secondary structure (multiloops, bulges)
     - Feature vector: 24+23+22 = 69 dimensions

  2. ERROR MITIGATION: Same circuits run with and without mitigation
     - Baseline: SamplerV2 default (no mitigation)
     - Mitigated: Dynamical Decoupling (XY4) + Pauli Twirling (3 randomizations)
     - DD suppresses T1/T2 decoherence in idle periods
     - Twirling converts coherent errors -> stochastic noise (easier to mitigate)

Noiseless reference: AerSimulator(method='statevector') -- handles 24 qubits, ~9s/circuit
Target: cosine sim >0.95 for 12-mers baseline, >0.97 with error mitigation.
"""
import sys, os, json, time
sys.path.insert(0, 'D:/Qubis_HiQ/qubis-hiq-local/qubis-hiq')

import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from qiskit_aer import AerSimulator
from qiskit import transpile as qiskit_transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions

from qubis.circuit_builder import build_circuit
from qubis.feature_extraction import extract_feature_vector

# ── Constants ────────────────────────────────────────────────────────
N_SEQS    = 20      # 12-mer sequences (7 high-GC, 7 low-GC, 6 mixed)
N_SHOTS   = 4096   # shots for IBM hardware
N_SHOTS_SV= 8192   # shots for AerSimulator noiseless reference
BASE_12   = 69     # 24 + 23 + 22 feature dims for 12-mers
SEQ_LEN   = 12
TOKEN     = '6dTJShdjSxCHyDai0J_FV5cjuDEWFktTYXk4IdjH8wgH'
OUT_DIR   = 'D:/Qubis_HiQ/results_exp1d_12mer'

os.makedirs(OUT_DIR, exist_ok=True)

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def make_diverse_12mers(seed=99):
    """20 diverse 12-mers: 7 high-GC, 7 low-GC, 6 mixed.
    Each group uses a restricted alphabet to enforce GC composition.
    """
    rng = np.random.RandomState(seed)
    group_specs = [
        ('high_gc', 'GC',   0.75, 1.01, 7),
        ('low_gc',  'AT',   0.00, 0.26, 7),
        ('mixed',   'ATGC', 0.25, 0.76, 6),
    ]
    seqs, labels = [], []
    seen = set()
    for gname, alphabet, gc_lo, gc_hi, count in group_specs:
        found, attempts = 0, 0
        while found < count and attempts < 500000:
            attempts += 1
            seq = ''.join(rng.choice(list(alphabet), SEQ_LEN))
            gc = gc_content(seq)
            if gc_lo <= gc < gc_hi and seq not in seen:
                seen.add(seq)
                seqs.append(seq)
                labels.append(gname)
                found += 1
        if found < count:
            raise RuntimeError('Only found {}/{} seqs for {}'.format(found, count, gname))
    return seqs, labels

print('[{}] Exp 1D-12mer: Scalability + Error Mitigation'.format(datetime.now()))
print('  SEQ_LEN={} | N_SEQS={} | N_SHOTS={} | BASE={}'.format(
    SEQ_LEN, N_SEQS, N_SHOTS, BASE_12))
sys.stdout.flush()

# ── Step 1: Generate diverse 12-mer sequences ────────────────────────
seqs, seq_labels = make_diverse_12mers(seed=99)
assert len(seqs) == N_SEQS

gc_vals = [gc_content(s) for s in seqs]
print('  GC ranges: high={:.2f}-{:.2f} | low={:.2f}-{:.2f} | mixed={:.2f}-{:.2f}'.format(
    min(gc_vals[:7]), max(gc_vals[:7]),
    min(gc_vals[7:14]), max(gc_vals[7:14]),
    min(gc_vals[14:]), max(gc_vals[14:])))
sys.stdout.flush()

params = np.zeros(12)   # trainable layer: 12 shared params (works for any seq length)

# ── Step 2: AerSimulator noiseless reference ─────────────────────────
print('\n[Step 1/3] AerSimulator statevector reference ({} 24-qubit circuits)...'.format(N_SEQS))
sys.stdout.flush()
t0 = time.time()

aer_sim = AerSimulator(method='statevector')
sim_features = []
hw_circuits  = []   # circuits WITH measurements for IBM hardware

for i, seq in enumerate(seqs):
    # Noiseless statevector reference (AerSimulator handles 24 qubits)
    qc_aer = build_circuit(seq, stem_pairs=[], trainable_params=params,
                            include_measurement=True)
    qc_t = qiskit_transpile(qc_aer, aer_sim, optimization_level=0)
    job_aer = aer_sim.run(qc_t, shots=N_SHOTS_SV)
    counts_aer = job_aer.result().get_counts()
    fv = extract_feature_vector(counts_aer, len(seq), [])[:BASE_12]
    sim_features.append(fv)

    # Circuit for IBM hardware (not transpiled yet)
    qc_hw = build_circuit(seq, stem_pairs=[], trainable_params=params,
                           include_measurement=True)
    hw_circuits.append(qc_hw)

    if (i + 1) % 5 == 0:
        print('  {}/{} done ({:.0f}s)'.format(i+1, N_SEQS, time.time()-t0))
        sys.stdout.flush()

sim_features = np.array(sim_features)
np.save('{}/sim_features_12mer.npy'.format(OUT_DIR), sim_features)
print('  AerSim done in {:.1f}s. Features: {} x {}'.format(
    time.time()-t0, *sim_features.shape))
sys.stdout.flush()

# ── Step 3: Connect to IBM and transpile ────────────────────────────
print('\n[Step 2/3] Transpiling {} circuits for ibm_fez...'.format(N_SEQS))
sys.stdout.flush()
t1 = time.time()

service = QiskitRuntimeService(channel='ibm_quantum_platform', token=TOKEN)
backend = service.backend('ibm_fez')

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuits = [pm.run(qc) for qc in hw_circuits]
depths = [c.depth() for c in isa_circuits]
print('  Transpiled in {:.1f}s. Mean depth: {:.1f} | Max: {} | Min: {}'.format(
    time.time()-t1, np.mean(depths), max(depths), min(depths)))
sys.stdout.flush()

def extract_hw_features(job_result, n_seqs):
    """Extract feature vectors from hardware job results."""
    features = []
    for idx in range(n_seqs):
        counts = job_result[idx].data.meas.get_counts()
        fv = extract_feature_vector(counts, SEQ_LEN, [])[:BASE_12]
        features.append(fv)
    return np.array(features)

def compute_cosines(sim_feats, hw_feats):
    """Per-sequence cosine similarity."""
    return [float(sk_cosine([sim_feats[i]], [hw_feats[i]])[0][0])
            for i in range(len(sim_feats))]

# ── Step 3a: Baseline (no error mitigation) ──────────────────────────
print('\n[Step 3a/3] IBM Hardware: BASELINE (no error mitigation)...')
sys.stdout.flush()
t2 = time.time()

sampler_base = Sampler(mode=backend)
job_base = sampler_base.run([(qc,) for qc in isa_circuits], shots=N_SHOTS)
job_id_base = job_base.job_id()
print('  Job ID (baseline): {}'.format(job_id_base))
sys.stdout.flush()

while True:
    status = job_base.status()
    elapsed = time.time() - t2
    print('  Baseline status: {}  ({:.0f}s)'.format(status, elapsed))
    sys.stdout.flush()
    if str(status) in ('JobStatus.DONE', 'DONE', 'done'):
        break
    if str(status) in ('JobStatus.ERROR', 'ERROR', 'CANCELLED'):
        print('ERROR: Baseline job failed: {}'.format(status))
        sys.exit(1)
    time.sleep(15)

hw_features_base = extract_hw_features(job_base.result(), N_SEQS)
np.save('{}/hw_features_12mer_base.npy'.format(OUT_DIR), hw_features_base)
cos_base = compute_cosines(sim_features, hw_features_base)
mean_base = float(np.mean(cos_base))
std_base  = float(np.std(cos_base))
time_base = time.time() - t2
print('  Baseline cosine: {:.4f} +/- {:.4f}  ({:.1f}s)'.format(
    mean_base, std_base, time_base))
print('  Target >0.95: {}'.format('MET' if mean_base > 0.95 else 'NOT MET'))
sys.stdout.flush()

# ── Step 3b: With error mitigation (DD + Pauli Twirling) ────────────
print('\n[Step 3b/3] IBM Hardware: ERROR MITIGATION (DD-XY4 + Twirling)...')
sys.stdout.flush()
t3 = time.time()

options_em = SamplerOptions()
# Dynamical Decoupling: XY4 suppresses both X and Y decoherence channels
options_em.dynamical_decoupling.enable = True
options_em.dynamical_decoupling.sequence_type = 'XY4'
# Pauli Twirling: converts coherent errors to stochastic (depolarizing) noise
options_em.twirling.enable_gates = True
options_em.twirling.enable_measure = True
options_em.twirling.num_randomizations = 'auto'

sampler_em = Sampler(mode=backend, options=options_em)
job_em = sampler_em.run([(qc,) for qc in isa_circuits], shots=N_SHOTS)
job_id_em = job_em.job_id()
print('  Job ID (mitigated): {}'.format(job_id_em))
sys.stdout.flush()

while True:
    status = job_em.status()
    elapsed = time.time() - t3
    print('  Mitigated status: {}  ({:.0f}s)'.format(status, elapsed))
    sys.stdout.flush()
    if str(status) in ('JobStatus.DONE', 'DONE', 'done'):
        break
    if str(status) in ('JobStatus.ERROR', 'ERROR', 'CANCELLED'):
        print('ERROR: Mitigated job failed: {}'.format(status))
        sys.exit(1)
    time.sleep(15)

hw_features_em = extract_hw_features(job_em.result(), N_SEQS)
np.save('{}/hw_features_12mer_em.npy'.format(OUT_DIR), hw_features_em)
cos_em   = compute_cosines(sim_features, hw_features_em)
mean_em  = float(np.mean(cos_em))
std_em   = float(np.std(cos_em))
time_em  = time.time() - t3
print('  Mitigated cosine: {:.4f} +/- {:.4f}  ({:.1f}s)'.format(
    mean_em, std_em, time_em))
print('  Target >0.97: {}'.format('MET' if mean_em > 0.97 else 'NOT MET'))
sys.stdout.flush()

# ── Per-group breakdown ──────────────────────────────────────────────
def group_stats(cos_list, n7_7_6):
    n0, n1, n2 = n7_7_6
    h  = cos_list[:n0]
    l  = cos_list[n0:n0+n1]
    m  = cos_list[n0+n1:]
    return {
        'high_gc': {'mean': float(np.mean(h)), 'std': float(np.std(h)), 'n': len(h)},
        'low_gc':  {'mean': float(np.mean(l)), 'std': float(np.std(l)), 'n': len(l)},
        'mixed':   {'mean': float(np.mean(m)), 'std': float(np.std(m)), 'n': len(m)},
    }

groups_base = group_stats(cos_base, (7, 7, 6))
groups_em   = group_stats(cos_em,   (7, 7, 6))

# ── Save results ─────────────────────────────────────────────────────
results = {
    'experiment':      '1D_12mer',
    'seq_len':         SEQ_LEN,
    'n_qubits':        2 * SEQ_LEN,
    'n_seqs':          N_SEQS,
    'n_shots':         N_SHOTS,
    'n_shots_sv':      N_SHOTS_SV,
    'base_features':   BASE_12,
    'backend':         'ibm_fez',
    'circuit_version': 'v2_ry_cx_ry',
    'sequences':       seqs,
    'seq_labels':      seq_labels,
    'gc_contents':     gc_vals,
    'transpiled_depths': depths,
    'aer_reference': {
        'features_shape': list(sim_features.shape)
    },
    'baseline': {
        'job_id':      job_id_base,
        'exec_seconds': time_base,
        'mean_cosine': mean_base,
        'std_cosine':  std_base,
        'min_cosine':  float(np.min(cos_base)),
        'max_cosine':  float(np.max(cos_base)),
        'success_95':  bool(mean_base > 0.95),
        'by_group':    groups_base,
        'cosines':     cos_base
    },
    'error_mitigated': {
        'job_id':        job_id_em,
        'exec_seconds':  time_em,
        'mean_cosine':   mean_em,
        'std_cosine':    std_em,
        'min_cosine':    float(np.min(cos_em)),
        'max_cosine':    float(np.max(cos_em)),
        'success_97':    bool(mean_em > 0.97),
        'dd_sequence':   'XY4',
        'twirling':      'gate+measure',
        'by_group':      groups_em,
        'cosines':       cos_em
    },
    'improvement': {
        'delta_cosine': float(mean_em - mean_base),
        'relative_pct': float(100.0 * (mean_em - mean_base) / (1.0 - mean_base))
                        if mean_base < 1.0 else 0.0
    }
}

with open('{}/exp1d_12mer_results.json'.format(OUT_DIR), 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*65)
print('EXPERIMENT 1D-12mer COMPLETE')
print('  Noiseless ref (AerSim, 24-qubit statevector):')
print('    Features: {} x {}'.format(*sim_features.shape))
print('  Baseline (no mitigation):')
print('    Cosine: {:.4f} +/- {:.4f}  target >0.95: {}'.format(
    mean_base, std_base, 'MET' if mean_base > 0.95 else 'NOT MET'))
for g, s in groups_base.items():
    print('    {:<10}: {:.4f} +/- {:.4f} (n={})'.format(
        g, s['mean'], s['std'], s['n']))
print('  Error mitigated (DD-XY4 + Pauli Twirling):')
print('    Cosine: {:.4f} +/- {:.4f}  target >0.97: {}'.format(
    mean_em, std_em, 'MET' if mean_em > 0.97 else 'NOT MET'))
print('  Improvement: delta={:.4f} ({:.1f}% error reduction)'.format(
    results['improvement']['delta_cosine'],
    results['improvement']['relative_pct']))
print('='*65)
sys.stdout.flush()
