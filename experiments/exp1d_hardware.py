"""Experiment 1D: IBM Hardware Validation (Steps 13-16).
Uses ibm_fez (Heron r2, 156 qubits) and FakeFez for simulator baseline.
stem_pairs=[] for all sequences.
Circuits use encoding + stacking layers (no WC) -- valid hardware fidelity test.

Scaled parameters:
  N_SEQS_HW   = 30   (IBM hardware -- 30 thermodynamically diverse seqs, ~3-4 min QPU)
    - 10 high-GC (75-100%): stable hairpin-prone sequences
    - 10 low-GC  (0-25%) : AT-rich, thermodynamically unstable
    - 10 mixed   (25-75%): varied composition
  N_SHOTS_HW  = 4096 (IBM hardware -- low shot noise, 4x more than v1)
  N_SEQS_FAKE = 10   (FakeFez -- one representative from each group)
  N_SHOTS_FAKE= 1024 (FakeFez -- practical for CPU-based Monte Carlo)
  N_SHOTS_SV  = 8192 (Statevector sampling -- near-noiseless reference)
"""
import sys, os, json, time
sys.path.insert(0, 'D:/Qubis_HiQ/qubis-hiq-local/qubis-hiq')

import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez

from qubis.circuit_builder import build_circuit
from qubis.feature_extraction import extract_feature_vector

# ── Constants ────────────────────────────────────────────────────────
N_SEQS_HW    = 30     # sequences submitted to ibm_fez (robust statistics)
N_SEQS_FAKE  = 10     # FakeFez subset (one from each group); limits local sim time
N_SHOTS_HW   = 4096  # shots on ibm_fez (~2x more accurate than v1)
N_SHOTS_FAKE = 1024  # shots for FakeFez (CPU Monte Carlo -- practical ceiling)
N_SHOTS_SV   = 8192  # statevector samples (near-noiseless reference)
BASE         = 45    # base feature length (no stem correlators)
OUT_DIR      = 'D:/Qubis_HiQ/results_exp1d'
TOKEN        = '6dTJShdjSxCHyDai0J_FV5cjuDEWFktTYXk4IdjH8wgH'

os.makedirs(OUT_DIR, exist_ok=True)

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def make_diverse_seqs(n_per_group=10, seed=42):
    """Generate thermodynamically diverse 8-mers.
    Group 1 (high-GC, 75-100%): GC-rich, stable, hairpin-prone.
    Group 2 (low-GC,  0-25%) : AT-rich, thermodynamically unstable.
    Group 3 (mixed, 25-75%)  : moderate, varied composition.
    Uniqueness enforced; each group sampled from appropriate alphabet.
    """
    rng = np.random.RandomState(seed)
    groups = {
        'high_gc': ('GC', 0.75, 1.01),
        'low_gc':  ('AT', 0.00, 0.26),
        'mixed':   ('ATGC', 0.25, 0.76),
    }
    all_seqs  = []
    labels    = []
    seen      = set()
    for group_name, (alphabet, gc_lo, gc_hi) in groups.items():
        count = 0
        attempts = 0
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
            raise RuntimeError('Could not generate {} seqs for group {}'.format(
                n_per_group, group_name))
    return all_seqs, labels

print('[{}] Exp 1D v2: IBM Hardware Validation (scaled, diverse seqs)'.format(datetime.now()))
print('  N_SEQS_HW={} | N_SHOTS_HW={} | N_SEQS_FAKE={} | N_SHOTS_FAKE={}'.format(
    N_SEQS_HW, N_SHOTS_HW, N_SEQS_FAKE, N_SHOTS_FAKE))
sys.stdout.flush()

# ── Step 1: Generate thermodynamically diverse sequences ─────────────
seqs, seq_labels = make_diverse_seqs(n_per_group=10, seed=42)
assert len(seqs) == N_SEQS_HW

gc_vals = [gc_content(s) for s in seqs]
print('  Seq GC content: high={:.2f}-{:.2f} | low={:.2f}-{:.2f} | mixed={:.2f}-{:.2f}'.format(
    min(gc_vals[:10]), max(gc_vals[:10]),
    min(gc_vals[10:20]), max(gc_vals[10:20]),
    min(gc_vals[20:]), max(gc_vals[20:])))
sys.stdout.flush()

params = np.zeros(12)

# ── Step 2: Build circuits + Statevector simulation ──────────────────
print('\n[Step 1/4] Building {} circuits & statevector simulation...'.format(N_SEQS_HW))
sys.stdout.flush()
t0 = time.time()

sim_features = []
hw_circuits  = []   # circuits WITH measurements (all 30)

for i, seq in enumerate(seqs):
    # Noiseless reference via statevector sampling
    qc_sv = build_circuit(seq, stem_pairs=[], trainable_params=params,
                          include_measurement=False)
    sv = Statevector.from_instruction(qc_sv)
    n_q = 2 * len(seq)   # 16
    counts_sv = {
        (format(k, '0{}b'.format(n_q)) if isinstance(k, int) else str(k)): v
        for k, v in sv.sample_counts(N_SHOTS_SV).items()
    }
    fv = extract_feature_vector(counts_sv, len(seq), [])[:BASE]
    sim_features.append(fv)

    # Circuit with measurements for hardware / FakeFez
    qc_hw = build_circuit(seq, stem_pairs=[], trainable_params=params,
                          include_measurement=True)
    hw_circuits.append(qc_hw)

sim_features = np.array(sim_features)
np.save('{}/sim_features.npy'.format(OUT_DIR), sim_features)
print('  Statevector done in {:.1f}s. Features: {} x {}'.format(
    time.time()-t0, *sim_features.shape))
sys.stdout.flush()

# ── Step 14: FakeFez (noisy simulator baseline, first N_SEQS_FAKE) ───
print('\n[Step 2/4 - Step 14] FakeFez noise simulation (first {} of {} seqs)...'.format(
    N_SEQS_FAKE, N_SEQS_HW))
sys.stdout.flush()
t1 = time.time()

# FakeFez subset: pick indices spread across all 3 groups (3-4 per group)
# Groups: [0:10]=high_gc, [10:20]=low_gc, [20:30]=mixed
fake_indices = [0, 2, 5, 8, 10, 13, 16, 19, 21, 25]  # ~3-4 per group
assert len(fake_indices) == N_SEQS_FAKE

fake_backend = FakeFez()
pm_fake      = generate_preset_pass_manager(backend=fake_backend, optimization_level=1)
isa_fake     = [pm_fake.run(hw_circuits[i]) for i in fake_indices]

print('  Transpiled in {:.1f}s. Mean circuit depth: {:.1f}'.format(
    time.time()-t1, np.mean([c.depth() for c in isa_fake])))
print('  FakeFez indices (across groups): {}'.format(fake_indices))
sys.stdout.flush()

sampler_fake = Sampler(mode=fake_backend)
pubs_fake    = [(qc,) for qc in isa_fake]
job_fake     = sampler_fake.run(pubs_fake, shots=N_SHOTS_FAKE)
results_fake = job_fake.result()

fake_features = []
for rank, idx in enumerate(fake_indices):
    counts = results_fake[rank].data.meas.get_counts()
    fv = extract_feature_vector(counts, 8, [])[:BASE]
    fake_features.append(fv)

fake_features = np.array(fake_features)
np.save('{}/fake_features.npy'.format(OUT_DIR), fake_features)

cos_fake      = [float(sk_cosine([sim_features[fake_indices[i]]], [fake_features[i]])[0][0])
                 for i in range(N_SEQS_FAKE)]
mean_cos_fake = float(np.mean(cos_fake))
std_cos_fake  = float(np.std(cos_fake))
print('  FakeFez cosine (n={}): {:.4f} +/- {:.4f}  ({:.1f}s)'.format(
    N_SEQS_FAKE, mean_cos_fake, std_cos_fake, time.time()-t1))
print('  Step 14 target >0.85: {}'.format('MET' if mean_cos_fake > 0.85 else 'NOT MET'))
sys.stdout.flush()

# ── Steps 15-16: Real ibm_fez hardware (all N_SEQS_HW circuits) ─────
print('\n[Step 3/4 - Steps 15-16] Submitting {} circuits to ibm_fez @ {} shots...'.format(
    N_SEQS_HW, N_SHOTS_HW))
sys.stdout.flush()
t2 = time.time()

service  = QiskitRuntimeService(channel='ibm_quantum_platform', token=TOKEN)
backend  = service.backend('ibm_fez')

pm_hw    = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_hw   = [pm_hw.run(qc) for qc in hw_circuits]
depths   = [c.depth() for c in isa_hw]
print('  Transpiled {} circuits in {:.1f}s. Mean depth: {:.1f} | Max: {}'.format(
    len(isa_hw), time.time()-t2, np.mean(depths), max(depths)))
sys.stdout.flush()

sampler_hw = Sampler(mode=backend)
pubs_hw    = [(qc,) for qc in isa_hw]
job_hw     = sampler_hw.run(pubs_hw, shots=N_SHOTS_HW)
job_id     = job_hw.job_id()

print('  Job ID: {}'.format(job_id))
print('  Submitted at: {}'.format(datetime.now()))
print('  Waiting for results...')
sys.stdout.flush()

while True:
    status = job_hw.status()
    elapsed = time.time() - t2
    print('  Status: {}  ({:.0f}s elapsed)'.format(status, elapsed))
    sys.stdout.flush()
    if str(status) in ('JobStatus.DONE', 'DONE', 'done'):
        break
    if str(status) in ('JobStatus.ERROR', 'ERROR', 'CANCELLED'):
        print('ERROR: Job failed with status: {}'.format(status))
        sys.exit(1)
    time.sleep(15)

hw_exec_time = time.time() - t2
print('  Hardware job completed in {:.1f}s'.format(hw_exec_time))
sys.stdout.flush()

# ── Step 4: Extract hardware features + cosine similarity ────────────
print('\n[Step 4/4] Extracting hardware features ({} seqs)...'.format(N_SEQS_HW))
sys.stdout.flush()
results_hw = job_hw.result()

hw_features = []
for idx in range(N_SEQS_HW):
    counts = results_hw[idx].data.meas.get_counts()
    fv = extract_feature_vector(counts, 8, [])[:BASE]
    hw_features.append(fv)

hw_features = np.array(hw_features)
np.save('{}/hw_features.npy'.format(OUT_DIR), hw_features)

cos_hw      = [float(sk_cosine([sim_features[i]], [hw_features[i]])[0][0])
               for i in range(N_SEQS_HW)]
mean_cos_hw = float(np.mean(cos_hw))
std_cos_hw  = float(np.std(cos_hw))

# Also compute hw vs FakeFez cosine (on the overlapping N_SEQS_FAKE subset)
cos_hw_vs_fake = [float(sk_cosine([hw_features[fake_indices[i]]], [fake_features[i]])[0][0])
                  for i in range(N_SEQS_FAKE)]
mean_cos_hw_vs_fake = float(np.mean(cos_hw_vs_fake))

print('  HW vs statevector cosine (n={}): {:.4f} +/- {:.4f}'.format(
    N_SEQS_HW, mean_cos_hw, std_cos_hw))
print('  HW vs FakeFez cosine  (n={}):  {:.4f} +/- {:.4f}'.format(
    N_SEQS_FAKE, mean_cos_hw_vs_fake, float(np.std(cos_hw_vs_fake))))
print('  Step 16 target >0.85: {}'.format('MET' if mean_cos_hw > 0.85 else 'NOT MET'))
sys.stdout.flush()

# ── Save full results ────────────────────────────────────────────────
# Per-group breakdown for hardware cosines
cos_hw_high_gc = cos_hw[0:10]
cos_hw_low_gc  = cos_hw[10:20]
cos_hw_mixed   = cos_hw[20:30]

results = {
    'experiment':          '1D_v2',
    'n_seqs_hw':           N_SEQS_HW,
    'n_seqs_fake':         N_SEQS_FAKE,
    'n_shots_hw':          N_SHOTS_HW,
    'n_shots_fake':        N_SHOTS_FAKE,
    'n_shots_sv':          N_SHOTS_SV,
    'backend':             'ibm_fez',
    'circuit_version':     'v2_ry_cx_ry',
    'stem_pairs':          'none (no ViennaRNA)',
    'job_id':              job_id,
    'hw_exec_seconds':     hw_exec_time,
    'sequences':           seqs,
    'seq_labels':          seq_labels,
    'gc_contents':         gc_vals,
    'statevector': {
        'features_shape':  list(sim_features.shape)
    },
    'fake_fez': {
        'n_seqs':          N_SEQS_FAKE,
        'n_shots':         N_SHOTS_FAKE,
        'fake_indices':    fake_indices,
        'mean_cosine':     mean_cos_fake,
        'std_cosine':      std_cos_fake,
        'success':         bool(mean_cos_fake > 0.85)
    },
    'ibm_fez_hardware': {
        'n_seqs':          N_SEQS_HW,
        'n_shots':         N_SHOTS_HW,
        'mean_cosine':     mean_cos_hw,
        'std_cosine':      std_cos_hw,
        'min_cosine':      float(np.min(cos_hw)),
        'max_cosine':      float(np.max(cos_hw)),
        'mean_cosine_vs_fake': mean_cos_hw_vs_fake,
        'success':         bool(mean_cos_hw > 0.85),
        'by_group': {
            'high_gc': {'mean': float(np.mean(cos_hw_high_gc)),
                        'std':  float(np.std(cos_hw_high_gc))},
            'low_gc':  {'mean': float(np.mean(cos_hw_low_gc)),
                        'std':  float(np.std(cos_hw_low_gc))},
            'mixed':   {'mean': float(np.mean(cos_hw_mixed)),
                        'std':  float(np.std(cos_hw_mixed))}
        }
    },
    'cosines_per_seq_hw':   cos_hw,
    'cosines_per_seq_fake': cos_fake,
    'cosines_hw_vs_fake':   cos_hw_vs_fake
}

with open('{}/exp1d_results.json'.format(OUT_DIR), 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*60)
print('EXPERIMENT 1D v2 COMPLETE')
print('  FakeFez  cosine (n={}):    {:.4f} +/- {:.4f}  ({})'.format(
    N_SEQS_FAKE, mean_cos_fake, std_cos_fake,
    'SUCCESS' if mean_cos_fake > 0.85 else 'NOT MET'))
print('  ibm_fez  cosine (n={}):   {:.4f} +/- {:.4f}  ({})'.format(
    N_SEQS_HW, mean_cos_hw, std_cos_hw,
    'SUCCESS' if mean_cos_hw > 0.85 else 'NOT MET'))
print('    high-GC group (n=10):   {:.4f} +/- {:.4f}'.format(
    float(np.mean(cos_hw_high_gc)), float(np.std(cos_hw_high_gc))))
print('    low-GC  group (n=10):   {:.4f} +/- {:.4f}'.format(
    float(np.mean(cos_hw_low_gc)), float(np.std(cos_hw_low_gc))))
print('    mixed   group (n=10):   {:.4f} +/- {:.4f}'.format(
    float(np.mean(cos_hw_mixed)), float(np.std(cos_hw_mixed))))
print('  HW vs FakeFez  (n={}):    {:.4f} +/- {:.4f}'.format(
    N_SEQS_FAKE, mean_cos_hw_vs_fake, float(np.std(cos_hw_vs_fake))))
print('  IBM hardware exec time:   {:.1f}s'.format(hw_exec_time))
print('='*60)
print('Results saved to {}/'.format(OUT_DIR))
sys.stdout.flush()
