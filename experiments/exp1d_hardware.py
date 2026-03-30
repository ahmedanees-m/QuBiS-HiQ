"""Experiment 1D: IBM Hardware Validation on ibm_fez (Heron r2, 156 qubits).

30 thermodynamically diverse 8-mer sequences submitted at 4096 shots each.
Compares hardware measurement outcomes against noiseless statevector reference
using per-sequence cosine similarity.

Sequence groups (10 each):
  high-GC  (75-100%): stable, hairpin-prone
    low-GC   (0-25%) : AT-rich, thermodynamically unstable
      mixed    (25-75%) : varied composition

      Usage:
        export IBM_QUANTUM_TOKEN=<your_token>
          python experiments/exp1d_hardware.py
          """
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez

from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.feature_extraction import extract_feature_vector

# ── Constants ────────────────────────────────────────────────────────
N_SEQS_HW   = 30    # sequences submitted to ibm_fez
N_SEQS_FAKE = 10    # FakeFez subset (one from each group)
N_SHOTS_HW  = 4096  # shots on ibm_fez
N_SHOTS_FAKE = 1024  # shots for FakeFez (CPU Monte Carlo)
N_SHOTS_SV  = 8192  # statevector samples (near-noiseless reference)
BASE        = 45    # base feature length (no stem correlators)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1d')
os.makedirs(OUT_DIR, exist_ok=True)

# Read token from environment — never hardcode credentials in source files
TOKEN = os.environ.get('IBM_QUANTUM_TOKEN')
if not TOKEN:
      raise EnvironmentError(
                "IBM_QUANTUM_TOKEN environment variable not set.\n"
                "Export it before running: export IBM_QUANTUM_TOKEN=<your_token>"
      )


def gc_content(seq):
      return (seq.count('G') + seq.count('C')) / len(seq)


def make_diverse_seqs(n_per_group=10, seed=42):
      """Generate 30 thermodynamically diverse 8-mers (10 per GC group)."""
      rng = np.random.RandomState(seed)
      groups = {
          'high_gc': ('GC',   0.75, 1.01),
          'low_gc':  ('AT',   0.00, 0.26),
          'mixed':   ('ATGC', 0.25, 0.76),
      }
      all_seqs, labels = [], []
      seen = set()
      for group_name, (alphabet, gc_lo, gc_hi) in groups.items():
                count = 0
                for _ in range(100000):
                              seq = ''.join(rng.choice(list(alphabet), 8))
                              gc = gc_content(seq)
                              if gc_lo <= gc < gc_hi and seq not in seen:
                                                seen.add(seq)
                                                all_seqs.append(seq)
                                                labels.append(group_name)
                                                count += 1
                                                if count == n_per_group:
                                                                      break
                                                          if count < n_per_group:
                                                                        raise RuntimeError(f'Only generated {count}/{n_per_group} seqs for {group_name}')
                                                                return all_seqs, labels


        print(f'[{datetime.now()}] Exp 1D: IBM Hardware Validation')
print(f'  N_SEQS={N_SEQS_HW} | N_SHOTS={N_SHOTS_HW} | N_SEQS_FAKE={N_SEQS_FAKE}')

seqs, seq_labels = make_diverse_seqs(n_per_group=10, seed=42)
assert len(seqs) == N_SEQS_HW
gc_vals = [gc_content(s) for s in seqs]
params = np.zeros(12)

# ── Step 1: Statevector reference ────────────────────────────────────
print(f'\n[1/4] Statevector reference ({N_SEQS_HW} circuits)...')
t0 = time.time()
sim_features, hw_circuits = [], []
for seq in seqs:
      qc_sv = build_circuit(seq, stem_pairs=[], trainable_params=params, include_measurement=False)
    sv = Statevector.from_instruction(qc_sv)
    n_q = 2 * len(seq)
    counts = {(format(k, f'0{n_q}b') if isinstance(k, int) else str(k)): v
                            for k, v in sv.sample_counts(N_SHOTS_SV).items()}
    sim_features.append(extract_feature_vector(counts, len(seq), [])[:BASE])
    hw_circuits.append(build_circuit(seq, stem_pairs=[], trainable_params=params,
                                                                           include_measurement=True))

sim_features = np.array(sim_features)
np.save(os.path.join(OUT_DIR, 'sim_features.npy'), sim_features)
print(f'  Done in {time.time()-t0:.1f}s. Features: {sim_features.shape}')

# ── Step 2: FakeFez noise simulation ─────────────────────────────────
print(f'\n[2/4] FakeFez noise simulation (first {N_SEQS_FAKE} seqs)...')
t1 = time.time()
fake_indices = [0, 2, 5, 8, 10, 13, 16, 19, 21, 25]
assert len(fake_indices) == N_SEQS_FAKE
fake_backend = FakeFez()
pm_fake = generate_preset_pass_manager(backend=fake_backend, optimization_level=1)
isa_fake = [pm_fake.run(hw_circuits[i]) for i in fake_indices]
print(f'  Transpiled in {time.time()-t1:.1f}s. Mean depth: {np.mean([c.depth() for c in isa_fake]):.1f}')

sampler_fake = Sampler(mode=fake_backend)
results_fake = sampler_fake.run([(qc,) for qc in isa_fake], shots=N_SHOTS_FAKE).result()
fake_features = np.array([
      extract_feature_vector(results_fake[r].data.meas.get_counts(), 8, [])[:BASE]
      for r in range(N_SEQS_FAKE)
])
np.save(os.path.join(OUT_DIR, 'fake_features.npy'), fake_features)
cos_fake = [float(sk_cosine([sim_features[fake_indices[i]]], [fake_features[i]])[0][0])
                        for i in range(N_SEQS_FAKE)]
mean_cos_fake = float(np.mean(cos_fake))
std_cos_fake  = float(np.std(cos_fake))
print(f'  FakeFez cosine (n={N_SEQS_FAKE}): {mean_cos_fake:.4f} +/- {std_cos_fake:.4f}')
print(f'  Target >0.85: {"MET" if mean_cos_fake > 0.85 else "NOT MET"}')

# ── Step 3: Real ibm_fez hardware ────────────────────────────────────
print(f'\n[3/4] Submitting {N_SEQS_HW} circuits to ibm_fez @ {N_SHOTS_HW} shots...')
t2 = time.time()
service = QiskitRuntimeService(channel='ibm_quantum_platform', token=TOKEN)
backend = service.backend('ibm_fez')
pm_hw  = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_hw = [pm_hw.run(qc) for qc in hw_circuits]
print(f'  Transpiled. Mean depth: {np.mean([c.depth() for c in isa_hw]):.1f}')

sampler_hw = Sampler(mode=backend)
job_hw = sampler_hw.run([(qc,) for qc in isa_hw], shots=N_SHOTS_HW)
job_id = job_hw.job_id()
print(f'  Job ID: {job_id}')
while True:
      status = str(job_hw.status())
    print(f'  Status: {status} ({time.time()-t2:.0f}s)', flush=True)
    if status in ('JobStatus.DONE', 'DONE', 'done'):
              break
          if status in ('JobStatus.ERROR', 'ERROR', 'CANCELLED'):
                    print('ERROR: job failed'); sys.exit(1)
                time.sleep(15)

# ── Step 4: Extract features + cosine similarity ─────────────────────
print(f'\n[4/4] Extracting hardware features...')
results_hw  = job_hw.result()
hw_features = np.array([
      extract_feature_vector(results_hw[i].data.meas.get_counts(), 8, [])[:BASE]
      for i in range(N_SEQS_HW)
])
np.save(os.path.join(OUT_DIR, 'hw_features.npy'), hw_features)

cos_hw     = [float(sk_cosine([sim_features[i]], [hw_features[i]])[0][0]) for i in range(N_SEQS_HW)]
mean_cos_hw = float(np.mean(cos_hw))
std_cos_hw  = float(np.std(cos_hw))
cos_hw_vs_fake = [float(sk_cosine([hw_features[fake_indices[i]]], [fake_features[i]])[0][0])
                                    for i in range(N_SEQS_FAKE)]

print(f'  HW vs statevector cosine (n={N_SEQS_HW}): {mean_cos_hw:.4f} +/- {std_cos_hw:.4f}')
print(f'  Target >0.85: {"MET" if mean_cos_hw > 0.85 else "NOT MET"}')

results = {
      'experiment': '1D',
      'backend': 'ibm_fez',
      'n_seqs': N_SEQS_HW, 'n_shots': N_SHOTS_HW,
      'circuit_version': 'v2_ry_cx_ry',
      'job_id': job_id,
      'sequences': seqs, 'seq_labels': seq_labels, 'gc_contents': gc_vals,
      'fake_fez': {
                'mean_cosine': mean_cos_fake, 'std_cosine': std_cos_fake,
                'success': bool(mean_cos_fake > 0.85)
      },
      'ibm_fez_hardware': {
                'mean_cosine': mean_cos_hw, 'std_cosine': std_cos_hw,
                'min_cosine': float(np.min(cos_hw)), 'max_cosine': float(np.max(cos_hw)),
                'success': bool(mean_cos_hw > 0.85),
                'by_group': {
                              'high_gc': {'mean': float(np.mean(cos_hw[0:10])),  'std': float(np.std(cos_hw[0:10]))},
                              'low_gc':  {'mean': float(np.mean(cos_hw[10:20])), 'std': float(np.std(cos_hw[10:20]))},
                              'mixed':   {'mean': float(np.mean(cos_hw[20:30])), 'std': float(np.std(cos_hw[20:30]))},
                }
      },
      'cosines_per_seq': cos_hw,
}
with open(os.path.join(OUT_DIR, 'exp1d_results.json'), 'w') as f:
      json.dump(results, f, indent=2)

print(f'\n{"="*60}')
print(f'EXPERIMENT 1D COMPLETE')
print(f'  FakeFez cosine: {mean_cos_fake:.4f} +/- {std_cos_fake:.4f}')
print(f'  ibm_fez cosine: {mean_cos_hw:.4f} +/- {std_cos_hw:.4f}')
print(f'  Results saved to {OUT_DIR}/')
