"""Experiment 1D-torino: Cross-platform portability on ibm_torino (Heron r1, 133 qubits).

Proves the circuit output is not overfit to ibm_fez's specific noise profile.
Runs the same 30 diverse 8-mer sequences on ibm_torino and compares cosine
similarity against the shared statevector reference from exp1d_hardware.py.

Usage:
    export IBM_QUANTUM_TOKEN=<your_token>
        python experiments/exp1d_torino.py
        """

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.feature_extraction import extract_feature_vector

# ── Constants ──────────────────────────────────────────────────────────
N_SEQS  = 30
N_SHOTS = 4096
BASE    = 45
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1d_torino')
REF_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1d')
os.makedirs(OUT_DIR, exist_ok=True)

TOKEN = os.environ.get('IBM_QUANTUM_TOKEN')
if not TOKEN:
      raise EnvironmentError(
                "IBM_QUANTUM_TOKEN environment variable not set.\n"
                "Export it before running: export IBM_QUANTUM_TOKEN=<your_token>"
      )

# ibm_fez reference result (from exp1d_hardware.py)
IBM_FEZ_MEAN = 0.9970
IBM_FEZ_STD  = 0.0005

def gc_content(seq):
      return (seq.count('G') + seq.count('C')) / len(seq)

def make_diverse_seqs(n_per_group=10, seed=42):
      """Same generation as exp1d_hardware.py — produces identical 30 sequences."""
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

                  print(f'[{datetime.now()}] Exp 1D-torino: Cross-Platform Portability')
        print(f'  Backend: ibm_torino (Heron r1, 133q) | N_SEQS={N_SEQS} | N_SHOTS={N_SHOTS}')

# ── Step 1: Load statevector reference + rebuild circuits ───────────────
print('\n[1/2] Loading statevector reference and rebuilding circuits...')
sim_path = os.path.join(REF_DIR, 'sim_features.npy')
if not os.path.exists(sim_path):
      raise FileNotFoundError(
          f'Statevector reference not found at {sim_path}.\n'
          'Run exp1d_hardware.py first to generate it.'
)
sim_features = np.load(sim_path)
assert sim_features.shape == (N_SEQS, BASE), \
    f'Expected ({N_SEQS}, {BASE}), got {sim_features.shape}'
print(f'  Loaded sim_features: {sim_features.shape}')

seqs, seq_labels = make_diverse_seqs(n_per_group=10, seed=42)
params = np.zeros(12)
hw_circuits = [
      build_circuit(seq, stem_pairs=[], trainable_params=params, include_measurement=True)
      for seq in seqs
]
print(f'  Rebuilt {len(hw_circuits)} circuits.')

# ── Step 2: Transpile + submit to ibm_torino ────────────────────────────
print('\n[2/2] Transpiling and submitting to ibm_torino...')
t0 = time.time()
service = QiskitRuntimeService(channel='ibm_quantum_platform', token=TOKEN)
backend = service.backend('ibm_torino')
print(f'  Backend: {backend.name} | Qubits: {backend.num_qubits}')
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circs = [pm.run(qc) for qc in hw_circuits]
depths = [c.depth() for c in isa_circs]
print(f'  Transpiled in {time.time()-t0:.1f}s. Mean depth: {np.mean(depths):.1f} | Max: {max(depths)}')
sampler = Sampler(mode=backend)
job = sampler.run([(qc,) for qc in isa_circs], shots=N_SHOTS)
job_id = job.job_id()
print(f'  Job ID: {job_id}')
while True:
      status = str(job.status())
    print(f'  Status: {status} ({time.time()-t0:.0f}s)', flush=True)
    if status in ('JobStatus.DONE', 'DONE', 'done'):
              break
          if status in ('JobStatus.ERROR', 'ERROR', 'CANCELLED'):
                    print('ERROR: job failed'); sys.exit(1)
                time.sleep(15)
exec_time = time.time() - t0
print(f'  Job completed in {exec_time:.1f}s')

# ── Extract features + cosine similarity ────────────────────────────────
results_hw = job.result()
hw_features = np.array([
      extract_feature_vector(results_hw[i].data.meas.get_counts(), 8, [])[:BASE]
      for i in range(N_SEQS)
])
np.save(os.path.join(OUT_DIR, 'hw_features_torino.npy'), hw_features)
cos_torino = [float(sk_cosine([sim_features[i]], [hw_features[i]])[0][0])
                            for i in range(N_SEQS)]
mean_cos_torino = float(np.mean(cos_torino))
std_cos_torino  = float(np.std(cos_torino))
delta = float(mean_cos_torino - IBM_FEZ_MEAN)

print(f'\n  ibm_torino cosine (n={N_SEQS}): {mean_cos_torino:.4f} +/- {std_cos_torino:.4f}')
print(f'  ibm_fez reference:               {IBM_FEZ_MEAN:.4f} +/- {IBM_FEZ_STD:.4f}')
print(f'  Cross-platform delta:            {delta:+.4f}')
print(f'  Target >0.85: {"MET" if mean_cos_torino > 0.85 else "NOT MET"}')

output = {
      'experiment': '1D_torino',
      'backend': 'ibm_torino',
      'n_seqs': N_SEQS,
      'n_shots': N_SHOTS,
      'circuit_version': 'v2_ry_cx_ry',
      'job_id': job_id,
      'exec_seconds': exec_time,
      'sequences': seqs,
      'seq_labels': seq_labels,
      'ibm_torino': {
                'mean_cosine': mean_cos_torino,
                'std_cosine':  std_cos_torino,
                'min_cosine':  float(np.min(cos_torino)),
                'max_cosine':  float(np.max(cos_torino)),
                'success_85':  bool(mean_cos_torino > 0.85),
                'by_group': {
                              'high_gc': {'mean': float(np.mean(cos_torino[0:10])),  'std': float(np.std(cos_torino[0:10]))},
                              'low_gc':  {'mean': float(np.mean(cos_torino[10:20])), 'std': float(np.std(cos_torino[10:20]))},
                              'mixed':   {'mean': float(np.mean(cos_torino[20:30])), 'std': float(np.std(cos_torino[20:30]))},
                },
                'cosines': cos_torino,
      },
      'cross_platform': {
                'ibm_fez_mean':    IBM_FEZ_MEAN,
                'ibm_fez_std':     IBM_FEZ_STD,
                'ibm_torino_mean': mean_cos_torino,
                'ibm_torino_std':  std_cos_torino,
                'delta':           delta,
                'portable':        bool(abs(delta) < 0.01),
      },
}
with open(os.path.join(OUT_DIR, 'exp1d_torino_results.json'), 'w') as f:
      json.dump(output, f, indent=2)

print(f'\n{"="*60}')
print(f'EXPERIMENT 1D-TORINO COMPLETE')
print(f'  ibm_torino: {mean_cos_torino:.4f} +/- {std_cos_torino:.4f}')
print(f'  Portable (|delta|<0.01): {"YES" if abs(delta) < 0.01 else "MARGINAL"}')
print(f'  Results saved to {OUT_DIR}/')
