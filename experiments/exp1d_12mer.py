"""Experiment 1D Extension: 12-mer scalability + error mitigation on ibm_fez.

Two tests on 20 diverse 12-mer sequences (24 qubits):
  1. Scalability baseline — SamplerV2 default (no mitigation)
    2. Error mitigation   — Dynamical Decoupling (XY4) + Pauli Twirling

    Noiseless reference: AerSimulator statevector (handles 24 qubits, ~9 s/circuit).
    Feature vector: 24 + 23 + 22 = 69 dimensions per sequence.

    Usage:
        export IBM_QUANTUM_TOKEN=<your_token>
            python experiments/exp1d_12mer.py
            """

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from qiskit_aer import AerSimulator
from qiskit import transpile as qiskit_transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions
from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.feature_extraction import extract_feature_vector

# ── Constants ──────────────────────────────────────────────────────────
N_SEQS     = 20
N_SHOTS    = 4096
N_SHOTS_SV = 8192
BASE_12    = 69     # 24 + 23 + 22 feature dims for 12-mers
SEQ_LEN    = 12
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1d_12mer')
os.makedirs(OUT_DIR, exist_ok=True)

TOKEN = os.environ.get('IBM_QUANTUM_TOKEN')
if not TOKEN:
      raise EnvironmentError(
                "IBM_QUANTUM_TOKEN environment variable not set.\n"
                "Export it before running: export IBM_QUANTUM_TOKEN=<your_token>"
      )

def gc_content(seq):
      return (seq.count('G') + seq.count('C')) / len(seq)

def make_diverse_12mers(seed=99):
      """20 diverse 12-mers: 7 high-GC, 7 low-GC, 6 mixed."""
      rng = np.random.RandomState(seed)
      group_specs = [
          ('high_gc', 'GC',   0.75, 1.01, 7),
          ('low_gc',  'AT',   0.00, 0.26, 7),
          ('mixed',   'ATGC', 0.25, 0.76, 6),
      ]
      seqs, labels = [], []
      seen = set()
      for gname, alphabet, gc_lo, gc_hi, count in group_specs:
                found = 0
                for _ in range(500000):
                              seq = ''.join(rng.choice(list(alphabet), SEQ_LEN))
                              gc = gc_content(seq)
                              if gc_lo <= gc < gc_hi and seq not in seen:
                                                seen.add(seq)
                                                seqs.append(seq)
                                                labels.append(gname)
                                                found += 1
                                                if found == count:
                                                                      break
                                                          if found < count:
                                                                        raise RuntimeError(f'Only found {found}/{count} seqs for {gname}')
                                                                return seqs, labels

                  print(f'[{datetime.now()}] Exp 1D-12mer: Scalability + Error Mitigation')
        print(f'  SEQ_LEN={SEQ_LEN} | N_SEQS={N_SEQS} | N_SHOTS={N_SHOTS} | BASE={BASE_12}')

seqs, seq_labels = make_diverse_12mers(seed=99)
assert len(seqs) == N_SEQS
gc_vals = [gc_content(s) for s in seqs]
params = np.zeros(12)

# ── Step 1: AerSimulator noiseless reference ────────────────────────────
print(f'\n[1/3] AerSimulator statevector reference ({N_SEQS} 24-qubit circuits)...')
t0 = time.time()
aer_sim = AerSimulator(method='statevector')
sim_features, hw_circuits = [], []
for i, seq in enumerate(seqs):
      qc_aer = build_circuit(seq, stem_pairs=[], trainable_params=params,
                                                         include_measurement=True)
    qc_t = qiskit_transpile(qc_aer, aer_sim, optimization_level=0)
    counts = aer_sim.run(qc_t, shots=N_SHOTS_SV).result().get_counts()
    sim_features.append(extract_feature_vector(counts, len(seq), [])[:BASE_12])
    hw_circuits.append(build_circuit(seq, stem_pairs=[], trainable_params=params,
                                                                          include_measurement=True))
    if (i + 1) % 5 == 0:
              print(f'  {i+1}/{N_SEQS} done ({time.time()-t0:.0f}s)')
      sim_features = np.array(sim_features)
np.save(os.path.join(OUT_DIR, 'sim_features_12mer.npy'), sim_features)
print(f'  AerSim done in {time.time()-t0:.1f}s. Features: {sim_features.shape}')

# ── Step 2: Transpile for ibm_fez ──────────────────────────────────────
print(f'\n[2/3] Transpiling {N_SEQS} circuits for ibm_fez...')
t1 = time.time()
service = QiskitRuntimeService(channel='ibm_quantum_platform', token=TOKEN)
backend = service.backend('ibm_fez')
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuits = [pm.run(qc) for qc in hw_circuits]
depths = [c.depth() for c in isa_circuits]
print(f'  Transpiled in {time.time()-t1:.1f}s. Mean depth: {np.mean(depths):.1f} | Max: {max(depths)}')

def extract_hw_features(job_result, n_seqs):
      return np.array([
          extract_feature_vector(job_result[i].data.meas.get_counts(), SEQ_LEN, [])[:BASE_12]
          for i in range(n_seqs)
])

def compute_cosines(sim_feats, hw_feats):
      return [float(sk_cosine([sim_feats[i]], [hw_feats[i]])[0][0])
                          for i in range(len(sim_feats))]

def wait_for_job(job, t_start):
      while True:
                status = str(job.status())
                print(f'  Status: {status} ({time.time()-t_start:.0f}s)', flush=True)
                if status in ('JobStatus.DONE', 'DONE', 'done'):
                              return
                          if status in ('JobStatus.ERROR', 'ERROR', 'CANCELLED'):
                                        print('ERROR: job failed'); sys.exit(1)
                                    time.sleep(15)

# ── Step 3a: Baseline (no error mitigation) ─────────────────────────────
print(f'\n[3a/3] IBM Hardware: BASELINE (no mitigation)...')
t2 = time.time()
sampler_base = Sampler(mode=backend)
job_base = sampler_base.run([(qc,) for qc in isa_circuits], shots=N_SHOTS)
job_id_base = job_base.job_id()
print(f'  Job ID (baseline): {job_id_base}')
wait_for_job(job_base, t2)
hw_base = extract_hw_features(job_base.result(), N_SEQS)
np.save(os.path.join(OUT_DIR, 'hw_features_12mer_base.npy'), hw_base)
cos_base = compute_cosines(sim_features, hw_base)
mean_base, std_base = float(np.mean(cos_base)), float(np.std(cos_base))
time_base = time.time() - t2
print(f'  Baseline cosine: {mean_base:.4f} +/- {std_base:.4f} ({time_base:.1f}s)')
print(f'  Target >0.95: {"MET" if mean_base > 0.95 else "NOT MET"}')

# ── Step 3b: Error mitigation (DD-XY4 + Pauli Twirling) ────────────────
print(f'\n[3b/3] IBM Hardware: ERROR MITIGATION (DD-XY4 + Twirling)...')
t3 = time.time()
options_em = SamplerOptions()
options_em.dynamical_decoupling.enable = True
options_em.dynamical_decoupling.sequence_type = 'XY4'
options_em.twirling.enable_gates = True
options_em.twirling.enable_measure = True
options_em.twirling.num_randomizations = 'auto'
sampler_em = Sampler(mode=backend, options=options_em)
job_em = sampler_em.run([(qc,) for qc in isa_circuits], shots=N_SHOTS)
job_id_em = job_em.job_id()
print(f'  Job ID (mitigated): {job_id_em}')
wait_for_job(job_em, t3)
hw_em = extract_hw_features(job_em.result(), N_SEQS)
np.save(os.path.join(OUT_DIR, 'hw_features_12mer_em.npy'), hw_em)
cos_em = compute_cosines(sim_features, hw_em)
mean_em, std_em = float(np.mean(cos_em)), float(np.std(cos_em))
time_em = time.time() - t3
print(f'  Mitigated cosine: {mean_em:.4f} +/- {std_em:.4f} ({time_em:.1f}s)')
print(f'  Target >0.97: {"MET" if mean_em > 0.97 else "NOT MET"}')

def group_stats(cos_list, counts=(7, 7, 6)):
      n0, n1, n2 = counts
    h, l, m = cos_list[:n0], cos_list[n0:n0+n1], cos_list[n0+n1:]
    return {
              'high_gc': {'mean': float(np.mean(h)), 'std': float(np.std(h)), 'n': len(h)},
              'low_gc':  {'mean': float(np.mean(l)), 'std': float(np.std(l)), 'n': len(l)},
              'mixed':   {'mean': float(np.mean(m)), 'std': float(np.std(m)), 'n': len(m)},
    }

results = {
      'experiment': '1D_12mer',
      'seq_len': SEQ_LEN,
      'n_qubits': 2 * SEQ_LEN,
      'n_seqs': N_SEQS,
      'n_shots': N_SHOTS,
      'backend': 'ibm_fez',
      'circuit_version': 'v2_ry_cx_ry',
      'sequences': seqs,
      'seq_labels': seq_labels,
      'gc_contents': gc_vals,
      'baseline': {
                'job_id': job_id_base,
                'exec_seconds': time_base,
                'mean_cosine': mean_base,
                'std_cosine':  std_base,
                'min_cosine':  float(np.min(cos_base)),
                'max_cosine':  float(np.max(cos_base)),
                'success_95':  bool(mean_base > 0.95),
                'by_group':    group_stats(cos_base),
                'cosines':     cos_base,
      },
      'error_mitigated': {
                'job_id': job_id_em,
                'exec_seconds': time_em,
                'mean_cosine': mean_em,
                'std_cosine':  std_em,
                'min_cosine':  float(np.min(cos_em)),
                'max_cosine':  float(np.max(cos_em)),
                'success_97':  bool(mean_em > 0.97),
                'dd_sequence': 'XY4',
                'twirling':    'gate+measure',
                'by_group':    group_stats(cos_em),
                'cosines':     cos_em,
      },
      'improvement': {
                'delta_cosine':  float(mean_em - mean_base),
                'relative_pct':  float(100.0 * (mean_em - mean_base) / max(1.0 - mean_base, 1e-9)),
      },
}
with open(os.path.join(OUT_DIR, 'exp1d_12mer_results.json'), 'w') as f:
      json.dump(results, f, indent=2)

print(f'\n{"="*65}')
print(f'EXPERIMENT 1D-12mer COMPLETE')
print(f'  Baseline cosine:  {mean_base:.4f} +/- {std_base:.4f}  target >0.95: {"MET" if mean_base > 0.95 else "NOT MET"}')
print(f'  Mitigated cosine: {mean_em:.4f} +/- {std_em:.4f}  target >0.97: {"MET" if mean_em > 0.97 else "NOT MET"}')
print(f'  Improvement: delta={results["improvement"]["delta_cosine"]:.4f} '
            f'({results["improvement"]["relative_pct"]:.1f}% error reduction)')
print(f'  Results saved to {OUT_DIR}/')
