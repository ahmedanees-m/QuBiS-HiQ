"""Parallelised Exp 1B: Five-Way Ablation Study."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from qiskit.quantum_info import Statevector

from qubis_hiq.santalucia import compute_total_dg
from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.classical_twin import classical_feature_vector
from qubis_hiq.vienna_interface import predict_structure
from qubis_hiq.feature_extraction import extract_feature_vector

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1b')


def extract_qf(seq, stem_pairs, params, **kw):
        qc = build_circuit(seq, stem_pairs, params, include_measurement=False, **kw)
        sv = Statevector.from_instruction(qc)
        n_q = 2 * len(seq)
        counts = {(format(k, f'0{n_q}b') if isinstance(k, int) else str(k)): v
                  for k, v in sv.sample_counts(4096).items()}
        return extract_feature_vector(counts, len(seq), stem_pairs)


def process_seq(seq):
        try:
                    dg = compute_total_dg(seq)
                    _, stem_pairs = predict_structure(seq)
                    params = np.zeros(12)
                    full_fv = extract_qf(seq, stem_pairs, params)
                    no_wc_fv = extract_qf(seq, stem_pairs, params, skip_wc=True)
                    no_stack_fv = extract_qf(seq, stem_pairs, params, skip_stacking=True)
                    random_fv = extract_qf(seq, stem_pairs, params, random_angles=True)
                    classical_fv = classical_feature_vector(seq, stem_pairs)
        return (dg, full_fv, no_wc_fv, no_stack_fv, random_fv, classical_fv)
except Exception:
        return None


def run(n_sequences=500):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        n_cpu = cpu_count()
        np.random.seed(42)
        seqs = [''.join(np.random.choice(list('ATGC'), 8)) for _ in range(n_sequences)]

    print(f"[{datetime.now()}] Exp 1B: {n_sequences} seqs | {n_cpu} CPUs | circuit v2")
    t0 = time.time()
    with Pool(n_cpu) as pool:
                raw = pool.map(process_seq, seqs)
            valid = [r for r in raw if r is not None]
    print(f"Valid: {len(valid)}/{n_sequences} in {time.time()-t0:.1f}s")

    dg_vals = []
    feats = {k: [] for k in ['full', 'no_wc', 'no_stacking', 'random', 'classical']}
    for r in valid:
                dg_vals.append(r[0])
                feats['full'].append(r[1])
                feats['no_wc'].append(r[2])
                feats['no_stacking'].append(r[3])
                feats['random'].append(r[4])
                feats['classical'].append(r[5])

    y = np.array(dg_vals)
    results = {}
    r2_means = {}
    variants = ['full', 'no_wc', 'no_stacking', 'random', 'classical']

    print(f"\n{'Variant':<16} {'CV R2 mean+/-std':<26} Rank")
    print("=" * 52)
    for v in variants:
                X_list = feats[v]
                max_w = max(x.shape[0] for x in X_list)
                X = np.vstack([np.pad(x, (0, max_w - len(x))) for x in X_list])
                sc = cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring='r2')
                r2_means[v] = sc.mean()
                results[v] = {'r2_mean': float(sc.mean()), 'r2_std': float(sc.std())}
                np.save(os.path.join(OUTPUT_DIR, f'features_{v}.npy'), X)

    ranked = sorted(r2_means.items(), key=lambda x: -x[1])
    for rank, (v, r2) in enumerate(ranked, 1):
                std = results[v]['r2_std']
                print(f"{v:<16} {r2:.4f} +/- {std:.4f}  {rank}")
                results[v]['rank'] = rank

    expected = (
                r2_means['full'] > max(r2_means['no_wc'], r2_means['no_stacking'])
                and min(r2_means['no_wc'], r2_means['no_stacking'])
                > max(r2_means['random'], r2_means['classical'])
    )
    print(f"\nExpected ordering holds: {'YES' if expected else 'PARTIAL'}")
    results['expected_ordering_holds'] = bool(expected)
    results['n_valid'] = len(valid)
    results['n_sequences'] = n_sequences
    results['circuit_version'] = 'v2_ry_cx_ry'

    np.save(os.path.join(OUTPUT_DIR, 'dg_values.npy'), y)
    with open(os.path.join(OUTPUT_DIR, 'exp1b_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Exp 1B: Five-way ablation study.')
        parser.add_argument('--n-seqs', type=int, default=500,
                            help='Number of random 8-mer sequences (default: 500)')
        args = parser.parse_args()
        run(args.n_seqs)
