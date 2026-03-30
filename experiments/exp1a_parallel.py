#!/usr/bin/env python3
"""Exp 1A Full - Parallelized: all 65,536 8-mers, multiprocessing + optional GPU."""
import sys, os, warnings, json, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import itertools
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from qubis_hiq.santalucia import compute_total_dg
from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.vienna_interface import predict_structure
from qubis_hiq.feature_extraction import extract_feature_vector

USE_GPU = os.environ.get('USE_GPU', '0') == '1'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1a')


def process_one(seq):
            """Process one sequence. Returns (dg, fv) tuple or None on error."""
            try:
                            dg = compute_total_dg(seq)
except (ValueError, KeyError):
        return None
    try:
                    _, stem_pairs = predict_structure(seq)
                    qc = build_circuit(seq, stem_pairs, np.zeros(12), include_measurement=False)
                    n_q = 2 * len(seq)
                    if USE_GPU:
                                        from qiskit_aer import AerSimulator
                                        from qiskit import transpile
                                        qcm = build_circuit(seq, stem_pairs, np.zeros(12), include_measurement=True)
                                        try:
                                                                sim = AerSimulator(method='statevector', device='GPU')
                                                                tc = transpile(qcm, sim)
                                                                res = sim.run(tc, shots=4096).result()
                                                                raw = res.get_counts()
                                                                str_counts = {str(k): v for k, v in raw.items()}
    except Exception:
                            from qiskit.quantum_info import Statevector
                            sv = Statevector.from_instruction(qc)
                            raw = sv.sample_counts(4096)
                            str_counts = {
                                (format(k, f'0{n_q}b') if isinstance(k, int) else str(k)): v
                                for k, v in raw.items()
                            }
else:
                    from qiskit.quantum_info import Statevector
                    sv = Statevector.from_instruction(qc)
                    raw = sv.sample_counts(4096)
                    str_counts = {
                        (format(k, f'0{n_q}b') if isinstance(k, int) else str(k)): v
                        for k, v in raw.items()
                    }
                fv = extract_feature_vector(str_counts, len(seq), stem_pairs)
        return (dg, fv)
except Exception:
        return None


def run_full(n_seqs=None, n_cpus=None):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            all_seqs = [''.join(s) for s in itertools.product('ATGC', repeat=8)]
            if n_seqs:
                            np.random.seed(42)
                            idx = np.random.choice(len(all_seqs), n_seqs, replace=False)
                            seqs = [all_seqs[i] for i in sorted(idx)]
else:
        seqs = all_seqs

    n_workers = n_cpus if n_cpus else cpu_count()
    print(
                    f"[{datetime.now()}] Exp 1A: {len(seqs)} seqs | {n_workers} CPUs | GPU={USE_GPU}",
                    flush=True,
    )

    dg_vals, feat_list = [], []
    chunk_size = max(500, len(seqs) // (n_workers * 4))
    for start in range(0, len(seqs), chunk_size):
                    batch = seqs[start:start + chunk_size]
                    with Pool(n_workers) as pool:
                                        results = pool.map(process_one, batch)
                                    for r in results:
                                                        if r is not None:
                                                                                dg_vals.append(r[0])
                                                                                feat_list.append(r[1])
                                                                        done = start + len(batch)
        print(f"  {done}/{len(seqs)} valid={len(dg_vals)}", flush=True)

    if not feat_list:
                    print("ERROR: No valid sequences processed.", flush=True)
        sys.exit(1)

    max_len = max(len(f) for f in feat_list)
    X = np.array([np.pad(f, (0, max_len - len(f))) for f in feat_list])
    y = np.array(dg_vals)
    print(f"\nFeature matrix: {X.shape}  dG range: [{y.min():.2f}, {y.max():.2f}]", flush=True)

    ridge = Ridge(alpha=1.0)
    cv5 = cross_val_score(ridge, X, y, cv=5, scoring='r2')
    ridge.fit(X, y)
    y_pred = ridge.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"\n5-fold CV R2: {cv5.mean():.4f} +/- {cv5.std():.4f}")
    print(f"Full-data R2={r2:.4f}  MAE={mae:.4f} kcal/mol  RMSE={rmse:.4f} kcal/mol")
    print(f"{'SUCCESS' if r2 > 0.90 else 'NEEDS IMPROVEMENT'}: R2={r2:.4f} (target>0.90)")

    out = {
                    "experiment": "1A_parallel",
                    "n_sequences": int(len(y)),
                    "n_features": int(X.shape[1]),
                    "r2_full": float(r2),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "cv_r2_mean": float(cv5.mean()),
                    "cv_r2_std": float(cv5.std()),
                    "dg_min": float(y.min()),
                    "dg_max": float(y.max()),
                    "n_workers": n_workers,
                    "use_gpu": USE_GPU,
                    "success": bool(r2 > 0.90),
    }
    tag = "gpu" if USE_GPU else f"cpu{n_workers}"
    with open(os.path.join(OUTPUT_DIR, f'exp1a_results_{tag}.json'), 'w') as f:
                    json.dump(out, f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, f'features_{tag}.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, f'dg_values_{tag}.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, f'predictions_{tag}.npy'), y_pred)
    print(f"Results saved to {OUTPUT_DIR}/")
    return out


if __name__ == '__main__':
            parser = argparse.ArgumentParser(description='Exp 1A: dG regression over all 8-mers.')
    parser.add_argument('--n-seqs', type=int, default=None,
                                                help='Number of sequences to sample (default: all 65,536)')
    parser.add_argument('--n-cpus', type=int, default=None,
                                                help='Number of CPU workers (default: all available)')
    args = parser.parse_args()
    run_full(n_seqs=args.n_seqs, n_cpus=args.n_cpus)
