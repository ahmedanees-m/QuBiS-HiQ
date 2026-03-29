"""Parallelised Exp 1C: Non-Nearest-Neighbor Effect Detection.
Uses fixed circuit v2 (Ry encoding, CX+Ry stacking, no final Hadamard).

Strategy: Generate pairs where |delta_dg| < 1.0 AND struct1 != struct2.
Parallelises both pair generation and feature extraction.
"""
import sys, os, json, time
sys.path.insert(0, '/ibdc-scratch2/home/IBDCHPCU0095/qubis-hiq')

import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from qiskit.quantum_info import Statevector
from qubis.santalucia import compute_total_dg
from qubis.circuit_builder import build_circuit
from qubis.vienna_interface import predict_structure
from qubis.feature_extraction import extract_feature_vector


def try_generate_pair_8nt(seed):
    """Try to generate one valid 8-nt pair with the given seed.

    Returns (seq1, seq2, struct1, struct2, dg1, dg2) or None.
    A valid pair has: different structures AND |dg1-dg2| < 1.0 kcal/mol.
    """
    rng = np.random.RandomState(seed)
    for _ in range(60):
        seq1 = ''.join(rng.choice(list('ATGC'), 8))
        # Generate composition-matched permutation
        chars = list(seq1)
        rng.shuffle(chars)
        seq2 = ''.join(chars)
        if seq2 == seq1:
            continue
        try:
            struct1, pairs1 = predict_structure(seq1)
            struct2, pairs2 = predict_structure(seq2)
            if struct1 == struct2:
                continue
            # Require at least one sequence to have structure
            if len(pairs1) == 0 and len(pairs2) == 0:
                continue
            dg1 = compute_total_dg(seq1)
            dg2 = compute_total_dg(seq2)
            if abs(dg1 - dg2) < 1.0:
                return (seq1, seq2, struct1, struct2, float(dg1), float(dg2), 8)
        except Exception:
            continue
    return None


def try_generate_pair_10nt(seed):
    """Fallback: try to generate one valid 10-nt pair."""
    rng = np.random.RandomState(seed + 100000)
    for _ in range(60):
        seq1 = ''.join(rng.choice(list('ATGC'), 10))
        chars = list(seq1)
        rng.shuffle(chars)
        seq2 = ''.join(chars)
        if seq2 == seq1:
            continue
        try:
            struct1, pairs1 = predict_structure(seq1)
            struct2, pairs2 = predict_structure(seq2)
            if struct1 == struct2:
                continue
            if len(pairs1) == 0 and len(pairs2) == 0:
                continue
            dg1 = compute_total_dg(seq1)
            dg2 = compute_total_dg(seq2)
            if abs(dg1 - dg2) < 1.5:
                return (seq1, seq2, struct1, struct2, float(dg1), float(dg2), 10)
        except Exception:
            continue
    return None


def extract_qf(args):
    """Extract quantum features for (seq, stem_pairs)."""
    seq, stem_pairs = args
    params = np.zeros(12)
    qc = build_circuit(seq, stem_pairs, params, include_measurement=False)
    sv = Statevector.from_instruction(qc)
    n_q = 2 * len(seq)
    counts = {(format(k, '0{}b'.format(n_q)) if isinstance(k, int) else str(k)): v
              for k, v in sv.sample_counts(4096).items()}
    return extract_feature_vector(counts, len(seq), stem_pairs)


def get_stem_pairs_for_seq(seq):
    """Get stem pairs for a sequence (used serially before Pool)."""
    _, sp = predict_structure(seq)
    return (seq, sp)


def run(n_pairs=200):
    out_dir = '/ibdc-scratch2/home/IBDCHPCU0095/qubis-hiq/experiments/paper1/results/exp1c'
    os.makedirs(out_dir, exist_ok=True)
    n_cpu = cpu_count()

    print("[{}] Exp 1C: {} pairs | {} CPUs | fixed circuit v2".format(
        datetime.now(), n_pairs, n_cpu))

    # ── Step 1: Generate composition-matched pairs in parallel ──────────
    print("Step 1: Generating composition-matched 8-nt pairs (parallel)...")
    t0 = time.time()
    seeds_8nt = list(range(n_pairs * 25))
    with Pool(n_cpu) as pool:
        raw_8nt = pool.map(try_generate_pair_8nt, seeds_8nt)
    valid_pairs = [r for r in raw_8nt if r is not None]
    print("  8-nt pairs found: {}/{} in {:.1f}s".format(
        len(valid_pairs), n_pairs, time.time()-t0))

    # Fallback to 10-nt if insufficient pairs
    if len(valid_pairs) < max(20, n_pairs // 2):
        print("  Insufficient 8-nt pairs. Trying 10-nt sequences...")
        t1 = time.time()
        seeds_10nt = list(range(n_pairs * 25))
        with Pool(n_cpu) as pool:
            raw_10nt = pool.map(try_generate_pair_10nt, seeds_10nt)
        valid_10nt = [r for r in raw_10nt if r is not None]
        print("  10-nt pairs found: {} in {:.1f}s".format(len(valid_10nt), time.time()-t1))
        valid_pairs = valid_pairs + valid_10nt

    valid_pairs = valid_pairs[:n_pairs]
    n_found = len(valid_pairs)
    seq_len = valid_pairs[0][6] if n_found > 0 else 8
    print("Total valid pairs: {} (seq_len={})".format(n_found, seq_len))

    if n_found < 10:
        print("ERROR: Too few pairs found. Cannot continue reliably.")
        with open('{}/exp1c_results.json'.format(out_dir), 'w') as f:
            json.dump({'error': 'too_few_pairs', 'n_found': n_found}, f, indent=2)
        return

    # ── Step 2: Get stem pairs for all sequences (quick, serial or parallel) ──
    print("Step 2: Computing secondary structures...")
    t2 = time.time()
    all_seq_list = []
    for p in valid_pairs:
        all_seq_list.append(p[0])  # seq1
        all_seq_list.append(p[1])  # seq2

    with Pool(n_cpu) as pool:
        seq_sp_pairs = pool.map(get_stem_pairs_for_seq, all_seq_list)
    print("  Done in {:.1f}s".format(time.time()-t2))

    # ── Step 3: Extract quantum features in parallel ──────────────────
    print("Step 3: Extracting quantum features ({} sequences)...".format(len(seq_sp_pairs)))
    t3 = time.time()
    with Pool(n_cpu) as pool:
        all_fv = pool.map(extract_qf, seq_sp_pairs)
    print("  Done in {:.1f}s".format(time.time()-t3))

    features_a = all_fv[0::2]  # seq1 features (even indices)
    features_b = all_fv[1::2]  # seq2 features (odd indices)

    # Pad to uniform feature vector width
    max_w = max(fv.shape[0] for fv in all_fv)
    X_a = np.vstack([np.pad(fv, (0, max_w - len(fv))) for fv in features_a])
    X_b = np.vstack([np.pad(fv, (0, max_w - len(fv))) for fv in features_b])
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(len(features_a)), np.ones(len(features_b))])

    # ── Step 4: SVM Classification ───────────────────────────────────
    print("Step 4: SVM classification ({} samples x {} features)...".format(
        X.shape[0], X.shape[1]))
    results_svm = {}
    for kernel in ['rbf', 'linear']:
        svc = SVC(kernel=kernel, C=1.0, random_state=42)
        scores = cross_val_score(svc, X, y, cv=5, scoring='accuracy')
        results_svm[kernel] = {
            'accuracy_mean': float(scores.mean()),
            'accuracy_std':  float(scores.std()),
            'success':       bool(scores.mean() > 0.70)
        }
        print("  SVM-{}: {:.4f} +/- {:.4f}  [{}]".format(
            kernel.upper(), scores.mean(), scores.std(),
            'SUCCESS' if scores.mean() > 0.70 else 'below target'))

    best_acc = max(v['accuracy_mean'] for v in results_svm.values())
    best_kernel = max(results_svm, key=lambda k: results_svm[k]['accuracy_mean'])

    # ── Step 5: Save results ─────────────────────────────────────────
    pairs_meta = [
        {'seq1': p[0], 'seq2': p[1], 'struct1': p[2], 'struct2': p[3],
         'dg1': p[4], 'dg2': p[5], 'seq_len': p[6]}
        for p in valid_pairs
    ]

    results = {
        'experiment': '1C',
        'n_pairs_requested': n_pairs,
        'n_pairs_found':     n_found,
        'seq_len':           seq_len,
        'feature_dim':       int(max_w),
        'circuit_version':   'v2_ry_cx_ry',
        'svm_rbf':           results_svm['rbf'],
        'svm_linear':        results_svm['linear'],
        'best_accuracy':     float(best_acc),
        'best_kernel':       best_kernel,
        'success':           bool(best_acc > 0.70)
    }

    np.save('{}/features_a.npy'.format(out_dir), X_a)
    np.save('{}/features_b.npy'.format(out_dir), X_b)
    with open('{}/exp1c_results.json'.format(out_dir), 'w') as f:
        json.dump(results, f, indent=2)
    with open('{}/pairs_metadata.json'.format(out_dir), 'w') as f:
        json.dump(pairs_meta, f, indent=2)

    print("")
    print("=" * 55)
    print("Best accuracy: {:.4f} (SVM-{})".format(best_acc, best_kernel.upper()))
    print("Target >0.70: {}".format('MET' if best_acc > 0.70 else 'NOT MET'))
    print("=" * 55)
    print("Results saved to {}/".format(out_dir))
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(n)
