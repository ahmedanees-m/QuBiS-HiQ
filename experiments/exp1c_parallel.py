"""Exp 1C: Non-nearest-neighbor effect detection via structural classification.

Generates composition-matched pairs where sequences differ only in secondary
structure (not dG°), then classifies them using quantum feature vectors.

Usage:
  python experiments/exp1c_parallel.py [--n-pairs 200]
  """
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from qiskit.quantum_info import Statevector

from qubis_hiq.santalucia import compute_total_dg
from qubis_hiq.circuit_builder import build_circuit
from qubis_hiq.vienna_interface import predict_structure
from qubis_hiq.feature_extraction import extract_feature_vector

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1c')


def try_generate_pair_8nt(seed):
        """Try to generate one valid composition-matched 8-nt pair.

            Returns (seq1, seq2, struct1, struct2, dg1, dg2, 8) or None.
                Valid pair: different structures AND |dg1 - dg2| < 1.0 kcal/mol.
                    """
        rng = np.random.RandomState(seed)
        for _ in range(60):
                    seq1 = ''.join(rng.choice(list('ATGC'), 8))
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
                                                if abs(dg1 - dg2) < 1.0:
                                                                    return (seq1, seq2, struct1, struct2, float(dg1), float(dg2), 8)
                    except Exception:
                                    continue
                            return None


def try_generate_pair_10nt(seed):
        """Fallback: try to generate one valid composition-matched 10-nt pair."""
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
        qc = build_circuit(seq, stem_pairs, np.zeros(12), include_measurement=False)
        sv = Statevector.from_instruction(qc)
        n_q = 2 * len(seq)
        counts = {(format(k, f'0{n_q}b') if isinstance(k, int) else str(k)): v
                  for k, v in sv.sample_counts(4096).items()}
        return extract_feature_vector(counts, len(seq), stem_pairs)


def get_stem_pairs_for_seq(seq):
        _, sp = predict_structure(seq)
        return (seq, sp)


def run(n_pairs=200):
        os.makedirs(OUT_DIR, exist_ok=True)
        n_cpu = cpu_count()
        print(f'[{datetime.now()}] Exp 1C: {n_pairs} pairs | {n_cpu} CPUs')

    # Step 1: generate composition-matched pairs
        print('Step 1: Generating composition-matched 8-nt pairs...')
    t0 = time.time()
    with Pool(n_cpu) as pool:
                raw_8nt = pool.map(try_generate_pair_8nt, range(n_pairs * 25))
            valid_pairs = [r for r in raw_8nt if r is not None]
    print(f'  8-nt pairs: {len(valid_pairs)}/{n_pairs} in {time.time()-t0:.1f}s')

    if len(valid_pairs) < max(20, n_pairs // 2):
                print('  Insufficient 8-nt pairs — trying 10-nt fallback...')
                with Pool(n_cpu) as pool:
                                raw_10nt = pool.map(try_generate_pair_10nt, range(n_pairs * 25))
                            valid_pairs += [r for r in raw_10nt if r is not None]

    valid_pairs = valid_pairs[:n_pairs]
    n_found = len(valid_pairs)
    print(f'  Total valid pairs: {n_found}')
    if n_found < 10:
                print('ERROR: Too few pairs. Cannot continue.')
        with open(os.path.join(OUT_DIR, 'exp1c_results.json'), 'w') as f:
                        json.dump({'error': 'too_few_pairs', 'n_found': n_found}, f, indent=2)
                    return

    # Step 2: get stem pairs for all sequences
    print('Step 2: Computing secondary structures...')
    t2 = time.time()
    all_seqs = [seq for p in valid_pairs for seq in (p[0], p[1])]
    with Pool(n_cpu) as pool:
                seq_sp_list = pool.map(get_stem_pairs_for_seq, all_seqs)
    print(f'  Done in {time.time()-t2:.1f}s')

    # Step 3: extract quantum features
    print(f'Step 3: Extracting quantum features ({len(seq_sp_list)} sequences)...')
    t3 = time.time()
    with Pool(n_cpu) as pool:
                all_fv = pool.map(extract_qf, seq_sp_list)
    print(f'  Done in {time.time()-t3:.1f}s')

    features_a = all_fv[0::2]
    features_b = all_fv[1::2]
    max_w = max(fv.shape[0] for fv in all_fv)
    X_a = np.vstack([np.pad(fv, (0, max_w - len(fv))) for fv in features_a])
    X_b = np.vstack([np.pad(fv, (0, max_w - len(fv))) for fv in features_b])
    X   = np.vstack([X_a, X_b])
    y   = np.concatenate([np.zeros(len(features_a)), np.ones(len(features_b))])

    # Step 4: classification
    print(f'Step 4: Classification ({X.shape[0]} samples x {X.shape[1]} features)...')
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    results_clf = {}
    for name, clf in [
                ('SVM-RBF',    SVC(kernel='rbf',    C=10,  random_state=42)),
                ('SVM-Linear', SVC(kernel='linear', C=1,   random_state=42)),
                ('RF-100',     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]:
                scores = cross_val_score(clf, Xs, y, cv=5, scoring='accuracy')
        results_clf[name] = {'mean': float(scores.mean()), 'std': float(scores.std())}
        tag = 'SUCCESS' if scores.mean() > 0.70 else 'below target'
        print(f'  {name:<14}: {scores.mean():.4f} +/- {scores.std():.4f}  [{tag}]')

    best_acc    = max(v['mean'] for v in results_clf.values())
    best_kernel = max(results_clf, key=lambda k: results_clf[k]['mean'])

    # Step 5: save
    np.save(os.path.join(OUT_DIR, 'features_a.npy'), X_a)
    np.save(os.path.join(OUT_DIR, 'features_b.npy'), X_b)
    pairs_meta = [
                {'seq1': p[0], 'seq2': p[1], 'struct1': p[2], 'struct2': p[3],
                          'dg1': p[4], 'dg2': p[5], 'seq_len': p[6]}
                for p in valid_pairs
    ]
    output = {
                'experiment': '1C',
                'n_pairs_requested': n_pairs, 'n_pairs_found': n_found,
                'feature_dim': int(max_w), 'circuit_version': 'v2_ry_cx_ry',
                'classifiers': results_clf,
                'best_accuracy': float(best_acc), 'best_classifier': best_kernel,
                'success': bool(best_acc > 0.70),
    }
    with open(os.path.join(OUT_DIR, 'exp1c_results.json'), 'w') as f:
                json.dump(output, f, indent=2)
    with open(os.path.join(OUT_DIR, 'pairs_metadata.json'), 'w') as f:
                json.dump(pairs_meta, f, indent=2)

    print(f'\n{"="*55}')
    print(f'Best accuracy: {best_acc:.4f} ({best_kernel})')
    print(f'Target >0.70: {"MET" if best_acc > 0.70 else "NOT MET"}')
    print(f'Results saved to {OUT_DIR}/')
    return output


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Exp 1C: structural classification.')
    parser.add_argument('--n-pairs', type=int, default=200,
                                                help='Number of composition-matched pairs (default: 200)')
    args = parser.parse_args()
    run(args.n_pairs)
