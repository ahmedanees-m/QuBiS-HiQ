"""Deeper analysis of Exp 1C results.

Reads features_a.npy, features_b.npy, and pairs_metadata.json written by
exp1c_parallel.py and runs additional classification tests including:
  - Full features (with stem correlators)
  - Base features only (interference terms, no stem indicator)
  - Pairwise delta features
  - dG regression sanity check

Usage:
  python experiments/exp1c_parallel.py --n-pairs 200   # run first
  python experiments/analyze_exp1c.py
"""
import os
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# -- Locate results relative to this script ---------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp1c')

X_a_path = os.path.join(RESULTS_DIR, 'features_a.npy')
X_b_path = os.path.join(RESULTS_DIR, 'features_b.npy')
meta_path = os.path.join(RESULTS_DIR, 'pairs_metadata.json')

for path in (X_a_path, X_b_path, meta_path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Missing: {path}\n'
            'Run exp1c_parallel.py first to generate the feature files.'
        )

X_a = np.load(X_a_path)
X_b = np.load(X_b_path)
with open(meta_path) as f:
    meta = json.load(f)


def has_structure(struct_str):
    return any(c != '.' for c in struct_str)


labels_a = np.array([1 if has_structure(m['struct1']) else 0 for m in meta])
labels_b = np.array([1 if has_structure(m['struct2']) else 0 for m in meta])

print(f'N={len(meta)} pairs | N_features={X_a.shape[1]}')
print(f'Structure balance: {int((labels_a+labels_b).sum())} structured / '
      f'{int(2*len(meta) - (labels_a+labels_b).sum())} unstructured '
      f'(total {2*len(meta)} seqs)')

BASE = 45  # 16 z_local + 15 zz_nn + 14 zz_nnn (8-nt, no stem pairs)

if X_a.shape[1] > BASE:
    print(f'\nStem correlator columns (cols {BASE}+):')
    print(f'  Structured   mean |col{BASE}|: {abs(X_a[labels_a==1, BASE]).mean():.5f}')
    print(f'  Unstructured mean |col{BASE}|: {abs(X_a[labels_a==0, BASE]).mean():.5f}')

X_all = np.vstack([X_a, X_b])
y = np.concatenate([labels_a, labels_b])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    'SVM-RBF C=10':  SVC(kernel='rbf',    C=10,  gamma='scale', random_state=42),
    'SVM-RBF C=100': SVC(kernel='rbf',    C=100, gamma='scale', random_state=42),
    'SVM-Linear':    SVC(kernel='linear', C=1,   random_state=42),
    'RF-100':        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

# -- Test 1: Full features --------------------------------------------------
print(f'\n[Full {X_all.shape[1]} features (incl. stem correlators)]:')
Xs_full = StandardScaler().fit_transform(X_all)
for name, clf in classifiers.items():
    s = cross_val_score(clf, Xs_full, y, cv=cv, scoring='accuracy')
    print(f'  {name:<22}: {s.mean():.4f} +/- {s.std():.4f}')

# -- Test 2: Base features only (the scientifically meaningful test) --------
print(f'\n[Base {BASE} features only — interference terms, no stem indicator]:')
X45 = X_all[:, :BASE]
Xs_base = StandardScaler().fit_transform(X45)
results_base = {}
for name, clf in classifiers.items():
    s = cross_val_score(clf, Xs_base, y, cv=cv, scoring='accuracy')
    tag = 'SUCCESS' if s.mean() > 0.70 else 'below'
    print(f'  {name:<22}: {s.mean():.4f} +/- {s.std():.4f}  [{tag}]')
    results_base[name] = (float(s.mean()), float(s.std()))

# -- Test 3: Pairwise delta features ----------------------------------------
print(f'\n[Pairwise delta (fv1 - fv2), base {BASE}]:')
delta45 = (X_a - X_b)[:, :BASE]
y_delta = labels_a
mask_one = (labels_a + labels_b) == 1
print(f'  Pairs with exactly one structured seq: {mask_one.sum()}/{len(meta)}')
if mask_one.sum() >= 20:
    Xd = StandardScaler().fit_transform(delta45[mask_one])
    yd = y_delta[mask_one]
    for name, clf in [
        ('SVM-RBF', SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
        ('RF-100',  RandomForestClassifier(n_estimators=100, random_state=42)),
    ]:
        s = cross_val_score(clf, Xd, yd, cv=5, scoring='accuracy')
        tag = 'SUCCESS' if s.mean() > 0.70 else 'below'
        print(f'  {name:<12}: {s.mean():.4f} +/- {s.std():.4f}  [{tag}]')

# -- Test 4: dG regression sanity check -------------------------------------
print(f'\n[dG regression on base {BASE} features (sanity check)]:')
dg = np.array([m['dg1'] for m in meta] + [m['dg2'] for m in meta])
r2s = cross_val_score(Ridge(alpha=1.0), Xs_base, dg, cv=cv, scoring='r2')
print(f'  Ridge R2: {r2s.mean():.4f} +/- {r2s.std():.4f}')

# -- Save revised results ---------------------------------------------------
best_nm = max(results_base, key=lambda k: results_base[k][0])
best_acc = results_base[best_nm][0]
best_std = results_base[best_nm][1]

revised = {
    'experiment': '1C_analysis',
    'n_pairs': len(meta),
    'n_structured': int((labels_a + labels_b).sum()),
    'n_unstructured': int(2*len(meta) - (labels_a+labels_b).sum()),
    'feature_dim_used': BASE,
    'best_classifier': best_nm,
    'best_accuracy_mean': float(best_acc),
    'best_accuracy_std': float(best_std),
    'success': bool(best_acc > 0.70),
    'classifiers': {k: {'mean': v[0], 'std': v[1]} for k, v in results_base.items()},
    'dg_r2_base': float(r2s.mean()),
}
out_path = os.path.join(RESULTS_DIR, 'exp1c_analysis_results.json')
with open(out_path, 'w') as f:
    json.dump(revised, f, indent=2)

print(f'\nAnalysis results saved to {out_path}')
print(f'Best accuracy (base {BASE}): {best_acc:.4f}  '
      f'target >0.70: {"MET" if best_acc > 0.70 else "NOT MET"}')
