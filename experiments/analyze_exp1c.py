Analysis of Exp 1C with structural labels and base-only features.
import numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

X_a = np.load('D:/Qubis_HiQ/results_exp1c/features_a.npy')
X_b = np.load('D:/Qubis_HiQ/results_exp1c/features_b.npy')
with open('D:/Qubis_HiQ/results_exp1c/pairs_metadata.json') as f:
    meta = json.load(f)

def has_structure(struct_str):
    return any(c != '.' for c in struct_str)

labels_a = np.array([1 if has_structure(m['struct1']) else 0 for m in meta])
labels_b = np.array([1 if has_structure(m['struct2']) else 0 for m in meta])

print('N=%d pairs | N_features=%d' % (len(meta), X_a.shape[1]))
print('Structure balance: %d structured / %d unstructured (total %d seqs)' % (
    int((labels_a+labels_b).sum()), int(2*len(meta) - (labels_a+labels_b).sum()), 2*len(meta)))

BASE = 45  # 16 z_local + 15 zz_nn + 14 zz_nnn (8-nt, no stem pairs)

print('\nChecking stem correlator columns (cols 45-46):')
if X_a.shape[1] > 45:
    print('  Structured   seq mean |col45|: %.5f' % abs(X_a[labels_a==1, 45]).mean())
    print('  Unstructured seq mean |col45|: %.5f' % abs(X_a[labels_a==0, 45]).mean())
if X_a.shape[1] > 46:
    print('  Structured   seq mean |col46|: %.5f' % abs(X_a[labels_a==1, 46]).mean())
    print('  Unstructured seq mean |col46|: %.5f' % abs(X_a[labels_a==0, 46]).mean())

# ── Test 1: Full 47 features (stem correlators included) ─────────────
X_all = np.vstack([X_a, X_b])
y = np.concatenate([labels_a, labels_b])
sc1 = StandardScaler()
Xs_full = sc1.fit_transform(X_all)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
s = cross_val_score(SVC(kernel='rbf', C=10, gamma='scale'), Xs_full, y, cv=cv, scoring='accuracy')
print('\n[Full 47 features (incl. stem correlators)]:')
print('  SVM-RBF: %.4f +/- %.4f' % (s.mean(), s.std()))

# ── Test 2: Base 45 features only (interference terms, no stem indicator) ──
print('\n[Base 45 features only — the scientifically meaningful test]:')
X_all45 = X_all[:, :BASE]
sc2 = StandardScaler()
Xs_base = sc2.fit_transform(X_all45)
results_base = {}
for nm, clf in [
    ('SVM-RBF C=10',  SVC(kernel='rbf',    C=10,  gamma='scale', random_state=42)),
    ('SVM-RBF C=100', SVC(kernel='rbf',    C=100, gamma='scale', random_state=42)),
    ('SVM-Linear',    SVC(kernel='linear', C=1,   random_state=42)),
    ('RF-100',        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
]:
    s = cross_val_score(clf, Xs_base, y, cv=cv, scoring='accuracy')
    tag = 'SUCCESS' if s.mean() > 0.70 else 'below'
    print('  %-22s: %.4f +/- %.4f  %s' % (nm, s.mean(), s.std(), tag))
    results_base[nm] = (s.mean(), s.std())

# ── Test 3: Pairwise delta on base 45 features ──────────────────────
print('\n[Pairwise delta features (fv1 - fv2), base 45]:')
delta45 = (X_a - X_b)[:, :BASE]
y_delta = labels_a  # 1 if seq1 is structured
mask_one = (labels_a + labels_b) == 1
print('  Pairs with exactly one structured seq: %d/%d' % (mask_one.sum(), len(meta)))

if mask_one.sum() >= 20:
    Xd = delta45[mask_one]
    yd = y_delta[mask_one]
    sc3 = StandardScaler()
    Xds = sc3.fit_transform(Xd)
    for nm, clf in [
        ('SVM-RBF', SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
        ('RF-100',  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]:
        s = cross_val_score(clf, Xds, yd, cv=5, scoring='accuracy')
        tag = 'SUCCESS' if s.mean() > 0.70 else 'below'
        print('  %-12s: %.4f +/- %.4f  %s' % (nm, s.mean(), s.std(), tag))

# ── Test 4: Ridge regression for DG ─────────────────────────────────
print('\n[DG regression on base 45 features (sanity check)]:')
dg = np.array([m['dg1'] for m in meta] + [m['dg2'] for m in meta])
r2s = cross_val_score(Ridge(alpha=1.0), Xs_base, dg, cv=cv, scoring='r2')
print('  Ridge R2: %.4f +/- %.4f' % (r2s.mean(), r2s.std()))

# ── Save revised results ─────────────────────────────────────────────
best_nm  = max(results_base, key=lambda k: results_base[k][0])
best_acc = results_base[best_nm][0]
best_std = results_base[best_nm][1]

import json as _j
revised = {
    'experiment':          '1C_revised',
    'n_pairs':             len(meta),
    'total_seqs':          2 * len(meta),
    'n_structured':        int((labels_a + labels_b).sum()),
    'n_unstructured':      int(2*len(meta) - (labels_a+labels_b).sum()),
    'feature_dim_used':    BASE,
    'task':                'binary: has_secondary_structure (base 45 features only)',
    'circuit_version':     'v2_ry_cx_ry',
    'best_classifier':     best_nm,
    'best_accuracy_mean':  float(best_acc),
    'best_accuracy_std':   float(best_std),
    'success':             bool(best_acc > 0.70),
    'classifiers':         {k: {'mean': v[0], 'std': v[1]} for k, v in results_base.items()},
    'dg_r2_base45':        float(r2s.mean()),
}
with open('D:/Qubis_HiQ/results_exp1c/exp1c_revised_results.json', 'w') as f:
    _j.dump(revised, f, indent=2)
print('\nRevised results saved.')
print('Best accuracy (base 45): %.4f (target >0.70: %s)' % (
    best_acc, 'MET' if best_acc > 0.70 else 'NOT MET'))
