"""Step 17 -- Proposition 1: Encoding-Mutation Isomorphism.

Formal claim:
  The 2-bit encoding A=00, T=01, G=10, C=11 defines a Hamming metric on
  nucleotide space such that:
    (a) All biochemically meaningful transitions (A<->G, T<->C) and
        Watson-Crick complements (A<->T, G<->C) have Hamming distance 1.
    (b) All purine-pyrimidine transversions (A<->C, G<->T) have Hamming
        distance 2.
    (c) The Hamming distance map is an ISOMORPHISM between the mutation
        graph on {A,T,G,C} and the Boolean hypercube B^2 = {00,01,10,11},
        preserving adjacency.

This script:
  1. Enumerates all 12 ordered single-nucleotide substitutions.
  2. Computes Hamming distances for each pair.
  3. Verifies the isomorphism claims with assertions.
  4. Prints the full 4x4 distance matrix.
  5. Prints a formal proof sketch.
"""
import numpy as np
import sys

# -- Encoding ------------------------------------------------------------------
ENCODING = {
    'A': (0, 0),   # 00
    'T': (0, 1),   # 01
    'G': (1, 0),   # 10
    'C': (1, 1),   # 11
}
NUCLEOTIDES = ['A', 'T', 'G', 'C']

def hamming(n1, n2):
    """Hamming distance between two nucleotides under the QuBiS-HiQ encoding."""
    b1 = ENCODING[n1]
    b2 = ENCODING[n2]
    return sum(x != y for x, y in zip(b1, b2))

# -- Biochemical classification ------------------------------------------------
TRANSITIONS    = {('A', 'G'), ('G', 'A'), ('T', 'C'), ('C', 'T')}
WC_COMPLEMENTS = {('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')}
TRANSVERSIONS  = {('A', 'C'), ('C', 'A'), ('G', 'T'), ('T', 'G')}

# -- Computation ---------------------------------------------------------------
print('=' * 65)
print('PROPOSITION 1: Encoding-Mutation Isomorphism')
print('=' * 65)

print('\nEncoding map:')
for idx, (nuc, bits) in enumerate(ENCODING.items()):
    print('  {} -> {}  (qubit pair q{}, q{})'.format(
        nuc, ''.join(str(b) for b in bits), 2*idx, 2*idx+1))

# Full 4x4 distance matrix
print('\n4x4 Hamming Distance Matrix (QuBiS-HiQ encoding):')
print('        ', end='')
for n2 in NUCLEOTIDES:
    print('{:>6}'.format(n2), end='')
print()
for n1 in NUCLEOTIDES:
    print('  {:>3}   '.format(n1), end='')
    for n2 in NUCLEOTIDES:
        print('{:>6}'.format(hamming(n1, n2)), end='')
    print()

# -- Verification --------------------------------------------------------------
print('\nVerification:')
errors = []

print('\n  (a) Transitions -- Hamming must = 1:')
for n1, n2 in sorted(TRANSITIONS):
    d = hamming(n1, n2)
    ok = (d == 1)
    print('      {} <-> {}  ({} <-> {})  d_H = {}  [{}]'.format(
        n1, n2,
        ''.join(str(b) for b in ENCODING[n1]),
        ''.join(str(b) for b in ENCODING[n2]),
        d, 'OK' if ok else 'FAIL'))
    if not ok:
        errors.append('Transition {} -> {} : d_H={} != 1'.format(n1, n2, d))

print('\n  (b) Watson-Crick complements -- Hamming must = 1:')
for n1, n2 in sorted(WC_COMPLEMENTS):
    d = hamming(n1, n2)
    ok = (d == 1)
    print('      {} <-> {}  ({} <-> {})  d_H = {}  [{}]'.format(
        n1, n2,
        ''.join(str(b) for b in ENCODING[n1]),
        ''.join(str(b) for b in ENCODING[n2]),
        d, 'OK' if ok else 'FAIL'))
    if not ok:
        errors.append('WC complement {} -> {} : d_H={} != 1'.format(n1, n2, d))

print('\n  (c) Purine-Pyrimidine transversions -- Hamming must = 2:')
for n1, n2 in sorted(TRANSVERSIONS):
    d = hamming(n1, n2)
    ok = (d == 2)
    print('      {} <-> {}  ({} <-> {})  d_H = {}  [{}]'.format(
        n1, n2,
        ''.join(str(b) for b in ENCODING[n1]),
        ''.join(str(b) for b in ENCODING[n2]),
        d, 'OK' if ok else 'FAIL'))
    if not ok:
        errors.append('Transversion {} -> {} : d_H={} != 2'.format(n1, n2, d))

print('\n  (d) Isomorphism with Boolean hypercube B^2:')
bio_adj = {}
for n1 in NUCLEOTIDES:
    bio_adj[n1] = [n2 for n2 in NUCLEOTIDES if n1 != n2 and hamming(n1, n2) == 1]

for nuc, neighbors in bio_adj.items():
    bits = ''.join(str(b) for b in ENCODING[nuc])
    neighbor_str = ', '.join(
        '{} ({})'.format(n, ''.join(str(b) for b in ENCODING[n]))
        for n in neighbors)
    print('      {} ({})  ->  neighbors: {}'.format(nuc, bits, neighbor_str))
    if len(neighbors) != 2:
        errors.append('{} has degree {} != 2 in mutation graph'.format(nuc, len(neighbors)))

# Verify 4-cycle A-T-C-G-A
cycle = [('A', 'T'), ('T', 'C'), ('C', 'G'), ('G', 'A')]
all_ok = all(hamming(a, b) == 1 for a, b in cycle)
print('\n      4-cycle A-T-C-G-A (all edges Hamming=1): {}'.format(
    'OK' if all_ok else 'FAIL'))
if not all_ok:
    errors.append('4-cycle A-T-C-G-A not fully Hamming-1')

# -- Qubit angle consequence ---------------------------------------------------
print('\n  (e) Qubit Bloch-sphere consequence:')
theta = {0: np.pi/3, 1: 2*np.pi/3}
Z_exp = {nuc: [np.cos(theta[b]) for b in ENCODING[nuc]] for nuc in NUCLEOTIDES}
print('      <Z_q0>, <Z_q1> for each nucleotide:')
for nuc in NUCLEOTIDES:
    print('        {} : <Z_q0>={:+.4f}  <Z_q1>={:+.4f}'.format(
        nuc, Z_exp[nuc][0], Z_exp[nuc][1]))

print('\n      Euclidean distance |<Z>_n1 - <Z>_n2| in feature space:')
print('        {:>6}'.format(''), end='')
for n2 in NUCLEOTIDES:
    print('{:>8}'.format(n2), end='')
print()
for n1 in NUCLEOTIDES:
    print('        {:>6}'.format(n1), end='')
    for n2 in NUCLEOTIDES:
        z1 = np.array(Z_exp[n1])
        z2 = np.array(Z_exp[n2])
        dist = np.linalg.norm(z1 - z2)
        print('{:>8.4f}'.format(dist), end='')
    print()
print('      Transitions & WC: dist = {:.4f}  |  Transversions: dist = {:.4f}'.format(
    np.linalg.norm(np.array(Z_exp['A']) - np.array(Z_exp['T'])),
    np.linalg.norm(np.array(Z_exp['A']) - np.array(Z_exp['C']))))

# -- Summary -------------------------------------------------------------------
print('\n' + '=' * 65)
if errors:
    print('PROPOSITION 1: FAILED')
    for e in errors:
        print('  ERROR:', e)
    sys.exit(1)
else:
    print('PROPOSITION 1: VERIFIED  (all {} assertions pass)'.format(
        4 + 4 + 4 + 4 + 1))  # transitions+WC+transversions+degree+cycle

print("""
Formal Proof Sketch
-------------------
Let phi: {{A,T,G,C}} -> {{0,1}}^2 be defined by
    phi(A)=00, phi(T)=01, phi(G)=10, phi(C)=11.

Let M be the mutation graph on {{A,T,G,C}} with edges for transitions
and Watson-Crick complementation. Let B^2 be the Boolean hypercube
with Hamming distance d_H.

Theorem (Encoding-Mutation Isomorphism):
  phi is an isomorphism of graphs M ~= (B^2, d_H=1).

Proof:
  (Bijection) phi maps 4 nucleotides to 4 distinct 2-bit strings. []

  (Transitions A<->G, T<->C) differ only in bit 0 (purine flag):
    phi(A)=00, phi(G)=10  =>  d_H=1.  phi(T)=01, phi(C)=11  =>  d_H=1.

  (WC complements A<->T, G<->C) differ only in bit 1 (pyrimidine flag):
    phi(A)=00, phi(T)=01  =>  d_H=1.  phi(G)=10, phi(C)=11  =>  d_H=1.

  (Transversions A<->C, G<->T) differ in both bits:
    phi(A)=00, phi(C)=11  =>  d_H=2.  phi(G)=10, phi(T)=01  =>  d_H=2.

  Therefore M is the 4-cycle A-T-C-G-A in B^2, confirming graph
  isomorphism. The map phi embeds biochemical mutation distance faithfully
  into the Hamming metric of the qubit register. QED.

Consequence for QuBiS-HiQ:
  Encoding Ry(pi/3) for bit=0, Ry(2pi/3) for bit=1 gives
  <Z_q> = cos(pi/3) = +0.5 for bit 0 and cos(2pi/3) = -0.5 for bit 1.
  A transition or WC substitution flips one qubit: delta<Z> = 1.0.
  A transversion flips both qubits: delta<Z> in BOTH observables.
  The quantum feature vector thus encodes mutational distance as
  Euclidean distance in Bloch-sphere expectation-value space.
""")
print('=' * 65)
