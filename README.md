# Quantum interferometric detection of non-nearest-neighbor effects in DNA thermodynamics

## Overview

QuBiS-HiQ is a physics-informed quantum circuit that encodes SantaLucia nearest-neighbour DNA thermodynamic parameters into gate angles, generating interferometric correlators that provably exceed classical product-state bounds (up to 358% mutual information gain).

## Installation

```bash
pip install -r requirements.txt
```

ViennaRNA (Exp 1C only):
```bash
conda install -c bioconda viennarna
```

## Reproducing All Experiments

### Exp 1A — ΔG° Regression (65,536 8-mers)

```bash
python experiments/exp1a_parallel.py --n-seqs 65536 --n-cpus 64
# Expected: R² = 0.868, CV R² = 0.764 ± 0.055, MAE = 0.626 kcal/mol
```

### Exp 1B — Five-way Ablation (500 sequences)

```bash
python experiments/exp1b_parallel.py --n-seqs 500
# Expected: Full=0.813, Random=−0.147 (0.96-unit drop)
```

### Exp 1C — Structural Classification (176 pairs)

```bash
python experiments/exp1c_parallel.py
python experiments/analyze_exp1c.py
# Expected: 100% ± 0.0% accuracy (all 3 classifiers)
```

### Exp 1D — IBM Hardware Validation

```bash
# Requires IBM Quantum Plan access
python experiments/exp1d_hardware.py   # ibm_fez,    30 seqs → 0.9970 ± 0.0005
python experiments/exp1d_torino.py    # ibm_torino, 30 seqs → 0.9948 ± 0.0015
python experiments/exp1d_12mer.py     # 12-mer, 24 qubits  → 0.9926 ± 0.0013
```

### Exp 1E — Experimental Tm Validation (Oliveira 2020)

```bash
python experiments/exp1e_corrected.py
# Expected: R² = 0.88, r = 0.94, p = 2.67×10⁻³⁰, MAE = 0.60°C
# Dataset:  64 canonical DNA duplexes, 38 qubits (MPS simulation)
# Scaffold: 5'-CGACGTGC[NNN]ATGTGCTG-3' (19 nt, page 8275 of Oliveira et al.)
# Buffer:   50 mM NaCl, 10 mM sodium phosphate, pH 7.4, 1.0 µM total strand
```

### Classical ML Baseline

```bash
python experiments/classical_baseline.py
# Expected: SVR-RBF Rich R² = 0.993, SVR-RBF Minimal R² = 0.995
```

### Proposition Proofs

```bash
python proofs/proposition1.py   # 17 assertions, ε < 3.4×10⁻¹⁶
python proofs/proposition2.py   # Uniqueness proof
python proofs/proposition3.py   # 358% info gain for AA/TT
```

## Data Sources

- **Oliveira et al. 2020**: *Chem. Sci.* 11, 8273–8287. [DOI: 10.1039/d0sc01700k](https://doi.org/10.1039/d0sc01700k) (Open Access, CC-BY)
- **SantaLucia & Hicks 2004**: *Annu. Rev. Biophys. Biomol. Struct.* 33, 415–440
  - **IBM Quantum Job IDs**: See Supplementary Table S4
   
    ## Hardware

| System | Device | Job ID |
|---|---|---|
| ibm_fez | Heron r2, 156 qubits | `d6qe32i0q0ls73cs7ah0` |
| ibm_torino | Heron r1, 133 qubits | `d6qeljropkic73fhv7rg` |
| 12-mer baseline | — | `d6qeh8nr88ds73dca350` |
| 12-mer mitigated | — | `d6qehkbopkic73fhv1o0` |
   
    - ## License
   
    Academic and Open-Source Use:
QuBiS-HiQ is released under the GNU General Public License v3.0 (GPLv3) to support open science and full academic reproducibility. This allows researchers to freely use, modify, and distribute the code under the condition that any derivative works are also open-sourced under the exact same GPLv3 terms. See the LICENSE file for full details.

Commercial Licensing:
The GPLv3 license requires that any proprietary software incorporating QuBiS-HiQ must also be open-sourced. If you represent a commercial entity (e.g., a biotech or pharmaceutical company) and wish to integrate QuBiS-HiQ into proprietary, closed-source products or internal commercial pipelines without the copyleft obligations of the GPLv3, a separate Commercial License is required.

For licensing inquiries, please contact the author directly at: ahmedaneesm@gmail.com; +91 90290 34496
