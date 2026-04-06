# Quantum interferometric detection of non-nearest-neighbour effects in DNA thermodynamics

## Overview

QuBiS-HiQ is a physics-informed quantum circuit that encodes SantaLucia nearest-neighbour DNA thermodynamic parameters into gate angles, generating interferometric correlators that provably exceed classical product-state bounds (up to 358% mutual information gain).

The circuit operates by encoding each nucleotide onto two qubits via Ry rotations calibrated to the SantaLucia free energy table, then entangling adjacent qubits with CX + Ry(θ) gates where θ is set by the Boltzmann-sigmoid of the nearest-neighbour ΔG°. The resulting measurement statistics exhibit three-way correlations among nucleotide identity, stacking energy, and secondary structure information that are inaccessible to classical product-state feature vectors. The approach is validated on 65,536 synthetic 8-mers (Exp 1A), a five-way ablation study (Exp 1B), structural classification of 176 sequence pairs (Exp 1C), real IBM quantum hardware across two processors (Exp 1D), and 64 experimental melting temperatures from Oliveira et al. 2020 (Exp 1E), achieving R² = 0.88, r = 0.94, and MAE = 0.60°C on the latter.

![Pipeline Architecture](pipeline_architecture.png)

*Figure 1: Overview of the QuBiS-HiQ computational pipeline. Input DNA sequences are first processed classically via ViennaRNA to establish thermodynamic baselines. The sequences and structural information are then encoded into the QuBiS-HiQ circuit, simulated on local simulators (HPC) or executed on IBM hardware. Z-basis measurements are processed to extract high-dimensional, physics-interpretable feature vectors for downstream machine learning tasks.*


---

## Repository Structure

```
QuBiS-HiQ/
├── qubis_hiq/              # Core library
│   ├── santalucia.py       # SantaLucia NN parameters + Boltzmann-sigmoid mapping
│   ├── encoding.py         # Layer 1: deterministic Ry nucleotide encoding
│   ├── watson_crick.py     # Layer 2: CRZ Watson-Crick complementarity gates
│   ├── stacking.py         # Layer 3: CX+Ry nearest-neighbour stacking gates
│   ├── trainable.py        # Layer 4: shared trainable local rotations
│   ├── circuit_builder.py  # Full 6-layer circuit assembly + ablation variants
│   └── feature_extraction.py  # ⟨Z⟩, ⟨ZZ⟩NN, ⟨ZZ⟩NNN correlator extraction
├── experiments/            # Reproducible experiment scripts (Exp 1A–1E + diagnostics)
├── proofs/                 # Executable mathematical verification scripts (Propositions 1–3)
│                           # NOTE: These are computational verifications, NOT formal
│                           # proof-assistant artifacts (e.g., Coq, Isabelle, Lean)
├── tests/                  # Test suite (24 tests covering core functionality)
├── .github/workflows/      # CI/CD pipeline (GitHub Actions)
├── data/                   # Oliveira 2020 dataset + SantaLucia parameters
├── results/                # Pre-computed JSON result files for all experiments
├── requirements.txt        # Minimum version requirements
├── requirements-pinned.txt # Pinned dependencies for exact reproducibility
└── setup.py                # Python package setup
```

---

## Installation

### Standard Installation
```bash
pip install -r requirements.txt
```

### Reproducible Installation (Pinned Dependencies)
For exact reproducibility, use the pinned dependency file:
```bash
pip install -r requirements-pinned.txt
```

### Development Installation
```bash
pip install -e .
pip install -r requirements-pinned.txt
```

ViennaRNA (Exp 1C only):
```bash
conda install -c bioconda viennarna
```

**Requirements:** Python ≥ 3.10, Qiskit ≥ 2.3, Qiskit-Aer ≥ 0.15. IBM Quantum account required only for Exp 1D hardware runs.

---

## Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py -v

# Or use pytest
pytest tests/
```

**Test Coverage:**
- `test_circuit_builder.py` - Circuit building and API correctness
- `test_encoding.py` - Nucleotide encoding verification
- `test_propositions.py` - Mathematical proposition verification

---

## Reproducing All Experiments

### Exp 1A - ΔG° Regression (65,536 8-mers)

```bash
python experiments/exp1a_parallel.py --n-seqs 65536 --n-cpus 64
# Expected: R² = 0.868, CV R² = 0.764 ± 0.055, MAE = 0.626 kcal/mol
```

### Exp 1B - Five-way Ablation (500 sequences)

```bash
python experiments/exp1b_parallel.py --n-seqs 500
# Expected: Full=0.813, Random=−0.147 (0.96-unit drop)
```

### Exp 1C - Structural Classification (176 pairs)

```bash
python experiments/exp1c_parallel.py
python experiments/analyze_exp1c.py
# Expected: 100% ± 0.0% accuracy (all 3 classifiers)
```

### Exp 1D - IBM Hardware Validation

```bash
# Requires IBM Quantum Plan access
python experiments/exp1d_hardware.py   # ibm_fez,    30 seqs → 0.9970 ± 0.0005
python experiments/exp1d_torino.py    # ibm_torino, 30 seqs → 0.9948 ± 0.0015
python experiments/exp1d_12mer.py     # 12-mer, 24 qubits  → 0.9926 ± 0.0013
```

### Exp 1E - Experimental Tm Validation (Oliveira 2020)

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

### Proposition Verification Scripts

**Note:** These are executable mathematical verification scripts that numerically verify the mathematical claims through exhaustive enumeration and statevector simulation. They are NOT formal proof-assistant artifacts (e.g., Coq, Isabelle, Lean).

```bash
python proofs/proposition1.py   # 17 assertions, ε < 3.4×10⁻¹⁶
python proofs/proposition2.py   # Uniqueness verification
python proofs/proposition3.py   # 358% info gain for AA/TT
```

---

## Quantum Kernel Diagnostic Experiments

These three experiments characterise the statistical properties of the quantum kernel, quantify whether the performance advantage is attributable to the quantum computation itself (vs. the SantaLucia physics encoding), and measure the contribution of each entangling layer.

### Kernel Condition Number Analysis

```bash
python experiments/kernel_condition_analysis.py
# Outputs: results/kernel_condition_results.json
```

Computes the condition number κ(K) = σ_max / σ_min for two kernels:

| Kernel | n | κ | log₁₀κ | Assessment |
|---|---|---|---|---|
| Exact quantum kernel (8-mer statevector) | 50 | **23** | 1.37 | ✅ Well-conditioned |
| Feature-vector linear kernel (Oliveira 19-nt, MPS) | 64 | **1.6 × 10⁷** | 7.20 | ⚠️ Poorly conditioned |

The 8-mer quantum kernel is full-rank and well-conditioned (all 50 eigenvalues positive, min eigenvalue 0.33). The Oliveira 19-nt feature kernel grows as κ ∝ n^1.4, reaching 1.6×10⁷ at n=64. This confirms that **Ridge regression regularisation is essential** for the full Oliveira dataset — vanilla SVM would be numerically unstable.

### Physics-Informed Classical Baseline

```bash
python experiments/classical_physics_baseline.py
# Outputs: results/classical_physics_baseline_results.json
```

Compares pure SantaLucia physics feature sets against QuBiS-HiQ for Oliveira 2020 Tm prediction (LOO-CV Ridge, same protocol as Exp 1E):

| Method | Features | LOO R² | MAE |
|---|---|---|---|
| Total ΔG° only | 1-d | 0.841 | 0.74°C |
| Per-step ΔG° | 18-d | 0.876 | 0.65°C |
| Variable-region features | 17-d | **0.935** | **0.43°C** |
| Rich physics | 41-d | 0.929 | 0.46°C |
| Rich physics + polynomial degree-2 | 989-d | 0.929 | 0.46°C |
| **QuBiS-HiQ quantum (variable region)** | 21-d | **0.880** | **0.60°C** |
| QuBiS-HiQ quantum (full features) | 111-d | 0.719 | 1.00°C |

**Interpretation:** Classical physics features derived directly from SantaLucia parameters (GC count, boundary ΔG° steps, one-hot centre encoding) achieve R²=0.935, exceeding the quantum headline result of R²=0.880 by 5.5 percentage points. This establishes that the predictive signal for Tm resides primarily in the **physics encoding** (SantaLucia parameter selection and feature construction), not in quantum computational effects such as entanglement or interference. This is consistent with the entanglement ablation results below.

### Entanglement Ablation Study

```bash
python experiments/entanglement_ablation.py
# Outputs: results/entanglement_ablation_results.json
```

Tests five circuit variants on the Oliveira 2020 Tm prediction task (MPS simulation, 38 qubits, LOO-CV Ridge):

| Circuit Variant | LOO R² | ΔR² vs full |
|---|---|---|
| Full (Encoding + WC + Stacking) | 0.776 | — |
| No Watson-Crick layer | **0.853** | +0.077 |
| No Stacking layer | 0.747 | −0.029 |
| Encoding only (no entanglement) | 0.561 | −0.214 |
| Random-angle CX (physics-uninformed) | −0.315 | −1.090 |

**Key findings:**

1. **Entanglement is necessary:** Removing all CX gates (encoding-only) drops R² by 0.21, confirming entangling gates carry genuine predictive information beyond local Ry rotations.
2. **Physics-informed structure is critical:** Replacing SantaLucia-derived rotation angles with random values collapses performance to R²=−0.315 — a 1.09 R² drop — demonstrating that the specific Boltzmann-sigmoid angle schedule is essential, not arbitrary entanglement.
3. **Watson-Crick layer on linear duplexes:** Removing the Watson-Crick CRZ layer *improves* performance for the Oliveira linear duplex scaffold (+0.077 R²). ViennaRNA predictions for these sequences may generate spurious hairpin stem pairs that add noise rather than signal in the duplex Tm prediction context.
4. **Stacking layer contribution is modest:** The CX+Ry stacking gates contribute +0.03 R² in isolation; most entanglement benefit comes from the combined WC+stacking interaction.

---

## Key Results Summary

| Experiment | Metric | Value |
|---|---|---|
| Exp 1A - ΔG° regression (65,536 8-mers) | CV R² | 0.764 ± 0.055 |
| Exp 1B - Ablation: full vs random | R² drop | 0.813 → -0.147 |
| Exp 1C - Structural classification | Accuracy | 100% ± 0.0% |
| Exp 1D - IBM ibm_fez hardware (30 seqs) | Cosine similarity | 0.9970 ± 0.0005 |
| Exp 1D - IBM ibm_torino cross-platform | Cosine similarity | 0.9948 ± 0.0015 |
| Exp 1E - Experimental Tm (Oliveira 2020) | R² / r / MAE | 0.88 / 0.94 / 0.60°C |
| Kernel condition (8-mer exact) | κ | 23 (well-conditioned) |
| Kernel condition (Oliveira, 64 seqs) | κ | 1.6×10⁷ (regularise) |
| Classical physics baseline (best) | LOO R² | 0.935 (variable-region 17-d) |
| Entanglement contribution | ΔR² | +0.21 (vs encoding-only) |

---

## Data Sources

- **Oliveira et al. 2020**: *Chem. Sci.* 11, 8273–8287. [DOI: 10.1039/d0sc01700k](https://doi.org/10.1039/d0sc01700k) (Open Access, CC-BY)
- **SantaLucia & Hicks 2004**: *Annu. Rev. Biophys. Biomol. Struct.* 33, 415–440
- **IBM Quantum Job IDs**: See Supplementary Table S4

## Hardware

| System | Device | Job ID |
|---|---|---|
| ibm_fez | Heron r2, 156 qubits | `d6qe32i0q0ls73cs7ah0` |
| ibm_torino | Heron r1, 133 qubits | `d6qeljropkic73fhv7rg` |
| 12-mer baseline | - | `d6qeh8nr88ds73dca350` |
| 12-mer mitigated | - | `d6qehkbopkic73fhv1o0` |

---

## License

**Academic and Open-Source Use:**
QuBiS-HiQ is released under the GNU General Public License v3.0 (GPLv3) to support open science and full academic reproducibility. This allows researchers to freely use, modify, and distribute the code under the condition that any derivative works are also open-sourced under the exact same GPLv3 terms. Please see the [LICENSE](LICENSE) file for full details.

**Commercial Licensing:**
The GPLv3 license requires that any proprietary software incorporating QuBiS-HiQ must also be open-sourced. If you represent a commercial entity (e.g., a biotech or pharmaceutical company) and wish to integrate QuBiS-HiQ into proprietary, closed-source products or internal commercial pipelines without the copyleft obligations of the GPLv3, a separate Commercial License is required.

---

## Contact & Enquiries

| Type | Details |
|---|---|
| **Scientific enquiries** | Open a [GitHub Issue](https://github.com/ahmedanees-m/QuBiS-HiQ/issues) |
| **Other** | ahmedaneesm@gmail.com · +91 90290 34496 |

Bugs, reproducibility issues, and questions about the methodology are all welcome via GitHub Issues.
