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
│   ├── feature_extraction.py  # ⟨Z⟩, ⟨ZZ⟩NN, ⟨ZZ⟩NNN correlator extraction
│   ├── vienna_interface.py # ViennaRNA wrapper + palindromic heuristic fallback
│   ├── classical_twin.py   # Classical SantaLucia feature vectors (D2 baseline)
│   ├── interference.py     # Quantum interference analysis utilities
│   └── topology_gated.py   # Topology-aware WC layer gating (ViennaRNA or heuristic)
├── experiments/            # Reproducible experiment scripts (Exp 1A–1E, D1–D4, X1–X2)
├── proofs/                 # Executable mathematical verification scripts (Propositions 1–3)
│                           # NOTE: These are computational verifications, NOT formal
│                           # proof-assistant artifacts (e.g., Coq, Isabelle, Lean)
├── tests/                  # Test suite (32 tests covering core functionality)
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

These four experiments characterise the statistical properties of the quantum kernel, quantify whether the performance advantage is attributable to the quantum computation itself (vs. the SantaLucia physics encoding), measure the contribution of each entangling layer, and identify the optimal circuit-feature configuration.

### Experiment D1 — Kernel Condition Number Analysis

```bash
python experiments/kernel_condition_analysis.py
# Outputs: results/kernel_condition_results.json
```

Computes the condition number κ(K) = σ_max / σ_min for two kernels:

| Kernel | n | κ | log₁₀κ | Assessment |
|---|---|---|---|---|
| Exact quantum kernel (8-mer statevector) | 50 | **23** | 1.37 | ✅ Well-conditioned |
| Feature-vector linear kernel (Oliveira 19-nt, MPS) | 64 | **1.6 × 10⁷** | 7.20 | ⚠️ Poorly conditioned |

The 8-mer quantum kernel is full-rank and well-conditioned (κ=23, all 50 eigenvalues positive, min eigenvalue 0.33). The condition number grows as κ ∝ n^1.4 across subset sizes, reaching 1.6×10⁷ at n=64 for the Oliveira 19-nt sequences. **Ridge regression regularisation is therefore essential** for the full Oliveira dataset — the results confirm that the LOO-CV Ridge protocol used in Exp 1E is the correct approach.

### Experiment D2 — Physics-Informed Classical Baseline

```bash
python experiments/classical_physics_baseline.py
# Outputs: results/classical_physics_baseline_results.json
```

Compares pure SantaLucia physics feature sets against QuBiS-HiQ for Oliveira 2020 Tm prediction (LOO-CV Ridge, same protocol as Exp 1E):

| Method | Features | LOO R² | MAE |
|---|---|---|---|
| Total ΔG° only | 1-d | 0.841 | 0.74°C |
| Per-step ΔG° | 18-d | 0.876 | 0.65°C |
| Variable-region features | **17-d** | **0.935** | **0.43°C** |
| Rich physics | 41-d | 0.929 | 0.46°C |
| QuBiS-HiQ quantum (variable region, no-WC) | 21-d | 0.911 | 0.53°C |
| QuBiS-HiQ quantum (full features) | 111-d | 0.719 | 1.00°C |

**Interpretation:** A 17-dimensional classical feature vector (GC count, boundary ΔG° steps, one-hot NNN centre encoding) achieves R²=0.935, exceeding the quantum variable-region result of R²=0.911 by 2.4 percentage points. The predictive signal for Tm resides primarily in the **physics encoding** — specifically in the direct SantaLucia parameterisation of the variable NNN centre — rather than in entanglement or interference effects. See Exp D4 for the combined result.

### Experiment D3 — Entanglement Ablation Study

```bash
python experiments/entanglement_ablation.py
# Outputs: results/entanglement_ablation_results.json
```

Tests five circuit variants on the Oliveira 2020 Tm prediction task (MPS simulation, 38 qubits, LOO-CV Ridge). Results are shown at two feature-extraction levels:

**Full 111-d features (all Z correlators across 38 qubits):**

| Circuit Variant | LOO R² | ΔR² vs full |
|---|---|---|
| Full (Encoding + WC + Stacking) | 0.776 | — |
| No Watson-Crick layer | 0.853 | +0.077 |
| No Stacking layer | 0.747 | −0.029 |
| Encoding only (no entanglement) | 0.561 | −0.214 |
| Random-angle CX | −0.315 | −1.090 |

**Variable-region 21-d features (qubits 16–21 only):**

| Circuit Variant | LOO R² | ΔR² vs full |
|---|---|---|
| Full (Encoding + WC + Stacking) | 0.902 | — |
| No Watson-Crick layer | **0.912** | +0.009 |
| No Stacking layer | 0.910 | +0.008 |
| Encoding only (no entanglement) | 0.909 | +0.007 |
| Random-angle CX | −0.194 | −1.097 |

**Key findings:**

1. **Entanglement on full features:** Removing all CX gates drops LOO R² by 0.21 (full features). Physics-informed entanglement is essential for the full-circuit correlator representation.
2. **Variable-region convergence:** When features are restricted to the variable-region qubits (16–21), all circuit variants except random achieve nearly identical performance (R²=0.90–0.91). Local Ry encoding captures most variable-region Tm information independently of entanglement. The no-WC variant is marginally best (R²=0.912).
3. **Physics-informed structure is non-negotiable:** Random-angle CX collapses to R²=−0.31 (full) and −0.19 (variable-region) — a catastrophic failure confirming that the Boltzmann-sigmoid angle schedule is the essential ingredient, not arbitrary entanglement.
4. **Watson-Crick layer on linear duplexes:** Removing the Watson-Crick CRZ layer improves performance for the Oliveira linear duplex scaffold (+0.077 R² on full features). ViennaRNA-predicted hairpin stem pairs are not appropriate for this linear-duplex melting context and add noise rather than signal.

### Experiment D4 — Best Configuration Synthesis

```bash
python experiments/best_configuration.py
# Outputs: results/best_configuration_results.json
```

Tests the optimal circuit-feature configurations, including the combined quantum+classical feature set:

| Configuration | LOO R² | r | MAE |
|---|---|---|---|
| Quantum var-region (no-WC, 21-d) | 0.911 | 0.955 | 0.53°C |
| Classical var-region (17-d) | 0.935 | 0.968 | 0.43°C |
| **Quantum + Classical combined (38-d)** | **0.941** | **0.970** | **0.41°C** |
| Quantum full features (no-WC, 111-d) | 0.750 | 0.870 | 0.88°C |
| Kernel Ridge (linear kernel, no-WC) | 0.895 | 0.946 | 0.58°C |

**The combined quantum+classical feature set (38-d) achieves R²=0.941, the best result in the project** — a synergistic gain of +0.006 R² over the best classical-only baseline (0.935). While the individual quantum variable-region features fall 2.4 points short of classical, they encode complementary information that classical features do not capture, yielding the best overall prediction when fused. This supports positioning QuBiS-HiQ as a **physics-informed feature extractor that complements rather than replaces classical thermodynamic features**.

### Experiment X1 — Watson-Crick Topology Validation

```bash
python experiments/x1_topology_validation.py
# Outputs: results/x1_topology_results.json
#          results/x1_sequence_classification.json
```

Tests whether the Watson-Crick entangling layer improves performance on sequences where base-pairing physics is genuinely present (hairpin-forming) vs. absent (linear). Uses ViennaRNA for structure prediction and classifies 65,536 random 8-mers.

**Classification result:** ViennaRNA predicts secondary structure for only 58/65,536 (0.1%) of random 8-mers — consistent with the minimum loop constraint that prevents stable 8-mer hairpin formation. The remaining 99.9% are classified as linear (WC layer is analytically a no-op since `stem_pairs=[]`).

| Subset | n | Full R² | No-WC R² | ΔR² | Verdict |
|---|---|---|---|---|---|
| Hairpin-classified | 58 | 0.878 ± 0.044 | 0.889 ± 0.057 | +0.011 | neutral |
| Linear (no stem pairs) | 58 | 0.504 ± 0.125 | 0.512 ± 0.107 | +0.008 | neutral |
| All combined | 116 | 0.669 ± 0.192 | 0.670 ± 0.218 | +0.002 | neutral |

**Interpretation:** For ΔG° prediction on 8-mers, the WC layer has negligible impact regardless of sequence topology (|ΔR²| < 0.012, within standard deviation). This is physically correct: ΔG° is entirely determined by nearest-neighbour stacking interactions, not intramolecular base-pairing. The Watson-Crick layer encodes duplex-stability physics (relevant to Tm prediction) and therefore adds no signal to a stacking-only target. The definitive topology test for the WC layer requires sequences that form stable ViennaRNA-predicted hairpins (≥16 nt) with experimental Tm measurements — a natural extension to the current dataset.

A note on the `topology_gated` module: the new `qubis_hiq/topology_gated.py` provides `build_topology_gated_circuit()` for automatic topology-aware WC gating, ready for use when extending to longer RNA or DNA sequences where ViennaRNA hairpin predictions are meaningful.

### Experiment X2 — Stacking-Only Optimised Benchmark

```bash
python experiments/x2_stacking_only_benchmark.py
# Outputs: results/x2_stacking_only_results.json
```

Formalises the stacking-only circuit (`skip_wc=True`) as the primary model for linear duplex Tm prediction, with bootstrapped 95% confidence intervals.

| Configuration | LOO R² | 95% CI | r | MAE |
|---|---|---|---|---|
| Quantum full (no-WC, 111-d) | 0.840 | [0.770, 0.885] | 0.917 | 0.71°C |
| Quantum var-region (no-WC, 21-d) | 0.904 | [0.857, 0.937] | 0.951 | 0.56°C |
| Classical var-region (17-d) | 0.935 | [0.894, 0.962] | 0.968 | 0.43°C |
| **Combined quantum+classical (38-d)** | **0.941** | **[0.907, 0.965]** | **0.970** | **0.40°C** |
| KRR linear kernel (no-WC) | 0.893 | [0.841, 0.928] | 0.945 | 0.59°C |

**Best configuration: Combined quantum+classical (38-d), LOO R²=0.941 [0.907, 0.965], r=0.970, p=8.8×10⁻⁴⁰, MAE=0.40°C.** This is the headline result for the linear duplex Tm prediction task.

Note: All Oliveira 2020 sequences are linear duplexes. ViennaRNA/heuristic may predict 4–5 stem pairs per 19-nt scaffold sequence from intramolecular folding, but these are phantom predictions irrelevant to intermolecular hybridisation. The stacking-only circuit correctly ignores them.

---

## Key Results Summary

| Experiment | Metric | Value |
|---|---|---|
| Exp 1A — ΔG° regression (65,536 8-mers) | CV R² | 0.764 ± 0.055 |
| Exp 1B — Ablation: full vs random | R² drop | 0.813 → −0.147 |
| Exp 1C — Structural classification | Accuracy | 100% ± 0.0% |
| Exp 1D — IBM ibm_fez hardware (30 seqs) | Cosine similarity | 0.9970 ± 0.0005 |
| Exp 1D — IBM ibm_torino cross-platform | Cosine similarity | 0.9948 ± 0.0015 |
| Exp 1E — Experimental Tm (Oliveira 2020) | R² / r / MAE | 0.88 / 0.94 / 0.60°C |
| D1 — Kernel condition (8-mer, n=50) | κ | 23 (well-conditioned) |
| D1 — Kernel condition (Oliveira, n=64) | κ | 1.6×10⁷ (regularise) |
| D2 — Best classical baseline | LOO R² | 0.935 (variable-region 17-d) |
| D3 — Entanglement contribution (full feats) | ΔR² | +0.21 vs encoding-only |
| D3 — Watson-Crick layer (linear duplexes) | Effect | +0.077 R² (adds noise, skip_wc=True is better) |
| D4 — Best configuration | LOO R² | 0.941 (quantum + classical, 38-d) |
| X1 — WC layer on 8-mer hairpin subset | ΔR² | ≈0 (neutral; ΔG° is stacking-only target) |
| X2 — Stacking-only + bootstrap CI | LOO R² / CI | **0.941 [0.907, 0.965]** |

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
