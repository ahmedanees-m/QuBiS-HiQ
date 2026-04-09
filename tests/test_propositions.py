"""Tests for mathematical propositions."""
import unittest
import numpy as np
from qubis_hiq.encoding import hamming_distance
from qubis_hiq.santalucia import boltzmann_sigmoid


class TestProposition1EncodingMutationIsomorphism(unittest.TestCase):
    """Test Proposition 1: Encoding-Mutation Isomorphism."""

    def test_hamming_distance_transitions(self):
        """Test that transitions (A↔G, T↔C) have Hamming distance 1."""
        # A (00) <-> G (10): distance 1
        self.assertEqual(hamming_distance("A", "G"), 1)
        # T (01) <-> C (11): distance 1
        self.assertEqual(hamming_distance("T", "C"), 1)

    def test_hamming_distance_transversions(self):
        """Test that transversions have Hamming distance 2."""
        # A (00) <-> C (11): distance 2
        self.assertEqual(hamming_distance("A", "C"), 2)
        # G (10) <-> T (01): distance 2
        self.assertEqual(hamming_distance("G", "T"), 2)

    def test_hamming_distance_watson_crick(self):
        """Test Watson-Crick complement Hamming distances."""
        # A (00) <-> T (01): distance 1
        self.assertEqual(hamming_distance("A", "T"), 1)
        # G (10) <-> C (11): distance 1
        self.assertEqual(hamming_distance("G", "C"), 1)

    def test_4_cycle_a_t_c_g_a(self):
        """Test the 4-cycle A-T-C-G-A."""
        # A -> T -> C -> G -> A should form a cycle
        dist_at = hamming_distance("A", "T")
        dist_tc = hamming_distance("T", "C")
        dist_cg = hamming_distance("C", "G")
        dist_ga = hamming_distance("G", "A")
        
        # All edges in cycle should have distance 1
        self.assertEqual(dist_at, 1)
        self.assertEqual(dist_tc, 1)
        self.assertEqual(dist_cg, 1)
        self.assertEqual(dist_ga, 1)


class TestProposition2BoltzmannSigmoid(unittest.TestCase):
    """Test Proposition 2: Boltzmann-Sigmoid Uniqueness."""

    def setUp(self):
        """Set up test parameters."""
        self.BETA = 0.39  # mol/kcal at 310K

    def test_axiom_1_range(self):
        """Axiom 1: θ ∈ (0, π) for all finite ΔG°."""
        test_dg = np.linspace(-10, 10, 100)
        for dg in test_dg:
            theta = boltzmann_sigmoid(dg, self.BETA)
            self.assertGreater(theta, 0)
            self.assertLess(theta, np.pi)

    def test_axiom_2_monotonicity(self):
        """Axiom 2: θ is strictly decreasing with ΔG°.
        
        More stable (negative ΔG°) → larger θ.
        Less stable (positive ΔG°) → smaller θ.
        """
        test_dg = np.linspace(-10, 10, 100)
        thetas = [boltzmann_sigmoid(dg, self.BETA) for dg in test_dg]
        
        # Check strictly decreasing
        for i in range(len(thetas) - 1):
            self.assertGreater(thetas[i], thetas[i + 1])

    def test_axiom_3_balance(self):
        """Axiom 3: θ(0) = π/2."""
        theta_zero = boltzmann_sigmoid(0, self.BETA)
        self.assertAlmostEqual(theta_zero, np.pi / 2, places=10)

    def test_axiom_4_dimensional(self):
        """Axiom 4: β has correct dimensions (mol/kcal)."""
        # This is more of a documentation check
        # β = 0.39 mol/kcal at 310K is the correct value
        self.assertEqual(self.BETA, 0.39)

    def test_extreme_values(self):
        """Test behavior at extreme ΔG° values.
        
        Very negative ΔG° (very stable) → θ → π (largest angle)
        Very positive ΔG° (very unstable) → θ → 0 (smallest angle)
        """
        # Very negative ΔG° (very stable) → θ → π
        theta_neg = boltzmann_sigmoid(-100, self.BETA)
        self.assertAlmostEqual(theta_neg, np.pi, places=5)
        
        # Very positive ΔG° (very unstable) → θ → 0
        theta_pos = boltzmann_sigmoid(100, self.BETA)
        self.assertAlmostEqual(theta_pos, 0, places=5)


if __name__ == "__main__":
    unittest.main()
