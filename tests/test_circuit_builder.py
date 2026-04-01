"""Tests for circuit_builder module."""
import unittest
import numpy as np
from qiskit.circuit import ParameterVector
from qubis_hiq.circuit_builder import build_circuit


class TestBuildCircuit(unittest.TestCase):
    """Test cases for build_circuit function."""

    def test_basic_circuit_creation(self):
        """Test basic circuit creation with a simple sequence."""
        sequence = "ATGC"
        qc = build_circuit(sequence)
        
        self.assertEqual(qc.num_qubits, 8)  # 2 qubits per nucleotide
        self.assertGreater(len(qc.data), 0)  # Circuit has gates

    def test_trainable_params_none_creates_parameterized_circuit(self):
        """Test that trainable_params=None creates a parameterized circuit.
        
        This is the key fix - the documentation says None = parameterized,
        and now the implementation matches.
        """
        sequence = "ATGC"
        qc = build_circuit(sequence, trainable_params=None)
        
        # Check that the circuit has parameters
        params = list(qc.parameters)
        self.assertGreater(len(params), 0, 
                          "Parameterized circuit should have parameters")
        
        # Check it's a ParameterVector
        param_names = [p.name for p in params]
        self.assertTrue(any("theta_train" in name for name in param_names),
                       "Should have theta_train parameters")

    def test_trainable_params_array_creates_bound_circuit(self):
        """Test that passing an array creates a bound circuit."""
        sequence = "ATGC"
        params = np.zeros(12)
        qc = build_circuit(sequence, trainable_params=params)
        
        # Check that the circuit has no parameters (all bound)
        self.assertEqual(len(list(qc.parameters)), 0,
                        "Bound circuit should have no free parameters")

    def test_skip_wc_layer(self):
        """Test skipping Watson-Crick layer."""
        sequence = "ATGC"
        stem_pairs = [(0, 3), (1, 2)]
        
        qc_with_wc = build_circuit(sequence, stem_pairs, np.zeros(12))
        qc_without_wc = build_circuit(sequence, stem_pairs, np.zeros(12), skip_wc=True)
        
        # Without WC layer should have fewer gates
        self.assertLess(len(qc_without_wc.data), len(qc_with_wc.data))

    def test_skip_stacking_layer(self):
        """Test skipping stacking layer."""
        sequence = "ATGC"
        
        qc_with_stack = build_circuit(sequence, trainable_params=np.zeros(12))
        qc_without_stack = build_circuit(sequence, trainable_params=np.zeros(12), 
                                         skip_stacking=True)
        
        # Without stacking should have fewer gates
        self.assertLess(len(qc_without_stack.data), len(qc_with_stack.data))

    def test_random_angles(self):
        """Test random angles mode."""
        sequence = "ATGC"
        
        qc1 = build_circuit(sequence, trainable_params=np.zeros(12), random_angles=True)
        qc2 = build_circuit(sequence, trainable_params=np.zeros(12), random_angles=True)
        
        # Two random circuits should be different
        self.assertNotEqual(qc1.qasm(), qc2.qasm())

    def test_include_measurement(self):
        """Test measurement inclusion."""
        sequence = "ATGC"
        
        qc_with_meas = build_circuit(sequence, trainable_params=np.zeros(12), 
                                     include_measurement=True)
        qc_without_meas = build_circuit(sequence, trainable_params=np.zeros(12), 
                                        include_measurement=False)
        
        # With measurement should have classical bits
        self.assertGreater(qc_with_meas.num_clbits, 0)
        self.assertEqual(qc_without_meas.num_clbits, 0)

    def test_different_sequence_lengths(self):
        """Test circuits with different sequence lengths."""
        for length in [4, 6, 8, 10]:
            sequence = "A" * length
            qc = build_circuit(sequence, trainable_params=np.zeros(12))
            self.assertEqual(qc.num_qubits, 2 * length)


class TestBuildAblationVariants(unittest.TestCase):
    """Test cases for build_ablation_variants function."""

    def test_returns_four_variants(self):
        """Test that four variants are returned."""
        from qubis_hiq.circuit_builder import build_ablation_variants
        
        sequence = "ATGC"
        stem_pairs = [(0, 3), (1, 2)]
        
        variants = build_ablation_variants(sequence, stem_pairs)
        
        self.assertEqual(len(variants), 4)
        self.assertIn("full", variants)
        self.assertIn("no_wc", variants)
        self.assertIn("no_stacking", variants)
        self.assertIn("random", variants)


if __name__ == "__main__":
    unittest.main()
