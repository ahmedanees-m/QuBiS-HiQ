"""Tests for encoding module."""
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qubis_hiq.encoding import apply_encoding_layer, encode_nucleotide, hamming_distance, is_transition, NUC_MAP


class TestEncodeNucleotide(unittest.TestCase):
    """Test cases for encode_nucleotide function."""

    def test_all_nucleotides(self):
        """Test encoding of all four nucleotides."""
        # encode_nucleotide applies gates to a circuit; verify it doesn't raise
        for nuc in ["A", "T", "G", "C", "a", "t", "g", "c"]:
            qc = QuantumCircuit(2)
            try:
                encode_nucleotide(qc, nuc, 0, 1)
                # Should have added 2 Ry gates
                self.assertGreater(len(qc.data), 0)
            except Exception as e:
                self.fail(f"encode_nucleotide raised {e} for nucleotide {nuc}")

    def test_invalid_nucleotide(self):
        """Test that invalid nucleotides raise ValueError."""
        qc = QuantumCircuit(2)
        with self.assertRaises((ValueError, KeyError)):
            encode_nucleotide(qc, "X", 0, 1)


class TestApplyEncodingLayer(unittest.TestCase):
    """Test cases for apply_encoding_layer function."""

    def test_basic_encoding(self):
        """Test basic encoding layer application."""
        sequence = "ATGC"
        qc = QuantumCircuit(8)
        
        apply_encoding_layer(qc, sequence)
        
        # Should have added gates (2 Ry gates per nucleotide)
        self.assertGreater(len(qc.data), 0)

    def test_encoding_deterministic(self):
        """Test that encoding is deterministic for same sequence."""
        sequence = "ATGC"
        
        qc1 = QuantumCircuit(8)
        apply_encoding_layer(qc1, sequence)
        
        qc2 = QuantumCircuit(8)
        apply_encoding_layer(qc2, sequence)
        
        # Same sequence should produce same number of gates
        self.assertEqual(len(qc1.data), len(qc2.data))
        # And same gate counts by type
        from collections import Counter
        gates1 = Counter(type(inst.operation).__name__ for inst in qc1.data)
        gates2 = Counter(type(inst.operation).__name__ for inst in qc2.data)
        self.assertEqual(gates1, gates2)

    def test_encoding_different_sequences(self):
        """Test that different sequences produce different encodings."""
        qc1 = QuantumCircuit(8)
        apply_encoding_layer(qc1, "ATGC")
        
        qc2 = QuantumCircuit(8)
        apply_encoding_layer(qc2, "CGTA")
        
        # Different sequences should produce different parameter values in Ry gates
        # Extract Ry rotation angles
        def get_ry_params(qc):
            params = []
            for inst in qc.data:
                if type(inst.operation).__name__ == 'RYGate':
                    params.append(float(inst.operation.params[0]))
            return params
        
        params1 = get_ry_params(qc1)
        params2 = get_ry_params(qc2)
        
        # ATGC vs CGTA should have different Ry parameters
        self.assertNotEqual(params1, params2)


class TestHammingDistance(unittest.TestCase):
    """Test cases for hamming_distance function."""

    def test_transitions_distance_1(self):
        """Test that transitions (A↔G, T↔C) have Hamming distance 1."""
        self.assertEqual(hamming_distance("A", "G"), 1)
        self.assertEqual(hamming_distance("G", "A"), 1)
        self.assertEqual(hamming_distance("T", "C"), 1)
        self.assertEqual(hamming_distance("C", "T"), 1)

    def test_watson_crick_distance_1(self):
        """Test Watson-Crick complements have Hamming distance 1."""
        self.assertEqual(hamming_distance("A", "T"), 1)
        self.assertEqual(hamming_distance("T", "A"), 1)
        self.assertEqual(hamming_distance("G", "C"), 1)
        self.assertEqual(hamming_distance("C", "G"), 1)

    def test_transversions_distance_2(self):
        """Test that transversions have Hamming distance 2."""
        self.assertEqual(hamming_distance("A", "C"), 2)
        self.assertEqual(hamming_distance("C", "A"), 2)
        self.assertEqual(hamming_distance("G", "T"), 2)
        self.assertEqual(hamming_distance("T", "G"), 2)

    def test_same_nucleotide_distance_0(self):
        """Test same nucleotide has Hamming distance 0."""
        for nuc in ["A", "T", "G", "C"]:
            self.assertEqual(hamming_distance(nuc, nuc), 0)


class TestIsTransition(unittest.TestCase):
    """Test cases for is_transition function."""

    def test_purine_transitions(self):
        """Test A↔G transitions."""
        self.assertTrue(is_transition("A", "G"))
        self.assertTrue(is_transition("G", "A"))

    def test_pyrimidine_transitions(self):
        """Test T↔C transitions."""
        self.assertTrue(is_transition("T", "C"))
        self.assertTrue(is_transition("C", "T"))

    def test_transversions(self):
        """Test that transversions return False."""
        self.assertFalse(is_transition("A", "C"))
        self.assertFalse(is_transition("C", "A"))
        self.assertFalse(is_transition("G", "T"))
        self.assertFalse(is_transition("T", "G"))

    def test_watson_crick_pairs(self):
        """Test Watson-Crick pairs return False."""
        self.assertFalse(is_transition("A", "T"))
        self.assertFalse(is_transition("T", "A"))
        self.assertFalse(is_transition("G", "C"))
        self.assertFalse(is_transition("C", "G"))


class TestNUCMap(unittest.TestCase):
    """Test cases for NUC_MAP constant."""

    def test_nuc_map_values(self):
        """Test that NUC_MAP has correct bit encodings."""
        self.assertEqual(NUC_MAP["A"], (0, 0))  # 00
        self.assertEqual(NUC_MAP["T"], (0, 1))  # 01
        self.assertEqual(NUC_MAP["U"], (0, 1))  # 01 (same as T)
        self.assertEqual(NUC_MAP["G"], (1, 0))  # 10
        self.assertEqual(NUC_MAP["C"], (1, 1))  # 11


if __name__ == "__main__":
    unittest.main()
