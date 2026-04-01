"""Tests for encoding module."""
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qubis_hiq.encoding import apply_encoding_layer, encode_nucleotide


class TestEncodeNucleotide(unittest.TestCase):
    """Test cases for encode_nucleotide function."""

    def test_all_nucleotides(self):
        """Test encoding of all four nucleotides."""
        test_cases = [
            ("A", "A"),
            ("T", "T"),
            ("G", "G"),
            ("C", "C"),
            ("a", "A"),  # lowercase
            ("t", "T"),
            ("g", "G"),
            ("c", "C"),
        ]
        
        for input_nuc, expected in test_cases:
            result = encode_nucleotide(input_nuc)
            self.assertEqual(result, expected)

    def test_invalid_nucleotide(self):
        """Test that invalid nucleotides raise ValueError."""
        with self.assertRaises(ValueError):
            encode_nucleotide("X")
        with self.assertRaises(ValueError):
            encode_nucleotide("")


class TestApplyEncodingLayer(unittest.TestCase):
    """Test cases for apply_encoding_layer function."""

    def test_basic_encoding(self):
        """Test basic encoding layer application."""
        sequence = "ATGC"
        qc = QuantumCircuit(8)
        
        apply_encoding_layer(qc, sequence)
        
        # Should have added gates
        self.assertGreater(len(qc.data), 0)

    def test_encoding_deterministic(self):
        """Test that encoding is deterministic for same sequence."""
        sequence = "ATGC"
        
        qc1 = QuantumCircuit(8)
        apply_encoding_layer(qc1, sequence)
        
        qc2 = QuantumCircuit(8)
        apply_encoding_layer(qc2, sequence)
        
        # Same sequence should produce same circuit
        self.assertEqual(qc1.qasm(), qc2.qasm())

    def test_encoding_different_sequences(self):
        """Test that different sequences produce different encodings."""
        qc1 = QuantumCircuit(8)
        apply_encoding_layer(qc1, "ATGC")
        
        qc2 = QuantumCircuit(8)
        apply_encoding_layer(qc2, "CGTA")
        
        # Different sequences should produce different circuits
        self.assertNotEqual(qc1.qasm(), qc2.qasm())


if __name__ == "__main__":
    unittest.main()
