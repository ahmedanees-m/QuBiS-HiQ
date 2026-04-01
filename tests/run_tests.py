#!/usr/bin/env python
"""Test runner for QuBiS-HiQ.

Usage:
    python tests/run_tests.py          # Run all tests
    python tests/run_tests.py -v       # Run with verbose output
    python -m pytest tests/            # Alternative using pytest
"""
import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_tests(verbosity=1):
    """Run all tests in the tests directory."""
    # Discover all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    verbosity = 2 if '-v' in sys.argv or '--verbose' in sys.argv else 1
    sys.exit(run_tests(verbosity))
