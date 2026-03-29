"""ViennaRNA interface for secondary structure prediction.
Falls back to simple heuristic if ViennaRNA not installed."""
from typing import List, Tuple, Optional
import warnings

def predict_structure(sequence: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Predict RNA secondary structure and return dot-bracket + base pairs.
    
    Returns:
        (dot_bracket_string, list_of_base_pair_tuples)
    """
    try:
        import RNA
        seq = sequence.upper().replace("T", "U")
        structure, mfe = RNA.fold(seq)
        pairs = _parse_dot_bracket(structure)
        return structure, pairs
    except ImportError:
        warnings.warn("ViennaRNA not installed. Using heuristic pairing.")
        return _heuristic_structure(sequence)

def _parse_dot_bracket(structure: str) -> List[Tuple[int, int]]:
    """Parse dot-bracket notation into base pair list."""
    pairs = []
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return sorted(pairs)

def _heuristic_structure(sequence: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Simple palindromic stem-loop heuristic for testing without ViennaRNA."""
    seq = sequence.upper()
    N = len(seq)
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "U": "A"}
    pairs = []
    struct = list('.' * N)
    
    # Try to pair from ends inward
    left, right = 0, N - 1
    while left < right - 2:  # Leave at least 3-nt loop
        if seq[left] in comp and comp[seq[left]] == seq[right]:
            pairs.append((left, right))
            struct[left] = '('
            struct[right] = ')'
            left += 1
            right -= 1
        else:
            break
    
    return ''.join(struct), pairs
