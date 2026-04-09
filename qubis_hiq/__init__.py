"""QuBiS-HiQ: physics-informed quantum circuit for DNA thermodynamics."""
from .santalucia import compute_total_dg, get_nn_dg, boltzmann_sigmoid
from .encoding import apply_encoding_layer, encode_nucleotide
from .circuit_builder import build_circuit, build_ablation_variants
from .feature_extraction import extract_feature_vector, feature_vector_dim
from .vienna_interface import predict_structure
from .topology_gated import build_topology_gated_circuit, classify_topology, batch_classify

__all__ = [
    "compute_total_dg",
    "get_nn_dg",
    "boltzmann_sigmoid",
    "apply_encoding_layer",
    "encode_nucleotide",
    "build_circuit",
    "build_ablation_variants",
    "extract_feature_vector",
    "feature_vector_dim",
    "predict_structure",
    "build_topology_gated_circuit",
    "classify_topology",
    "batch_classify",
]
