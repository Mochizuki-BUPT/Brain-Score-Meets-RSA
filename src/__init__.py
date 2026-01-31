"""
Brain-Score Meets RSA: Core Analysis Module

This package provides implementations for:
- Brain-Score calculation (PCA + Ridge regression + cross-validation)
- RSA calculation (RSM construction + Spearman correlation)
- LLM representation extraction

Reference: ICASSP 2026 - "Brain-Score Meets Representational Similarity Analysis"
"""

from .brain_score import BrainScoreCalculator, compute_brain_score
from .rsa import compute_rsm, compute_rsa, RSACalculator
from .extract_representations import extract_llm_representations
from .utils import load_eeg_data, load_llm_representations

__version__ = "1.0.0"
__all__ = [
    "BrainScoreCalculator",
    "compute_brain_score",
    "compute_rsm",
    "compute_rsa",
    "RSACalculator",
    "extract_llm_representations",
    "load_eeg_data",
    "load_llm_representations",
]
