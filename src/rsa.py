#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Representational Similarity Analysis (RSA) Module

Implements RSA for comparing representational geometries between LLM and EEG.
"""

import numpy as np
from typing import Tuple, Dict, Any, Literal
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def compute_rsm(
    features: np.ndarray,
    method: Literal["cosine", "pearson"] = "cosine"
) -> np.ndarray:
    """
    Compute Representational Similarity Matrix (RSM).
    
    Args:
        features: Shape (n_samples, n_features)
        method: "cosine" for LLM (Formula 4), "pearson" for EEG (Formula 5)
    
    Returns:
        RSM matrix, shape (n_samples, n_samples)
    """
    if method == "cosine":
        # Formula (4): LLM RSM - cosine similarity
        rsm = cosine_similarity(features)
    elif method == "pearson":
        # Formula (5): EEG RSM - Pearson correlation
        rsm = np.corrcoef(features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return rsm


def compute_llm_rsm(llm_features: np.ndarray) -> np.ndarray:
    """Compute LLM RSM using cosine similarity (Formula 4)."""
    return compute_rsm(llm_features, method="cosine")


def compute_eeg_rsm(eeg_data: np.ndarray) -> np.ndarray:
    """Compute EEG RSM using Pearson correlation (Formula 5)."""
    return compute_rsm(eeg_data, method="pearson")


def compute_rsa(
    rsm1: np.ndarray,
    rsm2: np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman"
) -> Tuple[float, float]:
    """
    Compute RSA score (second-order similarity) between two RSMs.
    
    Formula (6): Spearman correlation on upper triangular elements.
    
    Args:
        rsm1: First RSM (LLM), shape (n, n)
        rsm2: Second RSM (EEG), shape (n, n)
        method: "spearman" (default) or "pearson"
    
    Returns:
        Tuple of (correlation, p_value)
    """
    if rsm1.shape != rsm2.shape:
        raise ValueError(f"RSM shapes must match: {rsm1.shape} vs {rsm2.shape}")
    
    n = rsm1.shape[0]
    
    # Formula (6): Upper triangular elements, excluding diagonal
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_1 = rsm1[upper_tri_indices]
    upper_tri_2 = rsm2[upper_tri_indices]
    
    if method == "spearman":
        corr, p_value = spearmanr(upper_tri_1, upper_tri_2)
    elif method == "pearson":
        corr, p_value = pearsonr(upper_tri_1, upper_tri_2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr, p_value


class RSACalculator:
    """RSA calculator for comparing LLM and EEG representations."""
    
    def __init__(
        self,
        llm_method: Literal["cosine"] = "cosine",
        eeg_method: Literal["pearson"] = "pearson",
        rsa_method: Literal["spearman", "pearson"] = "spearman"
    ):
        self.llm_method = llm_method
        self.eeg_method = eeg_method
        self.rsa_method = rsa_method
    
    def compute(
        self,
        llm_features: np.ndarray,
        eeg_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute RSA score between LLM and EEG representations.
        
        Formula (5): e_i is constructed by concatenating signals 
        from all channels corresponding to word i.
        
        Args:
            llm_features: Shape (n_words, n_hidden_dim)
            eeg_data: Shape (n_words, n_features) where n_features = n_channels * n_timepoints
                      (concatenated EEG signals per word)
        
        Returns:
            Dict with rsa_score, p_value, and RSM matrices
        """
        if llm_features.shape[0] != eeg_data.shape[0]:
            raise ValueError(
                f"Sample count mismatch: LLM {llm_features.shape[0]} vs EEG {eeg_data.shape[0]}"
            )
        
        rsm_llm = compute_rsm(llm_features, method=self.llm_method)
        rsm_eeg = compute_rsm(eeg_data, method=self.eeg_method)
        rsa_score, p_value = compute_rsa(rsm_llm, rsm_eeg, method=self.rsa_method)
        
        return {
            "rsa_score": rsa_score,
            "p_value": p_value,
            "rsm_llm": rsm_llm,
            "rsm_eeg": rsm_eeg,
            "n_words": llm_features.shape[0],
            "llm_dim": llm_features.shape[1],
            "eeg_channels": eeg_data.shape[1],
        }


def compute_layerwise_rsa(
    llm_features_by_layer: Dict[int, np.ndarray],
    eeg_data: np.ndarray,
    exclude_embedding_layer: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Compute RSA scores for all layers of an LLM.
    
    Args:
        llm_features_by_layer: Dict mapping layer index to features
        eeg_data: Shape (n_words, n_channels)
        exclude_embedding_layer: Skip layer 0
    
    Returns:
        Dict mapping layer index to RSA results
    """
    calculator = RSACalculator()
    results = {}
    
    for layer_idx, features in llm_features_by_layer.items():
        if exclude_embedding_layer and layer_idx == 0:
            continue
        
        try:
            layer_results = calculator.compute(features, eeg_data)
            layer_results["layer"] = layer_idx
            results[layer_idx] = layer_results
        except Exception as e:
            results[layer_idx] = {
                "layer": layer_idx,
                "error": str(e),
                "rsa_score": np.nan,
                "p_value": np.nan
            }
    
    return results

def concatenate_eeg_channels(eeg_data: np.ndarray) -> np.ndarray:
    """
    Prepare EEG data for RSM computation.
    
    Args:
        eeg_data: Shape (n_words, n_channels) or (n_words, n_channels, n_timepoints)
    
    Returns:
        Shape (n_words, n_features)
    """
    if eeg_data.ndim == 3:
        n_words, n_channels, n_timepoints = eeg_data.shape
        return eeg_data.reshape(n_words, n_channels * n_timepoints)
    return eeg_data

