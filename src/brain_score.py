#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brain-Score Calculation Module

Implements Brain-Score metric using predictive mapping with Ridge regression.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import warnings


def compute_brain_score(
    llm_features: np.ndarray,
    eeg_responses: np.ndarray,
    pca_components: int = 500,
    cv_folds: int = 10,
    alpha_range: Optional[np.ndarray] = None,
    random_state: int = 42
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Brain-Score between LLM representations and EEG responses.
    
    Implements Formulas (1)-(3) from the paper:
    - PCA dimensionality reduction: k = min(500, n, d)
    - Ridge regression: α ∈ [10^-3, 10^3]
    - 10-fold cross-validation with Pearson correlation
    
    Args:
        llm_features: Shape (n_samples, n_features)
        eeg_responses: Shape (n_samples,)
        pca_components: Maximum PCA components (default: 500)
        cv_folds: Number of CV folds (default: 10)
        alpha_range: Ridge α search range
        random_state: Random seed
    
    Returns:
        Tuple of (brain_score, details_dict)
    """
    if alpha_range is None:
        alpha_range = np.logspace(-3, 3, 20)
    
    n_samples, n_features = llm_features.shape
    
    if n_samples < 10:
        return 0.0, {"error": "Insufficient samples", "n_samples": n_samples}
    
    actual_cv_folds = min(cv_folds, n_samples // 2)
    if actual_cv_folds < 2:
        return 0.0, {"error": "Cannot perform cross-validation", "n_samples": n_samples}
    
    kf = KFold(n_splits=actual_cv_folds, shuffle=True, random_state=random_state)
    
    fold_correlations = []
    fold_alphas = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(llm_features)):
        X_train, X_test = llm_features[train_idx], llm_features[test_idx]
        y_train, y_test = eeg_responses[train_idx], eeg_responses[test_idx]
        
        # Formula (1): PCA - k = min(500, n, d)
        k = min(pca_components, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=k)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Formula (2): Ridge with nested CV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nested_cv = min(3, len(X_train_pca))
            ridge = RidgeCV(alphas=alpha_range, cv=nested_cv)
            ridge.fit(X_train_pca, y_train)
            y_pred = ridge.predict(X_test_pca)
        
        # Formula (3): Pearson correlation
        if len(y_test) > 1 and np.std(y_test) > 1e-8 and np.std(y_pred) > 1e-8:
            corr, _ = pearsonr(y_test, y_pred)
            if not np.isnan(corr):
                fold_correlations.append(corr)
                fold_alphas.append(ridge.alpha_)
    
    if not fold_correlations:
        return 0.0, {"error": "All folds failed", "cv_folds": actual_cv_folds}
    
    brain_score = np.mean(fold_correlations)
    
    details = {
        "fold_correlations": fold_correlations,
        "fold_alphas": fold_alphas,
        "mean_correlation": brain_score,
        "std_correlation": np.std(fold_correlations),
        "n_folds": len(fold_correlations),
        "pca_components_used": k,
    }
    
    return brain_score, details


class BrainScoreCalculator:
    """Brain-Score calculator for multi-electrode EEG data."""
    
    def __init__(
        self,
        pca_components: int = 500,
        cv_folds: int = 10,
        alpha_range: Optional[np.ndarray] = None,
        random_state: int = 42
    ):
        self.pca_components = pca_components
        self.cv_folds = cv_folds
        self.alpha_range = alpha_range if alpha_range is not None else np.logspace(-3, 3, 20)
        self.random_state = random_state
    
    def compute_single_electrode(
        self,
        llm_features: np.ndarray,
        eeg_responses: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute Brain-Score for a single electrode."""
        return compute_brain_score(
            llm_features=llm_features,
            eeg_responses=eeg_responses,
            pca_components=self.pca_components,
            cv_folds=self.cv_folds,
            alpha_range=self.alpha_range,
            random_state=self.random_state
        )
    
    def compute_all_electrodes(
        self,
        llm_features: np.ndarray,
        eeg_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Brain-Score averaged across all electrodes (32 channels).
        
        Args:
            llm_features: Shape (n_samples, n_features)
            eeg_data: Shape (n_samples, n_electrodes)
        
        Returns:
            Dictionary with mean_brain_score and electrode_scores
        """
        n_samples, n_electrodes = eeg_data.shape
        
        electrode_scores = []
        electrode_details = []
        
        for electrode_idx in range(n_electrodes):
            eeg_responses = eeg_data[:, electrode_idx]
            score, details = self.compute_single_electrode(llm_features, eeg_responses)
            electrode_scores.append(score)
            electrode_details.append(details)
        
        valid_scores = [s for s in electrode_scores if s != 0.0]
        
        if not valid_scores:
            return {
                "mean_brain_score": 0.0,
                "std_brain_score": 0.0,
                "electrode_scores": electrode_scores,
                "n_valid_electrodes": 0,
                "error": "No valid electrode scores"
            }
        
        return {
            "mean_brain_score": np.mean(valid_scores),
            "std_brain_score": np.std(valid_scores),
            "electrode_scores": electrode_scores,
            "n_valid_electrodes": len(valid_scores),
            "n_total_electrodes": n_electrodes,
        }


def compute_layerwise_brain_scores(
    llm_features_by_layer: Dict[int, np.ndarray],
    eeg_data: np.ndarray,
    pca_components: int = 500,
    cv_folds: int = 10,
    exclude_embedding_layer: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Compute Brain-Scores for all layers of an LLM.
    
    Args:
        llm_features_by_layer: Dict mapping layer index to features
        eeg_data: Shape (n_samples, n_electrodes)
        pca_components: Maximum PCA components (default: 500)
        cv_folds: Number of CV folds (default: 10)
        exclude_embedding_layer: Skip layer 0
    
    Returns:
        Dict mapping layer index to Brain-Score results
    """
    calculator = BrainScoreCalculator(
        pca_components=pca_components,
        cv_folds=cv_folds
    )
    
    results = {}
    
    for layer_idx, features in llm_features_by_layer.items():
        if exclude_embedding_layer and layer_idx == 0:
            continue
        
        layer_results = calculator.compute_all_electrodes(features, eeg_data)
        layer_results["layer"] = layer_idx
        results[layer_idx] = layer_results
    
    return results
