#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility Functions for Data Loading
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Union
from pathlib import Path
import joblib
import warnings

try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


def load_eeg_data(
    eeg_file: str,
    time_window: Tuple[float, float] = (0.0, 0.5),
    baseline_start: float = -0.05,
    mode: str = "mean"
) -> np.ndarray:
    """
    Load EEG data from MNE epochs file.
    
    Args:
        eeg_file: Path to .fif file
        time_window: Time window in seconds (default: 0-500ms)
        baseline_start: Epoch start time (default: -50ms)
        mode: "mean" for Brain-Score (average over time), 
              "concatenate" for RSA (concatenate channels*timepoints per Formula 5)
    
    Returns:
        EEG data, shape (n_words, n_channels) if mode="mean"
                  shape (n_words, n_channels * n_timepoints) if mode="concatenate"
    """
    if not HAS_MNE:
        raise ImportError("MNE required. Install with: pip install mne")
    
    if not os.path.exists(eeg_file):
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")
    
    epochs = mne.read_epochs(eeg_file, preload=True, verbose=False)
    eeg_data = epochs.get_data()
    
    sfreq = epochs.info['sfreq']
    start_sample = int((time_window[0] - baseline_start) * sfreq)
    end_sample = int((time_window[1] - baseline_start) * sfreq)
    
    start_sample = max(0, start_sample)
    end_sample = min(eeg_data.shape[2], end_sample)
    
    eeg_window = eeg_data[:, :, start_sample:end_sample]
    
    if mode == "concatenate":
        # Formula (5): e_i is constructed by concatenating signals 
        # from all channels corresponding to word i
        n_epochs, n_channels, n_times = eeg_window.shape
        return eeg_window.reshape(n_epochs, n_channels * n_times)
    else:
        # Brain-Score: average over time, compute per electrode
        return np.mean(eeg_window, axis=2)


def load_llm_representations(file_path: str) -> np.ndarray:
    """
    Load LLM representations from joblib file.
    
    Args:
        file_path: Path to .joblib file
    
    Returns:
        Representations array, shape (n_words, n_features)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return joblib.load(file_path)


def load_layerwise_representations(
    model_dir: str,
    article_id: int,
    n_layers: int,
    file_pattern: str = "layer_{layer}_original.joblib",
    exclude_embedding_layer: bool = True
) -> Dict[int, np.ndarray]:
    """
    Load representations for all layers of a model.
    
    Args:
        model_dir: Directory containing representations
        article_id: Article ID to load
        n_layers: Number of layers
        file_pattern: Pattern for layer files
        exclude_embedding_layer: Skip layer 0
    
    Returns:
        Dict mapping layer index to representations
    """
    article_dir = os.path.join(model_dir, f"article_{article_id}")
    
    if not os.path.exists(article_dir):
        raise FileNotFoundError(f"Article directory not found: {article_dir}")
    
    representations = {}
    start_layer = 1 if exclude_embedding_layer else 0
    
    for layer in range(start_layer, n_layers):
        file_name = file_pattern.format(layer=layer)
        file_path = os.path.join(article_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                representations[layer] = load_llm_representations(file_path)
            except Exception as e:
                warnings.warn(f"Failed to load layer {layer}: {e}")
    
    return representations


def align_samples(*arrays: np.ndarray, axis: int = 0) -> List[np.ndarray]:
    """
    Align arrays to minimum size along axis.
    
    Args:
        *arrays: Arrays to align
        axis: Axis to align (default: 0)
    
    Returns:
        List of aligned arrays
    """
    min_samples = min(arr.shape[axis] for arr in arrays)
    
    aligned = []
    for arr in arrays:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, min_samples)
        aligned.append(arr[tuple(slices)])
    
    return aligned


def save_results(results: Dict, output_path: str) -> None:
    """Save results to joblib file."""
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    joblib.dump(results, output_path)


# DERCo dataset configuration
DERCO_CONFIG = {
    'n_articles': 5,
    'n_electrodes': 32,
    'sampling_rate': 1000,
    'word_window': (0.0, 0.5),
}


def get_derco_article_info(article_id: int) -> Dict[str, Union[int, str]]:
    """Get DERCo article information."""
    if article_id < 0 or article_id >= DERCO_CONFIG['n_articles']:
        raise ValueError(f"Invalid article_id: {article_id}")
    
    return {
        'article_id': article_id,
        'n_electrodes': DERCO_CONFIG['n_electrodes'],
        'sampling_rate': DERCO_CONFIG['sampling_rate'],
        'time_window': DERCO_CONFIG['word_window'],
    }
