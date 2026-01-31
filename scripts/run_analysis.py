#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis Script for Brain-Score and RSA

Usage:
    python scripts/run_analysis.py --model Qwen2.5-0.5B --article 0
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_score import BrainScoreCalculator
from rsa import RSACalculator
from utils import (
    load_eeg_data,
    load_layerwise_representations,
    align_samples,
    save_results,
    DERCO_CONFIG
)


def load_model_config(config_path: str = None) -> dict:
    """Load model configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def analyze_single_model(
    model_name: str,
    article_id: int,
    eeg_dir: str,
    llm_dir: str,
    output_dir: str,
    n_layers: int,
    verbose: bool = True
) -> dict:
    """
    Run Brain-Score and RSA analysis for a single model.
    
    Args:
        model_name: Name of the LLM model
        article_id: DERCo article ID (0-4)
        eeg_dir: Directory containing EEG data
        llm_dir: Directory containing LLM representations
        output_dir: Directory to save results
        n_layers: Number of layers in the model
        verbose: Print progress
    
    Returns:
        Results dictionary
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_name} - Article {article_id}")
        print(f"{'='*60}")
    
    # Load EEG data
    eeg_file = os.path.join(eeg_dir, f"article_{article_id}_average.fif")
    
    if verbose:
        print(f"\n[1/4] Loading EEG data...")
    
    try:
        # Brain-Score: time-averaged (per-electrode computation)
        eeg_data_bs = load_eeg_data(eeg_file, time_window=DERCO_CONFIG['word_window'], mode="mean")
        # RSA: concatenated channels*timepoints (Formula 5)
        eeg_data_rsa = load_eeg_data(eeg_file, time_window=DERCO_CONFIG['word_window'], mode="concatenate")
        if verbose:
            print(f"      Brain-Score EEG shape: {eeg_data_bs.shape}")
            print(f"      RSA EEG shape: {eeg_data_rsa.shape}")
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    # Load LLM representations
    model_dir = os.path.join(llm_dir, model_name)
    
    if verbose:
        print(f"\n[2/4] Loading LLM representations...")
    
    try:
        llm_layers = load_layerwise_representations(
            model_dir=model_dir,
            article_id=article_id,
            n_layers=n_layers,
            exclude_embedding_layer=True
        )
        if verbose:
            print(f"      Loaded {len(llm_layers)} layers")
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    # Compute Brain-Score
    if verbose:
        print(f"\n[3/4] Computing Brain-Score...")
    
    brain_score_results = {}
    calculator = BrainScoreCalculator(pca_components=500, cv_folds=10)
    
    for layer_idx, llm_features in llm_layers.items():
        llm_aligned, eeg_aligned = align_samples(llm_features, eeg_data_bs)
        result = calculator.compute_all_electrodes(llm_aligned, eeg_aligned)
        result['layer'] = layer_idx
        brain_score_results[layer_idx] = result
        
        if verbose:
            print(f"      Layer {layer_idx:2d}: {result['mean_brain_score']:.4f}")
    
    # Compute RSA
    if verbose:
        print(f"\n[4/4] Computing RSA...")
    
    rsa_results = {}
    rsa_calculator = RSACalculator()
    
    for layer_idx, llm_features in llm_layers.items():
        llm_aligned, eeg_aligned = align_samples(llm_features, eeg_data_rsa)
        result = rsa_calculator.compute(llm_aligned, eeg_aligned)
        result['layer'] = layer_idx
        result.pop('rsm_llm', None)
        result.pop('rsm_eeg', None)
        rsa_results[layer_idx] = result
        
        if verbose:
            print(f"      Layer {layer_idx:2d}: {result['rsa_score']:.4f} (p={result['p_value']:.2e})")
    
    # Compile results
    results = {
        'model_name': model_name,
        'article_id': article_id,
        'n_layers': n_layers,
        'brain_score': brain_score_results,
        'rsa': rsa_results,
    }
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_article_{article_id}_results.joblib")
        save_results(results, output_file)
        if verbose:
            print(f"\nSaved to: {output_file}")
    
    # Summary
    if verbose:
        bs_scores = [r['mean_brain_score'] for r in brain_score_results.values()]
        rsa_scores = [r['rsa_score'] for r in rsa_results.values()]
        
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"  Brain-Score: mean={np.mean(bs_scores):.4f}, max={np.max(bs_scores):.4f}")
        print(f"  RSA:         mean={np.mean(rsa_scores):.4f}, max={np.max(rsa_scores):.4f}")
        print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Brain-Score and RSA analysis")
    
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--article', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--eeg-dir', type=str, default=None)
    parser.add_argument('--llm-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    try:
        config = load_model_config(args.config)
        models_config = config.get('models', {})
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        models_config = {}
    
    if args.model in models_config:
        n_layers = models_config[args.model].get('layers', 32)
    else:
        print(f"Warning: Model {args.model} not in config, using n_layers=32")
        n_layers = 32
    
    eeg_dir = args.eeg_dir or config.get('paths', {}).get('eeg_dir', './data/eeg')
    llm_dir = args.llm_dir or config.get('paths', {}).get('llm_dir', './data/representations')
    
    results = analyze_single_model(
        model_name=args.model,
        article_id=args.article,
        eeg_dir=eeg_dir,
        llm_dir=llm_dir,
        output_dir=args.output_dir,
        n_layers=n_layers,
        verbose=not args.quiet
    )
    
    if 'error' in results:
        print(f"\nAnalysis failed: {results['error']}")
        sys.exit(1)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
