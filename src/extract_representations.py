#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Representation Extraction Module

Extracts hidden states from HuggingFace transformer models.
Uses last subword token for word-level representations (Qwen/DeepSeek style).
Excludes word embedding layer (layer 0) by default.

Note: Different model families may use different token alignment strategies.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
import warnings


def get_word_boundaries(
    text: str,
    tokenizer
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Get token indices for each word in the text.
    
    Args:
        text: Input text string
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Tuple of (word_boundaries, offset_mapping)
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    
    offset_mapping = encoding["offset_mapping"][0].tolist()
    
    words = text.split()
    word_boundaries = []
    
    current_char = 0
    for word in words:
        while current_char < len(text) and text[current_char].isspace():
            current_char += 1
        
        start_char = current_char
        end_char = start_char + len(word)
        word_boundaries.append((start_char, end_char))
        current_char = end_char
    
    return word_boundaries, offset_mapping


def get_last_subword_indices(
    word_boundaries: List[Tuple[int, int]],
    offset_mapping: List[Tuple[int, int]]
) -> List[int]:
    """
    Find the last subword token index for each word.
    
    Args:
        word_boundaries: List of (start_char, end_char) per word
        offset_mapping: Token offset mapping from tokenizer
    
    Returns:
        List of token indices (one per word)
    """
    last_subword_indices = []
    
    for word_start, word_end in word_boundaries:
        last_token_idx = None
        
        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == 0 and token_end == 0:
                continue
            
            if token_start < word_end and token_end > word_start:
                last_token_idx = token_idx
        
        if last_token_idx is not None:
            last_subword_indices.append(last_token_idx)
    
    return last_subword_indices


def extract_llm_representations(
    text: str,
    model_name: str,
    layers: Optional[List[int]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    exclude_embedding_layer: bool = True
) -> Dict[int, np.ndarray]:
    """
    Extract word-level hidden states from an LLM.
    
    Args:
        text: Input text to process
        model_name: HuggingFace model name
        layers: Layer indices to extract (None = all)
        device: "cuda" or "cpu"
        exclude_embedding_layer: Skip layer 0
    
    Returns:
        Dict mapping layer index to representations, shape (n_words, hidden_dim)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    word_boundaries, offset_mapping = get_word_boundaries(text, tokenizer)
    last_subword_indices = get_last_subword_indices(word_boundaries, offset_mapping)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states)
    
    if layers is None:
        layers = list(range(n_layers))
    
    if exclude_embedding_layer:
        layers = [l for l in layers if l > 0]
    
    representations = {}
    
    for layer_idx in layers:
        if layer_idx >= n_layers:
            warnings.warn(f"Layer {layer_idx} does not exist (max: {n_layers-1})")
            continue
        
        layer_hidden = hidden_states[layer_idx][0]
        
        word_representations = []
        for token_idx in last_subword_indices:
            if token_idx < layer_hidden.shape[0]:
                word_representations.append(layer_hidden[token_idx].cpu().numpy())
        
        if word_representations:
            representations[layer_idx] = np.stack(word_representations, axis=0)
    
    return representations


def extract_batch_representations(
    texts: List[str],
    model_name: str,
    layers: Optional[List[int]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    exclude_embedding_layer: bool = True
) -> List[Dict[int, np.ndarray]]:
    """
    Extract representations for multiple texts.
    
    Args:
        texts: List of input texts
        model_name: HuggingFace model name
        layers: Layer indices to extract (None = all)
        device: "cuda" or "cpu"
        exclude_embedding_layer: Skip layer 0
    
    Returns:
        List of dicts, one per text
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    all_representations = []
    
    for text in texts:
        word_boundaries, offset_mapping = get_word_boundaries(text, tokenizer)
        last_subword_indices = get_last_subword_indices(word_boundaries, offset_mapping)
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)
        
        if layers is None:
            extract_layers = list(range(n_layers))
        else:
            extract_layers = layers
        
        if exclude_embedding_layer:
            extract_layers = [l for l in extract_layers if l > 0]
        
        representations = {}
        
        for layer_idx in extract_layers:
            if layer_idx >= n_layers:
                continue
            
            layer_hidden = hidden_states[layer_idx][0]
            
            word_representations = []
            for token_idx in last_subword_indices:
                if token_idx < layer_hidden.shape[0]:
                    word_representations.append(layer_hidden[token_idx].cpu().numpy())
            
            if word_representations:
                representations[layer_idx] = np.stack(word_representations, axis=0)
        
        all_representations.append(representations)
    
    return all_representations


def get_model_layer_info(model_name: str) -> Dict[str, int]:
    """
    Get model layer information.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        Dict with n_layers, hidden_dim, n_attention_heads, model_type
    """
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    config = model.config
    
    return {
        "n_layers": getattr(config, "num_hidden_layers", None),
        "hidden_dim": getattr(config, "hidden_size", None),
        "n_attention_heads": getattr(config, "num_attention_heads", None),
        "model_type": getattr(config, "model_type", None),
    }
