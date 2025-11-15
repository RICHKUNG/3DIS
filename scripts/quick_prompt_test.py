#!/usr/bin/env python3
"""
Quick test of different text prompts using existing GT test results.

This directly modifies the text prompts from our baseline test
to see if different templates improve ranking.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_baseline_results(json_file: Path):
    """Load baseline results with raw labels."""
    with open(json_file, 'r') as f:
        return json.load(f)


def extract_text_features(model, tokenizer, texts, device="cuda"):
    """Extract L2-normalized text features."""
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu()


def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    model_name = "google/siglip-so400m-patch14-384"
    logger.info(f"Loading SigLIP model: {model_name}")
    model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model.eval()
    
    # Load baseline results (no softmax for fair comparison)
    baseline_file = Path("/tmp/test_20samples_no_softmax.json")
    logger.info(f"Loading baseline results from {baseline_file}")
    baseline = load_baseline_results(baseline_file)
    
    # Extract unique labels
    labels = list(set(d['text'] for d in baseline['details'] if d['text'] != "UNKNOWN"))
    logger.info(f"Found {len(labels)} unique labels: {labels}")
    
    # Define templates to test
    templates = [
        ("raw", "{}"),
        ("a_noun", "a {}"),
        ("photo_of", "a photo of a {}"),
        ("indoor", "{} in an indoor scene"),
        ("indoor_short", "indoor {}"),
        ("in_room", "a {} in a room"),
    ]
    
    # Test each template
    print("\n" + "="*100)
    print("TEXT PROMPT TEMPLATE TEST RESULTS (Self-Similarity)")
    print("="*100)
    
    results = {}
    for template_name, template_format in templates:
        # Apply template to all labels
        templated_texts = [template_format.format(label) for label in labels]
        
        # Extract features
        template_feats = extract_text_features(model, tokenizer, templated_texts, device)
        
        # Get raw features for comparison
        raw_feats = extract_text_features(model, tokenizer, labels, device)
        
        # Compute similarity to raw labels
        similarities = (template_feats @ raw_feats.T).numpy()
        
        # Extract diagonal (self-similarities)
        self_sims = np.diag(similarities)
        
        results[template_name] = {
            'mean': float(np.mean(self_sims)),
            'std': float(np.std(self_sims)),
            'min': float(np.min(self_sims)),
            'max': float(np.max(self_sims)),
            'sims': self_sims.tolist(),
        }
        
        logger.info(f"{template_name:15s} ({template_format:35s}): "
                   f"mean={results[template_name]['mean']:.4f}, "
                   f"min={results[template_name]['min']:.4f}, "
                   f"max={results[template_name]['max']:.4f}")
    
    # Print detailed table
    print(f"\n{'Template':<15} {'Format':<35} {'Mean':<10} {'Min':<10} {'Max':<10} {'Change %':<12}")
    print("-"*100)
    
    raw_mean = results['raw']['mean']
    for template_name, template_format in templates:
        res = results[template_name]
        change_pct = ((res['mean'] - raw_mean) / raw_mean * 100) if raw_mean != 0 else 0
        print(f"{template_name:<15} {template_format:<35} {res['mean']:<10.4f} {res['min']:<10.4f} {res['max']:<10.4f} {change_pct:<12.2f}%")
    
    # Per-label breakdown
    print("\n" + "="*100)
    print("PER-LABEL BREAKDOWN")
    print("="*100)
    
    for idx, label in enumerate(labels):
        print(f"\nLabel: '{label}'")
        for template_name, template_format in templates:
            sim = results[template_name]['sims'][idx]
            templated = template_format.format(label)
            print(f"  {template_name:15s}: {sim:.4f}  ('{templated}')")
    
    # Recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)
    
    best_template = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"\nBest template (highest mean similarity to raw labels):")
    print(f"  Template: {best_template[0]}")
    print(f"  Mean similarity: {best_template[1]['mean']:.4f}")
    print(f"\nConclusion:")
    if best_template[0] == 'raw':
        print("  ✓ Use RAW labels (no template) - they have highest self-similarity")
        print("  ✗ Adding ANY template reduces similarity to raw labels")
    else:
        print(f"  ! Use '{best_template[0]}' template if consistency with original labels is not critical")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
