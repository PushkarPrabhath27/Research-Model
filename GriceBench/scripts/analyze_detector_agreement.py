"""
Analyze Detector-Human Agreement
================================

Calculates Cohen's kappa between detector predictions and human annotations.
Per morechanges.md lines 1749-1844.

Outputs:
- Per-maxim kappa scores
- Overall agreement
- Classification report
- Interpretation

Author: GriceBench
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def calculate_cohen_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate Cohen's Kappa for binary classification.
    
    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lists must be same length")
    
    n = len(y_true)
    if n == 0:
        return 0.0
    
    # Observed agreement
    p_o = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    
    # Expected agreement
    true_pos = sum(y_true) / n
    true_neg = 1 - true_pos
    pred_pos = sum(y_pred) / n
    pred_neg = 1 - pred_pos
    
    p_e = (true_pos * pred_pos) + (true_neg * pred_neg)
    
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    
    return (p_o - p_e) / (1 - p_e)


def analyze_agreement(
    annotations_file: str,
    detector_predictions_file: str = None,
    output_file: str = "results/detector_human_agreement.json"
) -> Dict:
    """
    Calculate detector-human agreement.
    
    Args:
        annotations_file: Path to human annotations JSON
        detector_predictions_file: Path to detector predictions (optional if in annotations)
        output_file: Path to save results
    
    Returns:
        Agreement metrics dictionary
    """
    print("=" * 70)
    print("DETECTOR-HUMAN AGREEMENT ANALYSIS")
    print("=" * 70)
    
    # Load annotations
    print(f"\nLoading annotations from {annotations_file}...")
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"  Loaded {len(annotations)} annotations")
    
    # Load detector predictions if separate file
    if detector_predictions_file:
        with open(detector_predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        pred_dict = {p['id']: p for p in predictions}
    else:
        pred_dict = None
    
    # Calculate per-maxim agreement
    maxims = ['quantity', 'quality', 'relation', 'manner']
    results = {}
    
    for maxim in maxims:
        human_labels = []
        detector_labels = []
        
        for ann in annotations:
            # Get human label
            human_violations = ann.get('violations', {})
            human_label = 1 if human_violations.get(maxim, False) else 0
            
            # Get detector label
            if pred_dict:
                pred = pred_dict.get(ann['id'], {})
                detector_violations = pred.get('violations', {})
            else:
                detector_violations = ann.get('detector_violations', {})
            
            detector_label = 1 if detector_violations.get(maxim, False) else 0
            
            human_labels.append(human_label)
            detector_labels.append(detector_label)
        
        if not human_labels:
            continue
        
        # Calculate kappa
        kappa = calculate_cohen_kappa(human_labels, detector_labels)
        
        # Calculate accuracy, precision, recall, F1
        tp = sum(1 for h, d in zip(human_labels, detector_labels) if h == 1 and d == 1)
        tn = sum(1 for h, d in zip(human_labels, detector_labels) if h == 0 and d == 0)
        fp = sum(1 for h, d in zip(human_labels, detector_labels) if h == 0 and d == 1)
        fn = sum(1 for h, d in zip(human_labels, detector_labels) if h == 1 and d == 0)
        
        accuracy = (tp + tn) / len(human_labels) if human_labels else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[maxim] = {
            'kappa': round(kappa, 4),
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'confusion': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
        }
        
        print(f"\n{maxim.upper()}:")
        print(f"  Cohen's κ: {kappa:.3f}")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  F1:        {f1:.3f}")
    
    # Calculate overall kappa
    all_human = []
    all_detector = []
    
    for maxim in maxims:
        for ann in annotations:
            human_violations = ann.get('violations', {})
            human_label = 1 if human_violations.get(maxim, False) else 0
            
            if pred_dict:
                pred = pred_dict.get(ann['id'], {})
                detector_violations = pred.get('violations', {})
            else:
                detector_violations = ann.get('detector_violations', {})
            
            detector_label = 1 if detector_violations.get(maxim, False) else 0
            
            all_human.append(human_label)
            all_detector.append(detector_label)
    
    overall_kappa = calculate_cohen_kappa(all_human, all_detector)
    
    print(f"\n{'=' * 50}")
    print(f"OVERALL Cohen's κ: {overall_kappa:.3f}")
    
    # Interpretation
    if overall_kappa > 0.8:
        interpretation = "Excellent agreement - detector is valid"
        recommendation = "Continue with current detector"
    elif overall_kappa > 0.7:
        interpretation = "Good agreement - detector acceptable with minor issues"
        recommendation = "Consider targeted improvements"
    elif overall_kappa > 0.5:
        interpretation = "Moderate agreement - detector needs recalibration"
        recommendation = "Retrain detector on human annotations"
    else:
        interpretation = "Poor agreement - detector needs retraining"
        recommendation = "Major detector retraining required"
    
    print(f"Interpretation: {interpretation}")
    print(f"Recommendation: {recommendation}")
    
    # Save results
    output = {
        'per_maxim': results,
        'overall_kappa': round(overall_kappa, 4),
        'interpretation': interpretation,
        'recommendation': recommendation,
        'n_annotations': len(annotations)
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    return output


def main():
    """Run agreement analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze detector-human agreement")
    parser.add_argument("--annotations", default="data_processed/self_annotations.json",
                       help="Path to human annotations")
    parser.add_argument("--predictions", default=None,
                       help="Path to detector predictions (optional)")
    parser.add_argument("--output", default="results/detector_human_agreement.json",
                       help="Output path")
    
    args = parser.parse_args()
    
    analyze_agreement(args.annotations, args.predictions, args.output)


if __name__ == "__main__":
    main()
