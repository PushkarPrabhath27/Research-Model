"""
GriceBench Gold Annotation Tools
================================

This module provides tools for creating and managing the gold annotation set:
1. Strategic example selection (balanced by violation type)
2. Export to Google Sheets format (CSV)
3. Inter-annotator agreement calculation (Cohen's Kappa)

Based on Chapter 6 of the GriceBench Implementation Guide.
"""

import json
import csv
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass


# ============================================================================
# EXAMPLE SELECTION
# ============================================================================

def select_annotation_examples(
    examples: List[Dict],
    total_count: int = 1000,
    random_seed: int = 42
) -> List[Dict]:
    """
    Strategically select examples for gold annotation.
    
    Selection strategy (as per Chapter 6.3):
    - Balance across all violation types
    - Include synthetic violations (to verify injection worked)
    - Include clean examples (controls)
    - Include edge cases (ambiguous examples)
    
    Args:
        examples: Full dataset with violation labels
        total_count: Target number of examples to select
        random_seed: For reproducibility
        
    Returns:
        Selected examples ready for annotation
    """
    random.seed(random_seed)
    
    # Group by violation type
    by_type = defaultdict(list)
    for ex in examples:
        vtype = ex.get('violation_type', 'unknown')
        by_type[vtype].append(ex)
    
    # Calculate per-type target (balanced)
    num_types = len(by_type)
    per_type_target = total_count // num_types
    
    selected = []
    
    print(f"Selecting {total_count} examples balanced across {num_types} violation types...")
    
    for vtype, type_examples in by_type.items():
        # Sample from each type
        sample_size = min(len(type_examples), per_type_target)
        sampled = random.sample(type_examples, sample_size)
        selected.extend(sampled)
        print(f"  {vtype}: {sample_size} examples")
    
    # Shuffle final selection
    random.shuffle(selected)
    
    # Trim to exact target if needed
    selected = selected[:total_count]
    
    print(f"\nTotal selected: {len(selected)} examples")
    return selected


def add_annotation_metadata(examples: List[Dict]) -> List[Dict]:
    """Add annotation-related fields to examples."""
    annotated = []
    
    for i, ex in enumerate(examples):
        annotated_ex = ex.copy()
        annotated_ex['annotation_id'] = f"GOLD_{i:04d}"
        annotated_ex['annotator_1'] = {
            'quantity': None,
            'quality': None,
            'relation': None,
            'manner': None,
            'notes': ''
        }
        annotated_ex['annotator_2'] = {
            'quantity': None,
            'quality': None,
            'relation': None,
            'manner': None,
            'notes': ''
        }
        annotated.append(annotated_ex)
    
    return annotated


# ============================================================================
# GOOGLE SHEETS EXPORT
# ============================================================================

def export_to_csv(examples: List[Dict], output_path: Path) -> None:
    """
    Export examples to CSV format for annotation in Google Sheets.
    
    Creates a clean spreadsheet format with:
    - Example ID
    - Context (truncated for display)
    - Evidence (truncated)
    - Response
    - Columns for each maxim score
    - Notes column
    """
    headers = [
        'annotation_id',
        'violation_type',
        'context',
        'evidence',
        'response',
        'quantity_score',
        'quality_score',
        'relation_score',
        'manner_score',
        'annotator_notes'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for ex in examples:
            # Truncate long fields for readability
            context = str(ex.get('context', ''))[:500]
            evidence = str(ex.get('evidence', ''))[:300]
            response = str(ex.get('violated_response', ex.get('response', '')))[:500]
            
            row = [
                ex.get('annotation_id', ''),
                ex.get('violation_type', ''),
                context,
                evidence,
                response,
                '',  # quantity_score (to be filled)
                '',  # quality_score
                '',  # relation_score
                '',  # manner_score
                ''   # notes
            ]
            writer.writerow(row)
    
    print(f"Exported {len(examples)} examples to {output_path}")


def import_annotations_from_csv(csv_path: Path) -> List[Dict]:
    """Import completed annotations from CSV/Google Sheets export."""
    annotations = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotation = {
                'annotation_id': row.get('annotation_id', ''),
                'quantity': _parse_score(row.get('quantity_score', '')),
                'quality': _parse_score(row.get('quality_score', '')),
                'relation': _parse_score(row.get('relation_score', '')),
                'manner': _parse_score(row.get('manner_score', '')),
                'notes': row.get('annotator_notes', '')
            }
            annotations.append(annotation)
    
    return annotations


def _parse_score(value: str) -> Optional[int]:
    """Parse annotation score from string."""
    if value in ['', 'None', 'null']:
        return None
    try:
        return int(value)
    except ValueError:
        return None


# ============================================================================
# INTER-ANNOTATOR AGREEMENT
# ============================================================================

def calculate_cohens_kappa(
    annotations_1: List[int],
    annotations_2: List[int]
) -> float:
    """
    Calculate Cohen's Kappa for two annotators.
    
    Cohen's Kappa measures agreement beyond chance:
    - kappa = 0: Agreement is random chance
    - kappa = 1: Perfect agreement
    - kappa > 0.6: Acceptable for NLP tasks
    - kappa > 0.8: Excellent agreement
    
    Args:
        annotations_1: First annotator's labels
        annotations_2: Second annotator's labels
        
    Returns:
        Cohen's Kappa score
    """
    assert len(annotations_1) == len(annotations_2), "Annotation lists must be same length"
    n = len(annotations_1)
    
    if n == 0:
        return 0.0
    
    # Get unique labels
    labels = sorted(set(annotations_1) | set(annotations_2))
    
    # Build confusion matrix
    confusion = defaultdict(int)
    for a1, a2 in zip(annotations_1, annotations_2):
        confusion[(a1, a2)] += 1
    
    # Calculate observed agreement (p_o)
    observed_agreement = sum(confusion[(l, l)] for l in labels) / n
    
    # Calculate expected agreement (p_e)
    count_1 = Counter(annotations_1)
    count_2 = Counter(annotations_2)
    
    expected_agreement = sum(
        (count_1[l] / n) * (count_2[l] / n)
        for l in labels
    )
    
    # Calculate kappa
    if expected_agreement == 1.0:
        return 1.0  # Perfect agreement
    
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa


def calculate_agreement_report(
    annotations_1: List[Dict],
    annotations_2: List[Dict]
) -> Dict:
    """
    Calculate inter-annotator agreement for all maxims.
    
    Args:
        annotations_1: First annotator's full annotations
        annotations_2: Second annotator's full annotations
        
    Returns:
        Report with kappa scores per maxim
    """
    report = {'per_maxim': {}, 'overall': {}}
    
    maxims = ['quantity', 'quality', 'relation', 'manner']
    
    all_scores_1 = []
    all_scores_2 = []
    
    for maxim in maxims:
        # Extract scores for this maxim
        scores_1 = [a.get(maxim) for a in annotations_1 if a.get(maxim) is not None]
        scores_2 = [a.get(maxim) for a in annotations_2 if a.get(maxim) is not None]
        
        # Ensure same length (paired annotations)
        min_len = min(len(scores_1), len(scores_2))
        scores_1 = scores_1[:min_len]
        scores_2 = scores_2[:min_len]
        
        if len(scores_1) > 0:
            kappa = calculate_cohens_kappa(scores_1, scores_2)
            agreement_pct = sum(a == b for a, b in zip(scores_1, scores_2)) / len(scores_1)
            
            report['per_maxim'][maxim] = {
                'cohens_kappa': round(kappa, 3),
                'raw_agreement': round(agreement_pct, 3),
                'n_examples': len(scores_1)
            }
            
            all_scores_1.extend(scores_1)
            all_scores_2.extend(scores_2)
    
    # Overall agreement
    if all_scores_1:
        overall_kappa = calculate_cohens_kappa(all_scores_1, all_scores_2)
        overall_agreement = sum(a == b for a, b in zip(all_scores_1, all_scores_2)) / len(all_scores_1)
        
        report['overall'] = {
            'cohens_kappa': round(overall_kappa, 3),
            'raw_agreement': round(overall_agreement, 3),
            'n_examples': len(all_scores_1)
        }
    
    return report


def print_agreement_report(report: Dict) -> None:
    """Print a formatted agreement report."""
    print("\n" + "=" * 50)
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print("=" * 50)
    
    print("\nPer-Maxim Agreement:")
    for maxim, stats in report.get('per_maxim', {}).items():
        kappa = stats['cohens_kappa']
        raw = stats['raw_agreement']
        n = stats['n_examples']
        
        # Interpret kappa
        if kappa > 0.8:
            interp = "Excellent"
        elif kappa > 0.6:
            interp = "Good"
        elif kappa > 0.4:
            interp = "Moderate"
        else:
            interp = "Poor"
        
        print(f"  {maxim.capitalize()}: κ={kappa:.3f} ({interp}), raw={raw:.1%}, n={n}")
    
    overall = report.get('overall', {})
    if overall:
        print(f"\nOverall: κ={overall['cohens_kappa']:.3f}, raw={overall['raw_agreement']:.1%}")
    
    print("=" * 50)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Create gold annotation set."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare gold annotation set')
    parser.add_argument('--input', type=str, default='data_processed/gricebench_weak_50k.json')
    parser.add_argument('--output-json', type=str, default='data_processed/gold_annotation_set.json')
    parser.add_argument('--output-csv', type=str, default='data_processed/gold_annotation_sheet.csv')
    parser.add_argument('--count', type=int, default=1000, help='Number of examples to select')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_json = project_root / args.output_json
    output_csv = project_root / args.output_csv
    
    # Load dataset
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")
    
    # Select examples for annotation
    selected = select_annotation_examples(examples, total_count=args.count, random_seed=args.seed)
    
    # Add annotation metadata
    annotated = add_annotation_metadata(selected)
    
    # Save JSON
    print(f"\nSaving JSON to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)
    
    # Export CSV for Google Sheets
    print(f"Exporting CSV to {output_csv}...")
    export_to_csv(annotated, output_csv)
    
    print("\n" + "=" * 50)
    print("ANNOTATION SETUP COMPLETE")
    print("=" * 50)
    print(f"\nFiles created:")
    print(f"  1. {output_json} - Full annotation data")
    print(f"  2. {output_csv} - Import to Google Sheets for annotation")
    print(f"\nNext steps:")
    print("  1. Upload CSV to Google Sheets")
    print("  2. Share with annotators")
    print("  3. Fill in scores (0, 1, 2 for Quantity; 0, 1 for others)")
    print("  4. Export completed sheet as CSV")
    print("  5. Run agreement calculation")
    print("=" * 50)


if __name__ == "__main__":
    main()
