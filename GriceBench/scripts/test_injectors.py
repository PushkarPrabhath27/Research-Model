"""
Test script for violation injectors.
Validates all 4 injectors work correctly before generating full dataset.
"""

import json
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from violation_injectors import (
    QuantityInjector, QualityInjector, RelationInjector, MannerInjector,
    ViolationInjectionPipeline
)


def test_individual_injectors():
    """Test each injector independently."""
    print("=" * 60)
    print("TESTING INDIVIDUAL VIOLATION INJECTORS")
    print("=" * 60)
    
    # Sample test data
    response = "Mount Everest is 29,032 feet tall and was first summited in 1953 by Edmund Hillary and Tenzing Norgay."
    context = "Tell me about Mount Everest and its history."
    evidence = {'height': '29,032 feet', 'first_summit': '1953'}
    
    print(f"\nOriginal response ({len(response.split())} words):")
    print(f'  "{response}"')
    
    total_violations = 0
    
    # Test Quantity
    print("\n--- QUANTITY INJECTOR ---")
    qi = QuantityInjector()
    q_violations = qi.inject(response, context, evidence)
    for v in q_violations:
        print(f"  {v.violation_type}: ({len(v.violated_response.split())} words)")
        print(f'    "{v.violated_response[:100]}..."')
    total_violations += len(q_violations)
    print(f"  Generated {len(q_violations)} violations")
    
    # Test Quality
    print("\n--- QUALITY INJECTOR ---")
    qli = QualityInjector()
    ql_violations = qli.inject(response, context, evidence)
    for v in ql_violations:
        print(f"  {v.violation_type}:")
        print(f'    "{v.violated_response[:100]}..."')
    total_violations += len(ql_violations)
    print(f"  Generated {len(ql_violations)} violations")
    
    # Test Relation
    print("\n--- RELATION INJECTOR ---")
    ri = RelationInjector()
    r_violations = ri.inject(response, context, evidence)
    for v in r_violations:
        print(f"  {v.violation_type}:")
        print(f'    "{v.violated_response[:100]}..."')
    total_violations += len(r_violations)
    print(f"  Generated {len(r_violations)} violations")
    
    # Test Manner
    print("\n--- MANNER INJECTOR ---")
    mi = MannerInjector()
    m_violations = mi.inject(response, context, evidence)
    for v in m_violations:
        print(f"  {v.violation_type}:")
        print(f'    "{v.violated_response[:100]}..."')
    total_violations += len(m_violations)
    print(f"  Generated {len(m_violations)} violations")
    
    print("\n" + "=" * 60)
    print(f"Total violations from single example: {total_violations}")
    return total_violations > 0


def test_pipeline_small():
    """Test the full pipeline on a small sample."""
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE (small sample)")
    print("=" * 60)
    
    # Load a few real examples
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data_processed" / "train_examples.json"
    
    print(f"\nLoading examples from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        all_examples = json.load(f)
    
    # Take small sample
    sample_examples = all_examples[:50]
    print(f"Testing on {len(sample_examples)} examples...")
    
    # Run pipeline
    pipeline = ViolationInjectionPipeline(random_seed=42)
    dataset = pipeline.generate_dataset(sample_examples, target_size=200)
    
    print(f"\nGenerated {len(dataset)} examples from {len(sample_examples)} source examples")
    
    # Check distribution
    from collections import Counter
    type_dist = Counter(ex['violation_type'] for ex in dataset)
    print("\nViolation type distribution:")
    for vtype, count in sorted(type_dist.items()):
        print(f"  {vtype}: {count}")
    
    return len(dataset) > 0


def main():
    print("\n" + "=" * 60)
    print("GRICEBENCH VIOLATION INJECTOR TESTS")
    print("=" * 60)
    
    # Test 1: Individual injectors
    injectors_ok = test_individual_injectors()
    
    # Test 2: Full pipeline
    pipeline_ok = test_pipeline_small()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Individual injectors: {'PASSED' if injectors_ok else 'FAILED'}")
    print(f"  Pipeline test: {'PASSED' if pipeline_ok else 'FAILED'}")
    
    if injectors_ok and pipeline_ok:
        print("\n✓ All tests passed! Ready to generate full dataset.")
        return 0
    else:
        print("\n✗ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    exit(main())
