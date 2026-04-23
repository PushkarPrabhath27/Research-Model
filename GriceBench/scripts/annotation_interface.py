"""
Annotation Interface
====================

Interactive CLI tool for self-annotation of GriceBench examples.
Per morechanges.md lines 1631-1736.

Features:
- Loads examples from JSON
- Collects ratings for each maxim
- Auto-saves progress
- Resumes from last position

Author: GriceBench
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Annotation:
    """Single example annotation."""
    id: str
    context: str
    response: str
    quantity: str  # "too_little", "appropriate", "too_much"
    quality: str   # "unsupported", "appropriate"
    relation: str  # "off_topic", "tangential", "relevant"
    manner: str    # "unclear", "clear"
    helpfulness: int  # 1-5 scale
    justification: str
    detector_violations: Dict  # What detector said


class AnnotationInterface:
    """
    Interactive annotation interface for GriceBench.
    
    Usage:
        python scripts/annotation_interface.py
    """
    
    def __init__(self, data_file: str, output_file: str):
        self.data_file = Path(data_file)
        self.output_file = Path(output_file)
        self.data = self.load_data()
        self.annotations = self.load_existing_annotations()
        self.current_idx = len(self.annotations)
    
    def load_data(self) -> List[Dict]:
        """Load examples to annotate."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_existing_annotations(self) -> List[Dict]:
        """Load existing annotations to resume."""
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save(self):
        """Save annotations to file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
    
    def get_input(self, prompt: str, valid_options: List[str]) -> str:
        """Get validated input from user."""
        while True:
            response = input(prompt).strip()
            if response.lower() == 'q':
                raise KeyboardInterrupt
            if response in valid_options:
                return response
            print(f"  Invalid. Enter: {', '.join(valid_options)} (or 'q' to quit)")
    
    def annotate_example(self, example: Dict) -> Dict:
        """Annotate a single example."""
        context = example.get('context', example.get('context_text', ''))
        response = example.get('response', '')
        
        print("\n" + "=" * 80)
        print(f"Example {self.current_idx + 1}/{len(self.data)}")
        print("=" * 80)
        
        print(f"\n📝 CONTEXT:\n{context[:500]}{'...' if len(context) > 500 else ''}")
        print(f"\n💬 RESPONSE:\n{response[:500]}{'...' if len(response) > 500 else ''}")
        print("-" * 80)
        
        # Quantity
        print("\n1. QUANTITY (Information Amount):")
        print("   1 = Too little, 2 = Appropriate, 3 = Too much")
        quantity = self.get_input("   > ", ['1', '2', '3'])
        quantity_map = {'1': 'too_little', '2': 'appropriate', '3': 'too_much'}
        
        # Quality
        print("\n2. QUALITY (Truthfulness):")
        print("   1 = Unsupported/contradictory, 2 = Appropriate")
        quality = self.get_input("   > ", ['1', '2'])
        quality_map = {'1': 'unsupported', '2': 'appropriate'}
        
        # Relation
        print("\n3. RELATION (Relevance):")
        print("   1 = Off-topic, 2 = Tangential, 3 = Relevant")
        relation = self.get_input("   > ", ['1', '2', '3'])
        relation_map = {'1': 'off_topic', '2': 'tangential', '3': 'relevant'}
        
        # Manner
        print("\n4. MANNER (Clarity):")
        print("   1 = Unclear/disorganized, 2 = Clear")
        manner = self.get_input("   > ", ['1', '2'])
        manner_map = {'1': 'unclear', '2': 'clear'}
        
        # Helpfulness
        print("\n5. OVERALL HELPFULNESS (1-5):")
        helpfulness = self.get_input("   > ", ['1', '2', '3', '4', '5'])
        
        # Justification
        print("\n6. Brief justification (optional, Enter to skip):")
        justification = input("   > ").strip()
        
        # Convert to binary violations for comparison
        violations = {
            'quantity': quantity != '2',
            'quality': quality == '1',
            'relation': relation == '1',
            'manner': manner == '1',
        }
        
        return {
            'id': example.get('id', str(self.current_idx)),
            'context': context,
            'response': response,
            'quantity': quantity_map[quantity],
            'quality': quality_map[quality],
            'relation': relation_map[relation],
            'manner': manner_map[manner],
            'helpfulness': int(helpfulness),
            'justification': justification,
            'violations': violations,
            'detector_violations': example.get('labels', example.get('violations', {}))
        }
    
    def run(self):
        """Main annotation loop."""
        print("\n" + "=" * 80)
        print("GRICEBENCH ANNOTATION INTERFACE")
        print("=" * 80)
        print(f"\nData file: {self.data_file}")
        print(f"Output file: {self.output_file}")
        print(f"Progress: {self.current_idx}/{len(self.data)} annotated")
        print("\nPress 'q' at any prompt to quit and save.")
        
        try:
            while self.current_idx < len(self.data):
                example = self.data[self.current_idx]
                annotation = self.annotate_example(example)
                
                self.annotations.append(annotation)
                self.current_idx += 1
                
                # Auto-save every 5 examples
                if self.current_idx % 5 == 0:
                    self.save()
                    print(f"\n✅ Progress saved: {self.current_idx}/{len(self.data)}")
                
                # Self-consistency reminder every 50
                if self.current_idx % 50 == 0:
                    print("\n⚠️ REMINDER: Consider re-checking 5 random previous annotations")
        
        except KeyboardInterrupt:
            print("\n\nQuitting...")
        
        finally:
            self.save()
            print(f"\n✅ Final save: {len(self.annotations)} annotations")
            print(f"   Saved to: {self.output_file}")


def main():
    """Run annotation interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GriceBench Annotation Interface")
    parser.add_argument("--data", default="data_processed/annotation_sample_1000.json",
                       help="Path to data file")
    parser.add_argument("--output", default="data_processed/self_annotations.json",
                       help="Path to output file")
    
    args = parser.parse_args()
    
    interface = AnnotationInterface(args.data, args.output)
    interface.run()


if __name__ == "__main__":
    main()
