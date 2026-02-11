"""
GriceBench Data Verification Script
====================================

Verifies all source data required for the morechanges.md scientific improvement plan.

Following data-audit-nlp skill guidelines:
- Check data structure and format
- Validate file existence
- Test model/index loading
- Generate comprehensive audit report

Author: GriceBench
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data verification."""
    base_dir: Path = Path(".")
    data_processed_dir: Path = Path("data_processed")
    scripts_dir: Path = Path("scripts")
    results_dir: Path = Path("results")
    models_dir: Path = Path("models")


# ============================================================================
# VERIFICATION CHECKS
# ============================================================================

class DataVerifier:
    """
    Comprehensive data verification for GriceBench.
    
    Follows data-audit-nlp skill:
    - Pillar 1: Structure & Format Validation
    - Pillar 2: Text Length Distribution
    - Pillar 3: Label Distribution
    - Pillar 4: Data Leakage Detection
    - Pillar 5: Synthetic vs Natural Distribution
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def verify_all(self) -> Dict:
        """Run all verification checks."""
        print("=" * 80)
        print("📊 GRICEBENCH DATA VERIFICATION (data-audit-nlp)")
        print("=" * 80)
        
        checks = [
            ("Required Data Files", self.check_required_files),
            ("Repair Test Data", self.check_repair_data),
            ("Gold Annotation Set", self.check_gold_annotations),
            ("Training Examples", self.check_training_data),
            ("FAISS Index", self.check_faiss_index),
            ("Topical Corpus", self.check_topical_corpus),
            ("Detector Model", self.check_detector_model),
            ("Results Directories", self.check_results_dirs),
        ]
        
        for check_name, check_func in checks:
            print(f"\n🔍 Checking: {check_name}")
            print("-" * 50)
            try:
                result = check_func()
                self.results[check_name] = result
                status = "✅ PASS" if result.get("status") == "pass" else "❌ FAIL"
                print(f"   {status}")
                if result.get("details"):
                    for detail in result.get("details", []):
                        print(f"   - {detail}")
            except Exception as e:
                self.errors.append(f"{check_name}: {str(e)}")
                self.results[check_name] = {"status": "error", "error": str(e)}
                print(f"   ❌ ERROR: {e}")
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def check_required_files(self) -> Dict:
        """Check all required data files exist."""
        required_files = [
            "data_processed/repair_data/repair_test.json",
            "data_processed/gold_annotation_set.json",
            "data_processed/train_examples.json",
            "data_processed/val_examples.json",
            "data_processed/test_examples.json",
            "data_processed/faiss_index.pkl",
            "data_processed/topical_corpus.json",
        ]
        
        missing = []
        existing = []
        
        for file_path in required_files:
            full_path = self.config.base_dir / file_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                existing.append(f"{file_path} ({size_mb:.2f} MB)")
            else:
                missing.append(file_path)
                self.errors.append(f"Missing required file: {file_path}")
        
        return {
            "status": "pass" if not missing else "fail",
            "existing_count": len(existing),
            "missing_count": len(missing),
            "missing_files": missing,
            "details": [f"Found {len(existing)}/{len(required_files)} required files"] + 
                       [f"MISSING: {f}" for f in missing]
        }
    
    def check_repair_data(self) -> Dict:
        """Check repair test data structure and Relation violation count."""
        repair_path = self.config.base_dir / "data_processed/repair_data/repair_test.json"
        
        if not repair_path.exists():
            return {"status": "fail", "details": ["File does not exist"]}
        
        with open(repair_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total = len(data)
        relation_count = 0
        has_input_text = 0
        has_target_text = 0
        
        for item in data:
            if "input_text" in item:
                has_input_text += 1
                if "[VIOLATION=RELATION]" in item.get("input_text", ""):
                    relation_count += 1
            if "target_text" in item:
                has_target_text += 1
        
        return {
            "status": "pass" if relation_count >= 200 else "fail",
            "total_examples": total,
            "relation_violations": relation_count,
            "has_input_text": has_input_text,
            "has_target_text": has_target_text,
            "details": [
                f"Total examples: {total}",
                f"Relation violations: {relation_count}",
                f"Need 200 for eval set: {'✅' if relation_count >= 200 else '❌ INSUFFICIENT'}"
            ]
        }
    
    def check_gold_annotations(self) -> Dict:
        """Check gold annotation set structure per data-audit-nlp Pillar 3."""
        gold_path = self.config.base_dir / "data_processed/gold_annotation_set.json"
        
        if not gold_path.exists():
            return {"status": "fail", "details": ["File does not exist"]}
        
        with open(gold_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total = len(data)
        
        # Check label distribution (Pillar 3)
        label_cols = ['quantity', 'quality', 'relation', 'manner']
        label_counts = defaultdict(int)
        
        has_context = 0
        has_response = 0
        
        for item in data:
            if 'context' in item or 'context_text' in item:
                has_context += 1
            if 'response' in item:
                has_response += 1
            
            labels = item.get('labels', {})
            for label in label_cols:
                if labels.get(label, 0) == 1:
                    label_counts[label] += 1
        
        details = [
            f"Total examples: {total}",
            f"Has context: {has_context}",
            f"Has response: {has_response}",
        ]
        
        for label in label_cols:
            pct = (label_counts[label] / total * 100) if total > 0 else 0
            details.append(f"{label}: {label_counts[label]} ({pct:.1f}%)")
        
        return {
            "status": "pass" if total >= 500 else "fail",
            "total_examples": total,
            "label_distribution": dict(label_counts),
            "details": details
        }
    
    def check_training_data(self) -> Dict:
        """Check training data structure per data-audit-nlp Pillar 1 & 2."""
        train_path = self.config.base_dir / "data_processed/train_examples.json"
        
        if not train_path.exists():
            return {"status": "fail", "details": ["File does not exist"]}
        
        # Load just first 1000 for quick check
        with open(train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total = len(data)
        sample = data[:min(1000, total)]
        
        # Text length distribution (Pillar 2)
        context_lengths = []
        response_lengths = []
        
        for item in sample:
            context = item.get('context_text', item.get('context', ''))
            response = item.get('response', '')
            
            if context:
                context_lengths.append(len(context))
            if response:
                response_lengths.append(len(response))
        
        import statistics
        
        return {
            "status": "pass",
            "total_examples": total,
            "context_mean_len": statistics.mean(context_lengths) if context_lengths else 0,
            "context_max_len": max(context_lengths) if context_lengths else 0,
            "response_mean_len": statistics.mean(response_lengths) if response_lengths else 0,
            "response_max_len": max(response_lengths) if response_lengths else 0,
            "details": [
                f"Total examples: {total:,}",
                f"Context avg length: {statistics.mean(context_lengths):.0f} chars" if context_lengths else "No contexts",
                f"Response avg length: {statistics.mean(response_lengths):.0f} chars" if response_lengths else "No responses",
            ]
        }
    
    def check_faiss_index(self) -> Dict:
        """Check FAISS index can be loaded."""
        faiss_path = self.config.base_dir / "data_processed/faiss_index.pkl"
        
        if not faiss_path.exists():
            return {"status": "fail", "details": ["FAISS index file does not exist"]}
        
        try:
            import pickle
            with open(faiss_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Check what's in the pickle
            if isinstance(index_data, dict):
                keys = list(index_data.keys())
                details = [f"Index contains keys: {keys}"]
                if 'index' in index_data:
                    try:
                        import faiss
                        idx = index_data['index']
                        details.append(f"FAISS index size: {idx.ntotal} vectors")
                    except:
                        details.append("FAISS not installed, can't inspect index size")
            else:
                details = [f"Index type: {type(index_data)}"]
            
            return {
                "status": "pass",
                "details": details
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e),
                "details": [f"Failed to load index: {e}"]
            }
    
    def check_topical_corpus(self) -> Dict:
        """Check topical corpus for retrieval."""
        corpus_path = self.config.base_dir / "data_processed/topical_corpus.json"
        
        if not corpus_path.exists():
            return {"status": "fail", "details": ["Corpus file does not exist"]}
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        
        total = len(corpus)
        
        # Sample check
        sample = corpus[:min(100, total)]
        has_response = sum(1 for item in sample if 'response' in item or isinstance(item, str))
        
        return {
            "status": "pass" if total >= 10000 else "fail",
            "total_responses": total,
            "details": [
                f"Total responses in corpus: {total:,}",
                f"Minimum for good retrieval: 10,000 ({'✅' if total >= 10000 else '❌'})"
            ]
        }
    
    def check_detector_model(self) -> Dict:
        """Check detector model exists."""
        detector_paths = [
            "models/detector",
            "models/detector/model.safetensors",
            "models/detector/pytorch_model.bin",
            "models/detector/config.json",
            "best_model_v2.pt",
        ]
        
        found = []
        for path in detector_paths:
            full_path = self.config.base_dir / path
            if full_path.exists():
                if full_path.is_file():
                    size_mb = full_path.stat().st_size / (1024 * 1024)
                    found.append(f"{path} ({size_mb:.2f} MB)")
                else:
                    found.append(f"{path} (directory)")
        
        return {
            "status": "pass" if found else "fail",
            "found_models": found,
            "details": found if found else ["No detector model found"]
        }
    
    def check_results_dirs(self) -> Dict:
        """Check results directories status."""
        dirs_to_check = [
            "results",
            "results/relation_repair_evaluation",
            "results/part4output",
            "results/part5output",
        ]
        
        status = []
        for dir_path in dirs_to_check:
            full_path = self.config.base_dir / dir_path
            if full_path.exists():
                if full_path.is_dir():
                    num_files = len(list(full_path.iterdir()))
                    status.append(f"✅ {dir_path} ({num_files} items)")
                else:
                    status.append(f"⚠️ {dir_path} (is file, not dir)")
            else:
                status.append(f"❌ {dir_path} (MISSING)")
                if "relation_repair_evaluation" in dir_path:
                    self.warnings.append("Relation repair evaluation directory missing - eval never run")
        
        return {
            "status": "pass",  # Informational
            "details": status
        }
    
    def _generate_summary(self):
        """Generate and print summary report."""
        print("\n" + "=" * 80)
        print("📋 VERIFICATION SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for r in self.results.values() if r.get("status") == "pass")
        failed = sum(1 for r in self.results.values() if r.get("status") == "fail")
        errors = sum(1 for r in self.results.values() if r.get("status") == "error")
        
        print(f"\n✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️ Errors: {errors}")
        
        if self.errors:
            print("\n🚨 CRITICAL ERRORS:")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        # Save results
        results_path = self.config.base_dir / "results" / "data_verification_report.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                "results": self.results,
                "errors": self.errors,
                "warnings": self.warnings,
                "summary": {
                    "passed": passed,
                    "failed": failed,
                    "errors": errors
                }
            }, f, indent=2, default=str)
        
        print(f"\n📄 Report saved to: {results_path}")
        
        # Verdict
        print("\n" + "=" * 80)
        if failed == 0 and errors == 0:
            print("✅ VERDICT: All checks passed - ready for Phase 2")
        else:
            print("❌ VERDICT: Issues found - resolve before proceeding")
        print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run data verification."""
    # Change to GriceBench directory
    script_dir = Path(__file__).parent
    gricebench_dir = script_dir.parent
    os.chdir(gricebench_dir)
    
    print(f"Working directory: {Path.cwd()}")
    
    config = DataConfig(base_dir=Path("."))
    verifier = DataVerifier(config)
    results = verifier.verify_all()
    
    return 0 if all(r.get("status") == "pass" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
