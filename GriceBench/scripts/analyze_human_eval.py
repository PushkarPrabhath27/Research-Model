"""
GriceBench Human Evaluation - Analysis - Part 2, Step 4
========================================================

Analyzes human evaluation results:
1. Loads all annotation files
2. Calculates inter-annotator agreement (Krippendorff's alpha)
3. Compares systems using Mann-Whitney U tests
4. Generates comprehensive markdown report

Author: GriceBench
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    results_dir: str = "human_eval_results"
    key_path: str = "human_eval_key_DO_NOT_SHARE.json"
    report_path: str = "reports/human_evaluation_report.md"


# ============================================================================
# DIMENSIONS
# ============================================================================

DIMENSIONS = ["helpfulness", "accuracy", "relevance", "clarity", "conciseness"]


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

class HumanEvalAnalyzer:
    """
    Analyzes human evaluation results.
    
    Features:
    - Krippendorff's alpha for inter-annotator agreement
    - Mann-Whitney U tests for system comparison
    - Comprehensive markdown report generation
    """
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.all_results = []
        self.system_key = {}
    
    def load_annotations(self) -> List[Dict]:
        """Load all annotation files from results directory."""
        results_dir = Path(self.config.results_dir)
        
        if not results_dir.exists():
            print(f"Warning: Results directory not found: {results_dir}")
            return []
        
        all_results = []
        
        for filename in results_dir.iterdir():
            if filename.suffix == ".json":
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        results = json.load(f)
                        all_results.extend(results)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
        
        self.all_results = all_results
        print(f"Loaded {len(all_results)} annotations from {results_dir}")
        
        return all_results
    
    def load_system_key(self) -> Dict:
        """Load the system key for unblinding."""
        key_path = Path(self.config.key_path)
        
        if not key_path.exists():
            print(f"Warning: System key not found: {key_path}")
            return {}
        
        with open(key_path, "r", encoding="utf-8") as f:
            self.system_key = json.load(f)
        
        print(f"Loaded system key with {len(self.system_key)} mappings")
        return self.system_key
    
    def calculate_inter_annotator_agreement(self) -> Dict[str, float]:
        """
        Calculate Krippendorff's alpha for inter-annotator agreement.
        
        Krippendorff's alpha is preferred over Fleiss' kappa because:
        - Handles missing data
        - Works with ordinal scales
        - More robust to sample size
        """
        if not self.all_results:
            return {}
        
        # Group by sample_id
        by_sample = defaultdict(list)
        for r in self.all_results:
            sample_id = r.get("sample_id")
            if sample_id is not None:
                by_sample[sample_id].append(r)
        
        # Only use samples with multiple annotations
        multi_annotated = {k: v for k, v in by_sample.items() if len(v) >= 2}
        
        if not multi_annotated:
            print("Warning: No samples with multiple annotations found")
            return {dim: None for dim in DIMENSIONS}
        
        print(f"Found {len(multi_annotated)} samples with multiple annotations")
        
        # Get all annotators
        all_annotators = set()
        for sample_id, annotations in multi_annotated.items():
            for ann in annotations:
                all_annotators.add(ann.get("annotator_id", "unknown"))
        
        annotators = list(all_annotators)
        samples = list(multi_annotated.keys())
        
        agreements = {}
        
        for dim in DIMENSIONS:
            # Build reliability matrix (annotators x samples)
            # Use np.nan for missing values
            matrix = np.full((len(annotators), len(samples)), np.nan)
            
            for col, sample_id in enumerate(samples):
                for ann in multi_annotated[sample_id]:
                    annotator_id = ann.get("annotator_id", "unknown")
                    if annotator_id in annotators:
                        row = annotators.index(annotator_id)
                        rating = ann.get("ratings", {}).get(dim)
                        if rating is not None:
                            matrix[row, col] = rating
            
            # Calculate Krippendorff's alpha
            alpha_val = self._calculate_krippendorff_alpha(matrix)
            agreements[dim] = alpha_val
        
        return agreements
    
    def _calculate_krippendorff_alpha(self, matrix: np.ndarray) -> Optional[float]:
        """
        Calculate Krippendorff's alpha for ordinal data.
        
        Uses the krippendorff package if available, otherwise
        uses a simplified calculation.
        """
        try:
            from krippendorff import alpha
            return alpha(reliability_data=matrix, level_of_measurement="ordinal")
        except ImportError:
            # Simplified calculation (correlation-based approximation)
            return self._simplified_alpha(matrix)
        except Exception as e:
            print(f"Warning: Could not calculate alpha: {e}")
            return None
    
    def _simplified_alpha(self, matrix: np.ndarray) -> Optional[float]:
        """Simplified inter-annotator agreement (average correlation)."""
        valid_cols = []
        
        for col in range(matrix.shape[1]):
            column = matrix[:, col]
            non_nan = column[~np.isnan(column)]
            if len(non_nan) >= 2:
                valid_cols.append(column)
        
        if len(valid_cols) < 2:
            return None
        
        # Calculate pairwise correlations between annotators where both rated
        correlations = []
        
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[0]):
                # Find samples both annotators rated
                both_rated = ~(np.isnan(matrix[i, :]) | np.isnan(matrix[j, :]))
                if np.sum(both_rated) >= 3:
                    try:
                        corr = np.corrcoef(matrix[i, both_rated], matrix[j, both_rated])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                    except:
                        pass
        
        if correlations:
            return np.mean(correlations)
        return None
    
    def compare_systems(self) -> Dict:
        """Compare performance across systems."""
        if not self.all_results or not self.system_key:
            return {}
        
        # Group results by system
        by_system = defaultdict(list)
        
        for r in self.all_results:
            sample_id = str(r.get("sample_id"))
            if sample_id in self.system_key:
                system = self.system_key[sample_id]
                by_system[system].append(r)
        
        if not by_system:
            return {}
        
        # Calculate mean ratings per system per dimension
        system_scores = {}
        
        for system, annotations in by_system.items():
            system_scores[system] = {"n": len(annotations)}
            
            for dim in DIMENSIONS:
                ratings = [
                    a.get("ratings", {}).get(dim)
                    for a in annotations
                    if a.get("ratings", {}).get(dim) is not None
                ]
                
                if ratings:
                    system_scores[system][dim] = {
                        "mean": float(np.mean(ratings)),
                        "std": float(np.std(ratings)),
                        "n": len(ratings)
                    }
        
        return system_scores
    
    def run_significance_tests(self, reference_system: str = "gricebench_repair") -> Dict:
        """Run statistical significance tests comparing systems."""
        from scipy import stats
        
        if not self.all_results or not self.system_key:
            return {}
        
        # Group by system
        by_system = defaultdict(list)
        for r in self.all_results:
            sample_id = str(r.get("sample_id"))
            if sample_id in self.system_key:
                system = self.system_key[sample_id]
                by_system[system].append(r)
        
        if reference_system not in by_system:
            # Find any available reference
            reference_system = list(by_system.keys())[0] if by_system else None
        
        if not reference_system:
            return {}
        
        ref_scores = by_system[reference_system]
        significance_results = {}
        
        for system in by_system:
            if system == reference_system:
                continue
            
            other_scores = by_system[system]
            significance_results[system] = {}
            
            for dim in DIMENSIONS:
                ref_ratings = [
                    a.get("ratings", {}).get(dim)
                    for a in ref_scores
                    if a.get("ratings", {}).get(dim) is not None
                ]
                other_ratings = [
                    a.get("ratings", {}).get(dim)
                    for a in other_scores
                    if a.get("ratings", {}).get(dim) is not None
                ]
                
                if ref_ratings and other_ratings:
                    # Mann-Whitney U test (non-parametric)
                    stat, p_value = stats.mannwhitneyu(
                        ref_ratings, other_ratings,
                        alternative='two-sided'
                    )
                    
                    ref_mean = np.mean(ref_ratings)
                    other_mean = np.mean(other_ratings)
                    
                    significance_results[system][dim] = {
                        "ref_mean": ref_mean,
                        "other_mean": other_mean,
                        "difference": ref_mean - other_mean,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
        
        return significance_results
    
    def generate_report(self) -> str:
        """Generate comprehensive markdown report."""
        
        # Load data
        self.load_annotations()
        self.load_system_key()
        
        # Calculate metrics
        agreement = self.calculate_inter_annotator_agreement()
        system_scores = self.compare_systems()
        
        try:
            significance = self.run_significance_tests()
        except ImportError:
            significance = {}
            print("scipy not available, skipping significance tests")
        
        # Generate report
        report = f"""# Human Evaluation Report for GriceBench

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

Total annotations: {len(self.all_results)}
Systems evaluated: {len(system_scores)}
Dimensions: {', '.join(DIMENSIONS)}

---

## Inter-Annotator Agreement

Krippendorff's Alpha (ordinal scale, 1.0 = perfect agreement):

| Dimension | Alpha | Interpretation |
|-----------|-------|----------------|
"""
        
        for dim, alpha_val in agreement.items():
            if alpha_val is not None:
                if alpha_val >= 0.8:
                    interp = "Strong agreement"
                elif alpha_val >= 0.6:
                    interp = "Moderate agreement"
                elif alpha_val >= 0.4:
                    interp = "Fair agreement"
                else:
                    interp = "Poor agreement"
                report += f"| {dim.capitalize()} | {alpha_val:.3f} | {interp} |\n"
            else:
                report += f"| {dim.capitalize()} | N/A | Insufficient data |\n"
        
        report += """
---

## System Comparison

Mean ratings per system (1-5 scale):

| System | N | Helpfulness | Accuracy | Relevance | Clarity | Conciseness |
|--------|---|-------------|----------|-----------|---------|-------------|
"""
        
        for system, scores in system_scores.items():
            row = f"| {system} | {scores.get('n', 0)} "
            for dim in DIMENSIONS:
                if dim in scores:
                    row += f"| {scores[dim]['mean']:.2f} "
                else:
                    row += "| N/A "
            report += row + "|\n"
        
        if significance:
            report += """
---

## Statistical Significance

Mann-Whitney U tests comparing systems:

"""
            for system, dims in significance.items():
                report += f"### {system} vs Reference\n\n"
                report += "| Dimension | Δ Mean | p-value | Significant? |\n"
                report += "|-----------|--------|---------|-------------|\n"
                
                for dim, stats in dims.items():
                    sig_marker = "✅ Yes" if stats["significant"] else "❌ No"
                    report += f"| {dim.capitalize()} | {stats['difference']:+.2f} | {stats['p_value']:.4f} | {sig_marker} |\n"
                
                report += "\n"
        
        report += """
---

## Conclusions

Based on the human evaluation results, we can draw the following conclusions:

1. **Inter-annotator agreement** indicates the reliability of our evaluation protocol
2. **System comparison** shows how GriceBench compares to baselines
3. **Statistical significance** confirms whether differences are meaningful

---

*Note: Higher scores are better across all dimensions.*
"""
        
        # Save report
        report_path = Path(self.config.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\nReport saved to {report_path}")
        
        return report
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*70)
        print("HUMAN EVALUATION ANALYSIS SUMMARY")
        print("="*70)
        
        # Load if not already loaded
        if not self.all_results:
            self.load_annotations()
            self.load_system_key()
        
        print(f"\nTotal annotations: {len(self.all_results)}")
        
        # Agreement
        agreement = self.calculate_inter_annotator_agreement()
        print("\nInter-Annotator Agreement (Krippendorff's α):")
        for dim, alpha_val in agreement.items():
            if alpha_val is not None:
                print(f"  {dim}: {alpha_val:.3f}")
            else:
                print(f"  {dim}: N/A")
        
        # System scores
        system_scores = self.compare_systems()
        if system_scores:
            print("\nSystem Comparison (Mean Scores):")
            for system, scores in system_scores.items():
                print(f"\n  {system.upper()} (n={scores.get('n', 0)})")
                for dim in DIMENSIONS:
                    if dim in scores:
                        print(f"    {dim}: {scores[dim]['mean']:.2f} ± {scores[dim]['std']:.2f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run human evaluation analysis."""
    print("="*70)
    print("HUMAN EVALUATION ANALYSIS")
    print("="*70)
    
    analyzer = HumanEvalAnalyzer()
    
    # Print summary to console
    analyzer.print_summary()
    
    # Generate full report
    analyzer.generate_report()
    
    print("\n" + "="*70)
    print("PART 2 COMPLETE: Human Evaluation Framework Ready!")
    print("="*70)
    print("\nFiles created:")
    print("1. ✅ human_eval_interface.py (CLI)")
    print("2. ✅ human_eval_gradio.py (Web)")
    print("3. ✅ prepare_human_eval_samples.py (Sample prep)")
    print("4. ✅ analyze_human_eval.py (Analysis)")
    print("\nNext: Part 3 - Baseline Comparisons")


if __name__ == "__main__":
    main()
