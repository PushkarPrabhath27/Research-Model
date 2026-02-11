"""
GriceBench Comprehensive Diagnostic Suite

This script implements the complete diagnostic analysis to identify root causes:
1. Weak label distribution analysis
2. Detector calibration analysis  
3. DPO preference pair quality analysis
4. Baseline response quality analysis

Run this FIRST to confirm the hypotheses before applying fixes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import re

class GriceBenchDiagnostics:
    """Comprehensive diagnostic suite to identify root causes"""
    
    def __init__(self, data_dir="data_processed", results_dir="."):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
    # ============================================
    # DIAGNOSTIC 1: Weak Label Distribution
    # ============================================
    
    def analyze_weak_labels(self):
        """Analyze weak label quality and distribution"""
        print("="*60)
        print("DIAGNOSTIC 1: WEAK LABEL ANALYSIS")
        print("="*60)
        
        # Load weak labels
        weak_file = self.data_dir / "gricebench_weak_labeled.json"
        if not weak_file.exists():
            print(f"❌ File not found: {weak_file}")
            print("   Trying alternative: gricebench_weak_50k.json")
            weak_file = self.data_dir / "gricebench_weak_50k.json"
            
        if not weak_file.exists():
            print(f"❌ File not found: {weak_file}")
            return None
            
        with open(weak_file) as f:
            data = json.load(f)
        
        # Extract violation scores
        maxims = ['quantity', 'quality', 'relation', 'manner']
        scores = {m: [] for m in maxims}
        
        for item in data:
            for maxim in maxims:
                key = f"{maxim}_violation"
                if key in item:
                    val = item[key]
                    # Handle both binary and continuous labels
                    if isinstance(val, (int, float)):
                        scores[maxim].append(float(val))
        
        if not scores['quantity']:
            print("❌ No violation scores found in data")
            return None
        
        # Analyze distribution
        print(f"\n📊 Weak Label Statistics (n={len(data)}):")
        print("-"*60)
        
        results = {}
        for maxim in maxims:
            vals = np.array(scores[maxim])
            print(f"\n{maxim.upper()}:")
            print(f"  Mean:   {vals.mean():.3f}")
            print(f"  Std:    {vals.std():.3f}")
            print(f"  Min:    {vals.min():.3f}")
            print(f"  Max:    {vals.max():.3f}")
            print(f"  Median: {np.median(vals):.3f}")
            
            # Check if binary or continuous
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 10:
                print(f"  Type:   BINARY/DISCRETE ({len(unique_vals)} unique values)")
                print(f"  Values: {sorted(unique_vals)[:10]}")
            else:
                print(f"  Type:   CONTINUOUS ({len(unique_vals)} unique values)")
                
            # Check for clustering around 0.5
            mid_range = np.sum((vals > 0.4) & (vals < 0.6)) / len(vals)
            print(f"  🎯 Clustering at 0.5: {mid_range*100:.1f}% of values")
            
            if mid_range > 0.4:
                print(f"  🚨 WARNING: {mid_range*100:.1f}% of labels are uncertain (0.4-0.6)")
                print(f"     This indicates weak label noise!")
            
            results[maxim] = {
                'mean': vals.mean(),
                'std': vals.std(),
                'clustering_0.5': mid_range,
                'type': 'continuous' if len(unique_vals) > 10 else 'discrete'
            }
        
        # Create histogram
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Weak Label Distribution', fontsize=16)
            
            for idx, maxim in enumerate(maxims):
                ax = axes[idx//2, idx%2]
                ax.hist(scores[maxim], bins=50, edgecolor='black', alpha=0.7)
                ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision boundary')
                ax.set_title(f'{maxim.capitalize()} Violations')
                ax.set_xlabel('Violation Score')
                ax.set_ylabel('Count')
                ax.legend()
                ax.grid(alpha=0.3)
                
            plt.tight_layout()
            plt.savefig('weak_label_distribution.png', dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved histogram to weak_label_distribution.png")
        except Exception as e:
            print(f"⚠️  Could not create plot: {e}")
        
        return results
        
    # ============================================
    # DIAGNOSTIC 2: Detector Calibration
    # ============================================
    
    def analyze_detector_calibration(self):
        """Analyze detector probability calibration"""
        print("\n" + "="*60)
        print("DIAGNOSTIC 2: DETECTOR CALIBRATION ANALYSIS")
        print("="*60)
        
        # Load evaluation results
        results_file = self.results_dir / "dpo_evaluation_results.json"
        if not results_file.exists():
            print(f"❌ File not found: {results_file}")
            return None
            
        with open(results_file) as f:
            data = json.load(f)
        
        # Extract detector probabilities
        maxims = ['quantity', 'quality', 'relation', 'manner']
        baseline_probs = {m: [] for m in maxims}
        dpo_probs = {m: [] for m in maxims}
        
        for item in data['examples']:
            for maxim in maxims:
                prob_key = f"{maxim}_prob"
                baseline_probs[maxim].append(
                    item['baseline_violations'][prob_key]
                )
                dpo_probs[maxim].append(
                    item['dpo_violations'][prob_key]
                )
        
        # Analyze distribution
        print(f"\n📊 Detector Probability Distribution (n={len(data['examples'])}):")
        print("-"*60)
        
        print("\nBASELINE MODEL:")
        baseline_results = {}
        for maxim in maxims:
            probs = np.array(baseline_probs[maxim])
            print(f"\n{maxim.upper()}:")
            print(f"  Mean:   {probs.mean():.3f}")
            print(f"  Std:    {probs.std():.3f}")
            print(f"  Min:    {probs.min():.3f}")
            print(f"  Max:    {probs.max():.3f}")
            print(f"  Range:  {probs.max()-probs.min():.3f}")
            
            # Check compression
            if probs.std() < 0.1:
                print(f"  🚨 CRITICAL: Std dev < 0.1 indicates severe compression!")
            if (probs.max() - probs.min()) < 0.3:
                print(f"  🚨 CRITICAL: Range < 0.3 indicates poor discrimination!")
            
            baseline_results[maxim] = {
                'mean': probs.mean(),
                'std': probs.std(),
                'range': probs.max() - probs.min()
            }
        
        print("\nDPO MODEL:")
        dpo_results = {}
        for maxim in maxims:
            probs = np.array(dpo_probs[maxim])
            baseline_mean = np.array(baseline_probs[maxim]).mean()
            improvement = baseline_mean - probs.mean()
            
            print(f"\n{maxim.upper()}:")
            print(f"  Mean:        {probs.mean():.3f}")
            print(f"  Improvement: {improvement:.3f} ({improvement/baseline_mean*100:+.1f}%)")
            print(f"  Std:         {probs.std():.3f}")
            print(f"  Min:         {probs.min():.3f}")
            print(f"  Max:         {probs.max():.3f}")
            
            dpo_results[maxim] = {
                'mean': probs.mean(),
                'improvement': improvement,
                'improvement_pct': improvement/baseline_mean*100
            }
        
        # Calculate optimal threshold
        print("\n" + "="*60)
        print("OPTIMAL THRESHOLD ANALYSIS")
        print("="*60)
        
        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        threshold_results = {}
        
        for threshold in thresholds:
            print(f"\nThreshold = {threshold}:")
            print("-"*40)
            
            baseline_violations = {}
            dpo_violations = {}
            
            for maxim in maxims:
                b_probs = np.array(baseline_probs[maxim])
                d_probs = np.array(dpo_probs[maxim])
                
                baseline_violations[maxim] = (b_probs > threshold).mean()
                dpo_violations[maxim] = (d_probs > threshold).mean()
                
                if baseline_violations[maxim] > 0:
                    improvement = (baseline_violations[maxim] - dpo_violations[maxim]) / baseline_violations[maxim]
                else:
                    improvement = 0
                
                print(f"  {maxim.capitalize():10s}: "
                      f"Baseline={baseline_violations[maxim]*100:5.1f}% → "
                      f"DPO={dpo_violations[maxim]*100:5.1f}% "
                      f"({improvement*100:+6.1f}%)")
            
            # Calculate cooperative rate
            baseline_coop = np.array([
                all(baseline_probs[m][i] <= threshold for m in maxims)
                for i in range(len(baseline_probs['quantity']))
            ]).mean()
            
            dpo_coop = np.array([
                all(dpo_probs[m][i] <= threshold for m in maxims)
                for i in range(len(dpo_probs['quantity']))
            ]).mean()
            
            print(f"\n  Cooperative: "
                  f"Baseline={baseline_coop*100:5.1f}% → "
                  f"DPO={dpo_coop*100:5.1f}% "
                  f"({(dpo_coop-baseline_coop)*100:+6.1f} pp)")
            
            threshold_results[threshold] = {
                'baseline_violations': baseline_violations,
                'dpo_violations': dpo_violations,
                'baseline_coop': baseline_coop,
                'dpo_coop': dpo_coop
            }
        
        # Create calibration plot
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('Detector Probability Calibration', fontsize=16)
            
            for idx, maxim in enumerate(maxims):
                ax = axes[idx//2, idx%2]
                
                # Plot distributions
                ax.hist(baseline_probs[maxim], bins=30, alpha=0.5, 
                       label='Baseline', edgecolor='black', color='red')
                ax.hist(dpo_probs[maxim], bins=30, alpha=0.5, 
                       label='DPO', edgecolor='black', color='blue')
                
                # Add threshold lines
                ax.axvline(0.5, color='red', linestyle='--', 
                          linewidth=2, label='Current threshold')
                ax.axvline(0.6, color='green', linestyle='--', 
                          linewidth=2, label='Recommended threshold')
                
                ax.set_title(f'{maxim.capitalize()} Violations')
                ax.set_xlabel('Detector Probability')
                ax.set_ylabel('Count')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('detector_calibration.png', dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved calibration plot to detector_calibration.png")
        except Exception as e:
            print(f"⚠️  Could not create plot: {e}")
        
        return {
            'baseline': baseline_results,
            'dpo': dpo_results,
            'thresholds': threshold_results
        }
    
    # ============================================
    # DIAGNOSTIC 3: DPO Training Data Quality
    # ============================================
    
    def analyze_dpo_data_quality(self):
        """Analyze DPO preference pair quality"""
        print("\n" + "="*60)
        print("DIAGNOSTIC 3: DPO TRAINING DATA QUALITY")
        print("="*60)
        
        # Load DPO training data
        dpo_file = self.data_dir / "dpo_data" / "dpo_train.json"
        if not dpo_file.exists():
            print(f"❌ File not found: {dpo_file}")
            return None
            
        with open(dpo_file) as f:
            data = json.load(f)
        
        print(f"\n📊 DPO Training Data Statistics (n={len(data)}):")
        print("-"*60)
        
        # Check data structure
        sample = data[0] if data else {}
        print(f"\nSample keys: {list(sample.keys())}")
        
        # Try to find detector scores in the data
        has_scores = 'chosen_scores' in sample or 'detector_scores' in sample
        
        if not has_scores:
            print("\n⚠️  No detector score information found in DPO data")
            print("   Cannot analyze preference margins without scores")
            print("   This might explain why DPO training had mixed results!")
            return None
        
        # Analyze preference margins if scores exist
        maxims = ['quantity', 'quality', 'relation', 'manner']
        margins = {m: [] for m in maxims}
        
        for item in data:
            # Try different possible key structures
            chosen_scores = item.get('chosen_scores', item.get('chosen', {}).get('detector_scores', {}))
            rejected_scores = item.get('rejected_scores', item.get('rejected', {}).get('detector_scores', {}))
            
            if chosen_scores and rejected_scores:
                for maxim in maxims:
                    chosen = chosen_scores.get(maxim, 0.5)
                    rejected = rejected_scores.get(maxim, 0.5)
                    margin = rejected - chosen  # Positive = chosen is better
                    margins[maxim].append(margin)
        
        if not margins['quantity']:
            print("\n⚠️  Could not extract margins from data structure")
            return None
        
        print("\nPreference Margins (rejected - chosen):")
        print("(Positive = chosen is better, Negative = rejected is better)")
        print("-"*60)
        
        results = {}
        for maxim in maxims:
            m = np.array(margins[maxim])
            
            print(f"\n{maxim.upper()}:")
            print(f"  Mean margin:     {m.mean():.3f}")
            print(f"  Std:             {m.std():.3f}")
            print(f"  Min:             {m.min():.3f}")
            print(f"  Max:             {m.max():.3f}")
            
            # Quality indicators
            positive = (m > 0).mean()
            clear = (np.abs(m) > 0.2).mean()
            unclear = (np.abs(m) < 0.1).mean()
            
            print(f"  % Positive:      {positive*100:.1f}%")
            print(f"  % Clear (>0.2):  {clear*100:.1f}%")
            print(f"  % Unclear(<0.1): {unclear*100:.1f}%")
            
            if unclear > 0.3:
                print(f"  🚨 WARNING: {unclear*100:.1f}% of pairs have unclear preference!")
            if positive < 0.6:
                print(f"  🚨 WARNING: Only {positive*100:.1f}% of pairs prefer chosen!")
            
            results[maxim] = {
                'mean_margin': m.mean(),
                'pct_positive': positive,
                'pct_clear': clear,
                'pct_unclear': unclear
            }
        
        # Create margin distribution plot
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('DPO Preference Margins Distribution', fontsize=16)
            
            for idx, maxim in enumerate(maxims):
                ax = axes[idx//2, idx%2]
                
                ax.hist(margins[maxim], bins=50, edgecolor='black', alpha=0.7)
                ax.axvline(0, color='red', linestyle='--', linewidth=2, 
                          label='No preference')
                ax.axvline(0.1, color='orange', linestyle='--', linewidth=1, 
                          label='Weak preference')
                ax.axvline(-0.1, color='orange', linestyle='--', linewidth=1)
                ax.axvline(0.2, color='green', linestyle='--', linewidth=1, 
                          label='Clear preference')
                ax.axvline(-0.2, color='green', linestyle='--', linewidth=1)
                
                ax.set_title(f'{maxim.capitalize()} Preference Margins')
                ax.set_xlabel('Margin (rejected - chosen)')
                ax.set_ylabel('Count')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('dpo_preference_margins.png', dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved margin plot to dpo_preference_margins.png")
        except Exception as e:
            print(f"⚠️  Could not create plot: {e}")
        
        return results
    
    # ============================================
    # DIAGNOSTIC 4: Baseline Response Quality
    # ============================================
    
    def analyze_baseline_quality(self):
        """Analyze baseline model response quality"""
        print("\n" + "="*60)
        print("DIAGNOSTIC 4: BASELINE RESPONSE QUALITY")
        print("="*60)
        
        results_file = self.results_dir / "dpo_evaluation_results.json"
        if not results_file.exists():
            print(f"❌ File not found: {results_file}")
            return None
            
        with open(results_file) as f:
            data = json.load(f)
        
        # Analyze baseline responses
        artifacts = ['FS1', 'Evidence:', 'Generate a cooperative response', 
                    '[agent_', 'FS2', 'FS3', 'FS4', 'FS5']
        
        artifact_counts = Counter()
        repetition_counts = 0
        total_responses = len(data['examples'])
        
        print(f"\n📊 Baseline Response Analysis (n={total_responses}):")
        print("-"*60)
        
        for item in data['examples']:
            response = item['baseline_response']
            
            # Check for artifacts
            for artifact in artifacts:
                if artifact in response:
                    artifact_counts[artifact] += 1
            
            # Check for repetition (same phrase 3+ times)
            words = response.split()
            for i in range(len(words) - 6):
                phrase = ' '.join(words[i:i+3])
                if response.count(phrase) >= 3:
                    repetition_counts += 1
                    break
        
        print("\n🔍 Training Artifact Detection:")
        for artifact, count in artifact_counts.most_common():
            pct = count / total_responses * 100
            print(f"  '{artifact}': {count} ({pct:.1f}%)")
            if pct > 10:
                print(f"    🚨 CRITICAL: {pct:.1f}% of responses contain this artifact!")
        
        print(f"\n🔁 Repetition Issues:")
        rep_pct = repetition_counts / total_responses * 100
        print(f"  Repetitive responses: {repetition_counts} ({rep_pct:.1f}%)")
        if rep_pct > 15:
            print(f"    🚨 WARNING: {rep_pct:.1f}% of responses are repetitive!")
        
        # Sample problematic responses
        print("\n📝 Sample Problematic Responses:")
        print("-"*60)
        
        shown = 0
        for idx, item in enumerate(data['examples']):
            response = item['baseline_response']
            
            if any(artifact in response for artifact in artifacts[:3]):
                print(f"\nExample {idx+1} (Artifact contamination):")
                print(f"  Prompt: {item['prompt'][:100]}...")
                print(f"  Response: {response[:200]}...")
                shown += 1
                if shown >= 3:
                    break
        
        return {
            'artifact_counts': dict(artifact_counts),
            'repetition_pct': rep_pct,
            'total_responses': total_responses
        }
    
    # ============================================
    # MASTER RUN
    # ============================================
    
    def run_all_diagnostics(self):
        """Run complete diagnostic suite"""
        print("\n" + "="*60)
        print("GRICEBENCH COMPREHENSIVE DIAGNOSTIC SUITE")
        print("="*60)
        print("\nThis will analyze your entire system to identify root causes.")
        print("Generating detailed reports and visualizations...\n")
        
        results = {}
        
        results['weak_labels'] = self.analyze_weak_labels()
        results['detector_calibration'] = self.analyze_detector_calibration()
        results['dpo_quality'] = self.analyze_dpo_data_quality()
        results['baseline_quality'] = self.analyze_baseline_quality()
        
        print("\n" + "="*60)
        print("DIAGNOSTIC SUITE COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        print("  📊 weak_label_distribution.png")
        print("  📊 detector_calibration.png")
        print("  📊 dpo_preference_margins.png")
        print("\nReview the output above for critical warnings (🚨)")
        print("="*60)
        
        # Save results to JSON
        results_clean = {}
        for key, val in results.items():
            if val is not None:
                results_clean[key] = val
        
        with open('diagnostic_results.json', 'w') as f:
            json.dump(results_clean, f, indent=2, default=str)
        print("\n💾 Saved detailed results to diagnostic_results.json")
        
        return results

# Usage
if __name__ == "__main__":
    diagnostics = GriceBenchDiagnostics(
        data_dir="data_processed",
        results_dir="."
    )
    results = diagnostics.run_all_diagnostics()
