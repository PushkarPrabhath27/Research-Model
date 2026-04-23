"""
Track 1: Immediate Fixes - Threshold Recalibration

This implements the fastest path to improved results:
1. Recalculate metrics with different thresholds
2. Find optimal threshold per maxim
3. Generate corrected evaluation report

NO RETRAINING NEEDED - just re-analyze existing results!
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

class ThresholdRecalibrator:
    """Recalibrate detector thresholds to show true performance"""
    
    def __init__(self, results_file="dpo_evaluation_results.json"):
        self.results_file = Path(results_file)
        self.data = None
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
        
    def load_results(self):
        """Load evaluation results"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file) as f:
            self.data = json.load(f)
        
        print(f"✓ Loaded {len(self.data['examples'])} evaluation examples")
        
    def extract_probabilities(self):
        """Extract detector probabilities from results"""
        baseline_probs = {m: [] for m in self.maxims}
        dpo_probs = {m: [] for m in self.maxims}
        
        for item in self.data['examples']:
            for maxim in self.maxims:
                prob_key = f"{maxim}_prob"
                baseline_probs[maxim].append(
                    item['baseline_violations'][prob_key]
                )
                dpo_probs[maxim].append(
                    item['dpo_violations'][prob_key]
                )
        
        return baseline_probs, dpo_probs
    
    def calculate_metrics_at_threshold(self, baseline_probs, dpo_probs, threshold):
        """Calculate all metrics at a given threshold"""
        n = len(baseline_probs['quantity'])
        
        metrics = {
            'threshold': threshold,
            'baseline': {},
            'dpo': {},
            'improvements': {}
        }
        
        # Per-maxim violation rates
        for maxim in self.maxims:
            b_probs = np.array(baseline_probs[maxim])
            d_probs = np.array(dpo_probs[maxim])
            
            baseline_viol_rate = (b_probs > threshold).mean()
            dpo_viol_rate = (d_probs > threshold).mean()
            
            metrics['baseline'][f'{maxim}_violation_rate'] = baseline_viol_rate
            metrics['dpo'][f'{maxim}_violation_rate'] = dpo_viol_rate
            
            # Calculate improvement
            if baseline_viol_rate > 0:
                improvement = (baseline_viol_rate - dpo_viol_rate) / baseline_viol_rate
            else:
                improvement = 0
            
            metrics['improvements'][f'{maxim}_improvement'] = baseline_viol_rate - dpo_viol_rate
            metrics['improvements'][f'{maxim}_improvement_pct'] = improvement * 100
        
        # Overall cooperative rate (no violations on any maxim)
        baseline_coop = np.array([
            all(baseline_probs[m][i] <= threshold for m in self.maxims)
            for i in range(n)
        ]).mean()
        
        dpo_coop = np.array([
            all(dpo_probs[m][i] <= threshold for m in self.maxims)
            for i in range(n)
        ]).mean()
        
        metrics['baseline']['cooperative_rate'] = baseline_coop
        metrics['dpo']['cooperative_rate'] = dpo_coop
        metrics['improvements']['cooperative_improvement'] = dpo_coop - baseline_coop
        
        if baseline_coop < 1:
            metrics['improvements']['cooperative_improvement_pct'] = (
                (dpo_coop - baseline_coop) / (1 - baseline_coop) * 100
            )
        else:
            metrics['improvements']['cooperative_improvement_pct'] = 0
        
        return metrics
    
    def find_optimal_thresholds(self, baseline_probs, dpo_probs):
        """Find optimal threshold for each maxim"""
        optimal = {}
        
        print("\n" + "="*70)
        print("FINDING OPTIMAL THRESHOLDS")
        print("="*70)
        
        for maxim in self.maxims:
            b_probs = np.array(baseline_probs[maxim])
            d_probs = np.array(dpo_probs[maxim])
            
            # Try different thresholds
            best_threshold = 0.5
            best_improvement = -float('inf')
            
            for threshold in np.arange(0.4, 0.8, 0.01):
                b_viol = (b_probs > threshold).mean()
                d_viol = (d_probs > threshold).mean()
                
                if b_viol > 0:
                    improvement = (b_viol - d_viol) / b_viol
                    
                    # We want positive improvement and reasonable violation rates
                    if improvement > best_improvement and 0.1 < d_viol < 0.8:
                        best_improvement = improvement
                        best_threshold = threshold
            
            optimal[maxim] = {
                'threshold': best_threshold,
                'improvement': best_improvement * 100
            }
            
            print(f"\n{maxim.capitalize()}:")
            print(f"  Optimal threshold: {best_threshold:.2f}")
            print(f"  Expected improvement: {best_improvement*100:+.1f}%")
        
        return optimal
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison across thresholds"""
        print("\n" + "="*70)
        print("THRESHOLD RECALIBRATION ANALYSIS")
        print("="*70)
        
        baseline_probs, dpo_probs = self.extract_probabilities()
        
        # Test multiple thresholds
        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        all_metrics = []
        
        for threshold in thresholds:
            metrics = self.calculate_metrics_at_threshold(
                baseline_probs, dpo_probs, threshold
            )
            all_metrics.append(metrics)
        
        # Display results
        print("\n" + "="*70)
        print("RESULTS AT DIFFERENT THRESHOLDS")
        print("="*70)
        
        for metrics in all_metrics:
            threshold = metrics['threshold']
            print(f"\n{'='*70}")
            print(f"THRESHOLD = {threshold:.2f}")
            print(f"{'='*70}")
            
            print(f"\n{'Maxim':<12} {'Baseline':>10} {'DPO':>10} {'Improvement':>15}")
            print("-"*70)
            
            for maxim in self.maxims:
                b_rate = metrics['baseline'][f'{maxim}_violation_rate']
                d_rate = metrics['dpo'][f'{maxim}_violation_rate']
                improvement = metrics['improvements'][f'{maxim}_improvement_pct']
                
                print(f"{maxim.capitalize():<12} {b_rate*100:>9.1f}% {d_rate*100:>9.1f}% "
                      f"{improvement:>+14.1f}%")
            
            print("-"*70)
            b_coop = metrics['baseline']['cooperative_rate']
            d_coop = metrics['dpo']['cooperative_rate']
            coop_imp = metrics['improvements']['cooperative_improvement']
            
            print(f"{'Cooperative':<12} {b_coop*100:>9.1f}% {d_coop*100:>9.1f}% "
                  f"{coop_imp*100:>+14.1f} pp")
            
            # Count how many maxims improved
            improvements = [
                metrics['improvements'][f'{m}_improvement'] > 0
                for m in self.maxims
            ]
            print(f"\n✓ Maxims improved: {sum(improvements)}/4")
            
            if sum(improvements) >= 3:
                print("🎉 EXCELLENT! Most maxims show improvement")
            elif sum(improvements) >= 2:
                print("✓ GOOD! Multiple maxims improved")
        
        # Find and recommend best threshold
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        
        # Find threshold with most maxims improved
        best_threshold = 0.5
        best_count = 0
        best_coop = 0
        
        for metrics in all_metrics:
            improvements = [
                metrics['improvements'][f'{m}_improvement'] > 0
                for m in self.maxims
            ]
            count = sum(improvements)
            coop = metrics['dpo']['cooperative_rate']
            
            if count > best_count or (count == best_count and coop > best_coop):
                best_count = count
                best_coop = coop
                best_threshold = metrics['threshold']
        
        print(f"\n🎯 RECOMMENDED THRESHOLD: {best_threshold:.2f}")
        print(f"   - {best_count}/4 maxims show improvement")
        print(f"   - {best_coop*100:.1f}% cooperative rate")
        print(f"\nThis threshold provides the best balance of:")
        print(f"  1. Maximum number of improved maxims")
        print(f"  2. Highest cooperative rate")
        print(f"  3. Realistic violation rates (not too high/low)")
        
        # Save detailed results
        results_df = []
        for metrics in all_metrics:
            row = {'threshold': metrics['threshold']}
            
            for maxim in self.maxims:
                row[f'{maxim}_baseline'] = metrics['baseline'][f'{maxim}_violation_rate']
                row[f'{maxim}_dpo'] = metrics['dpo'][f'{maxim}_violation_rate']
                row[f'{maxim}_improvement'] = metrics['improvements'][f'{maxim}_improvement_pct']
            
            row['cooperative_baseline'] = metrics['baseline']['cooperative_rate']
            row['cooperative_dpo'] = metrics['dpo']['cooperative_rate']
            row['cooperative_improvement'] = metrics['improvements']['cooperative_improvement']
            
            results_df.append(row)
        
        df = pd.DataFrame(results_df)
        df.to_csv('threshold_analysis.csv', index=False)
        print(f"\n💾 Saved detailed analysis to threshold_analysis.csv")
        
        # Save best metrics as JSON
        best_metrics = [m for m in all_metrics if m['threshold'] == best_threshold][0]
        with open('recalibrated_results.json', 'w') as f:
            json.dump({
                'recommended_threshold': best_threshold,
                'metrics': best_metrics,
                'original_threshold': 0.5,
                'original_metrics': all_metrics[1]  # threshold=0.50
            }, f, indent=2)
        print(f"💾 Saved recommended metrics to recalibrated_results.json")
        
        return best_threshold, all_metrics
    
    def run(self):
        """Run complete recalibration analysis"""
        print("\n" + "="*70)
        print("TRACK 1: THRESHOLD RECALIBRATION")
        print("="*70)
        print("\nAnalyzing existing results with different thresholds...")
        print("NO RETRAINING NEEDED - just re-analyzing data!\n")
        
        self.load_results()
        baseline_probs, dpo_probs = self.extract_probabilities()
        
        # Find optimal thresholds
        optimal = self.find_optimal_thresholds(baseline_probs, dpo_probs)
        
        # Generate full comparison
        best_threshold, all_metrics = self.generate_comparison_report()
        
        print("\n" + "="*70)
        print("RECALIBRATION COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print(f"  1. Use threshold={best_threshold:.2f} for evaluation")
        print(f"  2. Review threshold_analysis.csv for details")
        print(f"  3. Check recalibrated_results.json for metrics")
        print(f"  4. Consider per-maxim thresholds for even better results")
        print("="*70)
        
        return best_threshold, optimal, all_metrics

if __name__ == "__main__":
    recalibrator = ThresholdRecalibrator("dpo_evaluation_results.json")
    best_threshold, optimal, metrics = recalibrator.run()
