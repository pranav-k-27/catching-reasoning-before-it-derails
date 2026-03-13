# experiments/run_ablation.py
# OIP Ablation Study for Process Supervision

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ModelConfig, ExperimentConfig
from prompts import get_all_prompts, get_categories
from experiments.run_experiments import ExperimentRunner, ExperimentResult


@dataclass
class AblationResult:
    """Results for a single OIP value."""
    oip: int
    total_experiments: int
    
    # Accuracy metrics
    true_positives: int   # Correctly stopped (expected stop, got stop)
    true_negatives: int   # Correctly proceeded (expected proceed, got proceed)
    false_positives: int  # Incorrectly stopped (expected proceed, got stop)
    false_negatives: int  # Incorrectly proceeded (expected stop, got proceed)
    
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    # Efficiency metrics
    avg_tokens_used: float
    avg_tokens_saved: float
    tokens_saved_percent: float
    
    # Timing
    avg_student_time: float
    avg_observer_time: float


class AblationRunner:
    """
    Runner for OIP ablation study.
    
    Tests multiple OIP values to find optimal intervention point.
    """
    
    def __init__(
        self, 
        model_config: ModelConfig = None,
        experiment_config: ExperimentConfig = None
    ):
        self.model_config = model_config or ModelConfig()
        self.exp_config = experiment_config or ExperimentConfig()
        
        self.ablation_results: Dict[int, AblationResult] = {}
        self.all_results: Dict[int, List[ExperimentResult]] = {}
    
    def compute_metrics(self, results: List[ExperimentResult], oip: int) -> AblationResult:
        """Compute evaluation metrics for a set of results."""
        
        total = len(results)
        
        # Count outcomes
        tp = sum(1 for r in results 
                 if r.expected_behavior == "stop" and r.observer_decision == "STOP")
        tn = sum(1 for r in results 
                 if r.expected_behavior == "proceed" and r.observer_decision == "PROCEED")
        fp = sum(1 for r in results 
                 if r.expected_behavior == "proceed" and r.observer_decision == "STOP")
        fn = sum(1 for r in results 
                 if r.expected_behavior == "stop" and r.observer_decision == "PROCEED")
        
        # Calculate metrics (with safety for division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # Efficiency metrics
        avg_tokens = sum(r.student_tokens for r in results) / total if total > 0 else 0
        max_tokens = self.model_config.max_tokens_student
        avg_saved = max_tokens - avg_tokens
        saved_percent = 100 * avg_saved / max_tokens if max_tokens > 0 else 0
        
        # Timing
        avg_student_time = sum(r.student_time for r in results) / total if total > 0 else 0
        avg_observer_time = sum(r.observer_time for r in results) / total if total > 0 else 0
        
        return AblationResult(
            oip=oip,
            total_experiments=total,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            avg_tokens_used=avg_tokens,
            avg_tokens_saved=avg_saved,
            tokens_saved_percent=saved_percent,
            avg_student_time=avg_student_time,
            avg_observer_time=avg_observer_time
        )
    
    def run_ablation(self, oip_values: List[int] = None) -> Dict[int, AblationResult]:
        """
        Run ablation study across multiple OIP values.
        
        Args:
            oip_values: List of OIP values to test. Defaults to [100, 150, 200]
        """
        if oip_values is None:
            oip_values = list(self.exp_config.oip_ablation_values)
        
        print("\n" + "="*60)
        print("OIP ABLATION STUDY")
        print(f"Testing OIP values: {oip_values}")
        print("="*60)
        
        for oip in oip_values:
            print(f"\n{'#'*60}")
            print(f"# OIP = {oip} tokens")
            print(f"{'#'*60}")
            
            # Run experiments with this OIP
            runner = ExperimentRunner(self.model_config, self.exp_config)
            results = runner.run_all(oip=oip)
            
            # Compute metrics
            ablation_result = self.compute_metrics(results, oip)
            
            self.ablation_results[oip] = ablation_result
            self.all_results[oip] = results
            
            # Print intermediate results
            print(f"\nOIP={oip} Results:")
            print(f"  Precision: {ablation_result.precision:.3f}")
            print(f"  Recall: {ablation_result.recall:.3f}")
            print(f"  F1 Score: {ablation_result.f1_score:.3f}")
            print(f"  Tokens Saved: {ablation_result.tokens_saved_percent:.1f}%")
        
        return self.ablation_results
    
    def save_results(self, filename: str = None):
        """Save ablation results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ablation_results_{timestamp}.json"
        
        filepath = Path(self.exp_config.results_dir) / filename
        
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "oip_values": list(self.ablation_results.keys()),
                "student_model": self.model_config.student_model,
                "observer_model": self.model_config.observer_model,
            },
            "ablation_summary": {
                str(oip): asdict(result) 
                for oip, result in self.ablation_results.items()
            },
            "detailed_results": {
                str(oip): [asdict(r) for r in results]
                for oip, results in self.all_results.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nAblation results saved to: {filepath}")
        return filepath
    
    def print_summary_table(self):
        """Print summary table for paper."""
        if not self.ablation_results:
            print("No ablation results to display.")
            return
        
        print("\n" + "="*70)
        print("ABLATION STUDY SUMMARY")
        print("="*70)
        print("\nTable: Effect of Optimal Intervention Point (OIP) on Detection Performance")
        print("-"*70)
        print(f"{'OIP':>6} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Tokens Saved':>12}")
        print("-"*70)
        
        for oip in sorted(self.ablation_results.keys()):
            r = self.ablation_results[oip]
            print(f"{oip:>6} | {r.precision:>10.2f} | {r.recall:>10.2f} | {r.f1_score:>10.2f} | {r.tokens_saved_percent:>11.1f}%")
        
        print("-"*70)
        
        # Additional details
        print("\nDetailed Breakdown:")
        print("-"*70)
        print(f"{'OIP':>6} | {'TP':>5} | {'TN':>5} | {'FP':>5} | {'FN':>5} | {'Accuracy':>10}")
        print("-"*70)
        
        for oip in sorted(self.ablation_results.keys()):
            r = self.ablation_results[oip]
            print(f"{oip:>6} | {r.true_positives:>5} | {r.true_negatives:>5} | {r.false_positives:>5} | {r.false_negatives:>5} | {r.accuracy:>10.2f}")
        
        print("-"*70)
        print("\nLegend:")
        print("  TP = True Positive (correctly stopped flawed reasoning)")
        print("  TN = True Negative (correctly proceeded with valid reasoning)")
        print("  FP = False Positive (incorrectly stopped valid reasoning)")
        print("  FN = False Negative (missed flawed reasoning)")
        print("="*70)


def main():
    """Main entry point for ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OIP Ablation Study")
    parser.add_argument("--oip-values", type=int, nargs="+", default=[100, 150, 200],
                        help="OIP values to test (default: 100 150 200)")
    parser.add_argument("--output", type=str, help="Output filename")
    
    args = parser.parse_args()
    
    # Run ablation
    runner = AblationRunner()
    runner.run_ablation(args.oip_values)
    
    # Save and display results
    runner.save_results(args.output)
    runner.print_summary_table()


if __name__ == "__main__":
    main()
