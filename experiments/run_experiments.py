# experiments/run_experiments.py
# Main Experiment Runner for Process Supervision Evaluation

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ModelConfig, ExperimentConfig
from prompts import get_all_prompts, get_prompts_by_category, get_categories, Prompt
from student import StudentModel, StudentResponse
from observer import ObserverModel, ObserverDecision


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    prompt_id: str
    category: str
    task: str
    expected_behavior: str
    
    # Student results
    student_reasoning: str
    student_full_response: str
    student_tokens: int
    student_time: float
    
    # Observer results  
    observer_decision: str
    observer_confidence: float
    observer_failure_type: str
    observer_comment: str
    observer_time: float
    
    # Evaluation
    decision_correct: bool
    tokens_saved: int  # Compared to full generation
    
    # Metadata
    oip: int
    timestamp: str
    student_model: str
    observer_model: str


class ExperimentRunner:
    """
    Main experiment runner for process supervision evaluation.
    """
    
    def __init__(
        self, 
        model_config: Optional[ModelConfig] = None,
        experiment_config: Optional[ExperimentConfig] = None
    ):
        self.model_config = model_config or ModelConfig()
        self.exp_config = experiment_config or ExperimentConfig()
        
        self.student = StudentModel(self.model_config)
        self.observer = ObserverModel(self.model_config)
        
        # Create output directories
        Path(self.exp_config.logs_dir).mkdir(exist_ok=True)
        Path(self.exp_config.results_dir).mkdir(exist_ok=True)
        
        self.results: List[ExperimentResult] = []
    
    def run_single(self, prompt: Prompt, oip: int = None) -> ExperimentResult:
        """Run a single experiment."""
        
        oip = oip or self.exp_config.default_oip
        
        print(f"\n{'='*60}")
        print(f"Running: {prompt.id} ({prompt.category})")
        print(f"Task: {prompt.task[:80]}...")
        print(f"Expected: {prompt.expected_behavior.upper()}")
        
        # Step 1: Generate reasoning with Student
        print("\n[Student] Generating reasoning...")
        student_response = self.student.generate(prompt.task, oip=oip)
        
        print(f"  Tokens: {student_response.tokens_used}")
        print(f"  Time: {student_response.generation_time:.2f}s")
        print(f"  Truncated at OIP: {student_response.truncated_at_oip}")
        
        # Step 2: Evaluate with Observer
        print("\n[Observer] Evaluating reasoning...")
        observer_decision = self.observer.evaluate(
            student_response.reasoning_segment,
            prompt.task
        )
        
        print(f"  Decision: {observer_decision.decision}")
        print(f"  Confidence: {observer_decision.confidence:.2f}")
        print(f"  Failure Type: {observer_decision.failure_type}")
        
        # Step 3: Evaluate correctness
        expected_stop = prompt.expected_behavior == "stop"
        actual_stop = observer_decision.decision == "STOP"
        decision_correct = expected_stop == actual_stop
        
        # Calculate tokens saved (if stopped early)
        full_tokens = self.model_config.max_tokens_student
        tokens_saved = full_tokens - student_response.tokens_used if actual_stop else 0
        
        print(f"\n[Result] Correct: {decision_correct}")
        
        result = ExperimentResult(
            prompt_id=prompt.id,
            category=prompt.category,
            task=prompt.task,
            expected_behavior=prompt.expected_behavior,
            student_reasoning=student_response.reasoning_segment,
            student_full_response=student_response.full_response,
            student_tokens=student_response.tokens_used,
            student_time=student_response.generation_time,
            observer_decision=observer_decision.decision,
            observer_confidence=observer_decision.confidence,
            observer_failure_type=observer_decision.failure_type,
            observer_comment=observer_decision.comment,
            observer_time=observer_decision.evaluation_time,
            decision_correct=decision_correct,
            tokens_saved=tokens_saved,
            oip=oip,
            timestamp=datetime.now().isoformat(),
            student_model=student_response.model,
            observer_model=observer_decision.model
        )
        
        self.results.append(result)
        return result
    
    def run_category(self, category: str, oip: int = None) -> List[ExperimentResult]:
        """Run all experiments for a category."""
        prompts = get_prompts_by_category(category)
        results = []
        
        print(f"\n{'#'*60}")
        print(f"# CATEGORY: {category.upper()}")
        print(f"# Prompts: {len(prompts)}")
        print(f"{'#'*60}")
        
        for prompt in prompts:
            result = self.run_single(prompt, oip)
            results.append(result)
            time.sleep(0.5)  # Rate limiting
        
        return results
    
    def run_all(self, oip: int = None) -> List[ExperimentResult]:
        """Run all 40 experiments."""
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS")
        print(f"Total prompts: 40")
        print(f"OIP: {oip or self.exp_config.default_oip} tokens")
        print("="*60)
        
        for category in get_categories():
            self.run_category(category, oip)
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        filepath = Path(self.exp_config.results_dir) / filename
        
        results_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_experiments": len(self.results),
                "student_model": self.model_config.student_model,
                "observer_model": self.model_config.observer_model,
                "oip": self.exp_config.default_oip,
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        # Overall accuracy
        correct = sum(1 for r in self.results if r.decision_correct)
        total = len(self.results)
        print(f"\nOverall Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        
        # By category
        print("\nBy Category:")
        print("-" * 50)
        
        for category in get_categories():
            cat_results = [r for r in self.results if r.category == category]
            if not cat_results:
                continue
            
            cat_correct = sum(1 for r in cat_results if r.decision_correct)
            cat_total = len(cat_results)
            
            stop_count = sum(1 for r in cat_results if r.observer_decision == "STOP")
            proceed_count = cat_total - stop_count
            
            avg_tokens = sum(r.student_tokens for r in cat_results) / cat_total
            
            print(f"\n{category.upper()}:")
            print(f"  Accuracy: {cat_correct}/{cat_total} ({100*cat_correct/cat_total:.1f}%)")
            print(f"  STOP: {stop_count}, PROCEED: {proceed_count}")
            print(f"  Avg Tokens: {avg_tokens:.0f}")
        
        # Token efficiency
        total_tokens = sum(r.student_tokens for r in self.results)
        max_possible = len(self.results) * self.model_config.max_tokens_student
        tokens_saved = max_possible - total_tokens
        
        print(f"\nToken Efficiency:")
        print(f"  Total tokens used: {total_tokens}")
        print(f"  Max possible: {max_possible}")
        print(f"  Tokens saved: {tokens_saved} ({100*tokens_saved/max_possible:.1f}%)")
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Process Supervision Experiments")
    parser.add_argument("--category", type=str, help="Run specific category only")
    parser.add_argument("--oip", type=int, default=150, help="Optimal Intervention Point (tokens)")
    parser.add_argument("--output", type=str, help="Output filename for results")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner()
    
    # Run experiments
    if args.category:
        runner.run_category(args.category, oip=args.oip)
    else:
        runner.run_all(oip=args.oip)
    
    # Save and summarize
    runner.save_results(args.output)
    runner.print_summary()


if __name__ == "__main__":
    main()
