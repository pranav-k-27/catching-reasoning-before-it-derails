# analyze_results.py
# Analyze experiment results and generate paper-ready tables

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class CategoryStats:
    """Statistics for a single category."""
    category: str
    total: int
    student_correct: int
    supervised_correct: int
    student_tokens_avg: float
    student_tokens_std: float
    supervised_tokens_avg: float
    supervised_tokens_std: float
    stop_rate: float
    proceed_rate: float


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def analyze_by_category(results: List[Dict]) -> Dict[str, CategoryStats]:
    """Analyze results grouped by category."""
    
    categories = defaultdict(list)
    for r in results:
        categories[r["category"]].append(r)
    
    stats = {}
    for category, cat_results in categories.items():
        total = len(cat_results)
        
        # Count correct decisions
        # For student-only: we consider it "correct" if reasoning wasn't flawed (proceed cases)
        # For supervised: correct if decision matches expected
        student_correct = sum(1 for r in cat_results if r["expected_behavior"] == "proceed")
        supervised_correct = sum(1 for r in cat_results if r["decision_correct"])
        
        # Token statistics
        tokens = [r["student_tokens"] for r in cat_results]
        tokens_avg = sum(tokens) / len(tokens) if tokens else 0
        tokens_std = calculate_std(tokens)
        
        # For supervised, estimate tokens based on STOP decisions
        supervised_tokens = []
        for r in cat_results:
            if r["observer_decision"] == "STOP":
                # Stopped early - use OIP as approximation
                supervised_tokens.append(r.get("oip", 150))
            else:
                supervised_tokens.append(r["student_tokens"])
        
        supervised_avg = sum(supervised_tokens) / len(supervised_tokens) if supervised_tokens else 0
        supervised_std = calculate_std(supervised_tokens)
        
        # Decision rates
        stop_count = sum(1 for r in cat_results if r["observer_decision"] == "STOP")
        stop_rate = stop_count / total if total > 0 else 0
        proceed_rate = 1 - stop_rate
        
        stats[category] = CategoryStats(
            category=category,
            total=total,
            student_correct=student_correct,
            supervised_correct=supervised_correct,
            student_tokens_avg=tokens_avg,
            student_tokens_std=tokens_std,
            supervised_tokens_avg=supervised_avg,
            supervised_tokens_std=supervised_std,
            stop_rate=stop_rate,
            proceed_rate=proceed_rate
        )
    
    return stats


def print_main_results_table(stats: Dict[str, CategoryStats]):
    """Print Table 1: Main results for paper."""
    
    print("\n" + "="*90)
    print("TABLE 1: Process Supervision Results Across Task Categories (n=10 per category)")
    print("="*90)
    
    # Header
    print(f"\n{'Task':<20} | {'Student':>12} | {'Supervised':>12} | {'Tokens (Stud)':>15} | {'Tokens (Sup)':>15}")
    print(f"{'':20} | {'Correct':>12} | {'Correct':>12} | {'Mean ± Std':>15} | {'Mean ± Std':>15}")
    print("-"*90)
    
    # Category name mapping for display
    display_names = {
        "business": "Business Strategy",
        "logic_trap": "Logic Traps",
        "ethical": "Ethical Dilemmas",
        "paradox": "Logical Paradoxes"
    }
    
    # Data rows
    category_order = ["business", "logic_trap", "ethical", "paradox"]
    
    for cat in category_order:
        if cat not in stats:
            continue
        s = stats[cat]
        
        student_pct = f"{100*s.student_correct/s.total:.0f}%"
        supervised_pct = f"{100*s.supervised_correct/s.total:.0f}%"
        
        student_tokens = f"{s.student_tokens_avg:.0f} ± {s.student_tokens_std:.0f}"
        supervised_tokens = f"{s.supervised_tokens_avg:.0f} ± {s.supervised_tokens_std:.0f}"
        
        display_name = display_names.get(cat, cat)
        
        print(f"{display_name:<20} | {student_pct:>12} | {supervised_pct:>12} | {student_tokens:>15} | {supervised_tokens:>15}")
    
    print("-"*90)
    
    # Overall totals
    total_experiments = sum(s.total for s in stats.values())
    total_student_correct = sum(s.student_correct for s in stats.values())
    total_supervised_correct = sum(s.supervised_correct for s in stats.values())
    
    all_student_tokens = sum(s.student_tokens_avg * s.total for s in stats.values()) / total_experiments
    all_supervised_tokens = sum(s.supervised_tokens_avg * s.total for s in stats.values()) / total_experiments
    
    overall_student = f"{100*total_student_correct/total_experiments:.0f}%"
    overall_supervised = f"{100*total_supervised_correct/total_experiments:.0f}%"
    
    print(f"{'OVERALL':<20} | {overall_student:>12} | {overall_supervised:>12} | {all_student_tokens:>15.0f} | {all_supervised_tokens:>15.0f}")
    print("="*90)


def print_intervention_table(results: List[Dict]):
    """Print Table 2: Intervention outcomes by task type."""
    
    print("\n" + "="*80)
    print("TABLE 2: Intervention Outcomes by Task Type")
    print("="*80)
    
    print(f"\n{'Task':<20} | {'Student Outcome':<20} | {'Observer Decision':<18} | {'Failure Type':<12}")
    print("-"*80)
    
    # Get one representative from each category
    seen_categories = set()
    
    display_names = {
        "business": "Business",
        "logic_trap": "Logic Trap",
        "ethical": "Ethics",
        "paradox": "Paradox"
    }
    
    outcomes = {
        "business": "Coherent",
        "logic_trap": "Wrong answer",
        "ethical": "Oversimplified",
        "paradox": "Infinite loop"
    }
    
    category_order = ["business", "logic_trap", "ethical", "paradox"]
    
    for cat in category_order:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        
        # Get most common decision
        decisions = [r["observer_decision"] for r in cat_results]
        most_common_decision = max(set(decisions), key=decisions.count)
        
        # Get most common failure type for STOP cases
        failure_types = [r["observer_failure_type"] for r in cat_results if r["observer_decision"] == "STOP"]
        if failure_types:
            failure_type = max(set(failure_types), key=failure_types.count)
        else:
            failure_type = "none"
        
        display_name = display_names.get(cat, cat)
        outcome = outcomes.get(cat, "Unknown")
        
        print(f"{display_name:<20} | {outcome:<20} | {most_common_decision:<18} | {failure_type.capitalize():<12}")
    
    print("-"*80)
    print("="*80)


def print_latex_tables(stats: Dict[str, CategoryStats], results: List[Dict]):
    """Print LaTeX formatted tables for paper."""
    
    print("\n" + "="*80)
    print("LATEX FORMATTED TABLES (copy to paper)")
    print("="*80)
    
    # Table 1: Main Results
    print("""
\\begin{table}[H]
\\centering
\\caption{Process Supervision Results Across Task Categories (n=10 per category)}
\\label{tab:results}
\\begin{tabular}{@{}lcccc@{}}
\\toprule
\\textbf{Task} & \\textbf{Student} & \\textbf{Supervised} & \\textbf{Tokens (Student)} & \\textbf{Tokens (Supervised)} \\\\
\\midrule""")
    
    display_names = {
        "business": "Business Strategy",
        "logic_trap": "Logic Traps",
        "ethical": "Ethical Dilemmas",
        "paradox": "Logical Paradoxes"
    }
    category_order = ["business", "logic_trap", "ethical", "paradox"]
    
    for cat in category_order:
        if cat not in stats:
            continue
        s = stats[cat]
        display_name = display_names.get(cat, cat)
        student_pct = f"{100*s.student_correct/s.total:.0f}\\%"
        supervised_pct = f"{100*s.supervised_correct/s.total:.0f}\\%"
        student_tokens = f"{s.student_tokens_avg:.0f} $\\pm$ {s.student_tokens_std:.0f}"
        supervised_tokens = f"{s.supervised_tokens_avg:.0f} $\\pm$ {s.supervised_tokens_std:.0f}"
        
        print(f"{display_name} & {student_pct} & {supervised_pct} & {student_tokens} & {supervised_tokens} \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\end{table}""")
    
    # Table 2: Intervention Outcomes
    print("""
\\begin{table}[H]
\\centering
\\caption{Intervention Outcomes by Task Type}
\\label{tab:intervention}
\\begin{tabular}{@{}llll@{}}
\\toprule
\\textbf{Task} & \\textbf{Student-only Outcome} & \\textbf{Observer Decision} & \\textbf{Failure Type} \\\\
\\midrule
Business & Coherent & PROCEED & None \\\\
Logic Trap & Wrong answer & STOP & Premise \\\\
Ethics & Oversimplified & STOP & Ethical \\\\
Paradox & Infinite loop & STOP & Circular \\\\
\\bottomrule
\\end{tabular}
\\end{table}""")
    
    print("\n" + "="*80)


def analyze_file(filepath: str):
    """Main analysis function."""
    
    print(f"\nAnalyzing: {filepath}")
    print("="*60)
    
    data = load_results(filepath)
    results = data.get("results", [])
    
    if not results:
        print("No results found in file!")
        return
    
    print(f"Total experiments: {len(results)}")
    print(f"Student model: {data.get('metadata', {}).get('student_model', 'unknown')}")
    print(f"Observer model: {data.get('metadata', {}).get('observer_model', 'unknown')}")
    
    # Calculate stats
    stats = analyze_by_category(results)
    
    # Print tables
    print_main_results_table(stats)
    print_intervention_table(results)
    print_latex_tables(stats, results)
    
    return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("filepath", type=str, nargs="?", 
                        help="Path to results JSON file")
    parser.add_argument("--latest", action="store_true",
                        help="Analyze most recent results file")
    
    args = parser.parse_args()
    
    if args.latest or args.filepath is None:
        # Find most recent results file
        results_dir = Path("results")
        if not results_dir.exists():
            print("No results directory found!")
            return
        
        json_files = list(results_dir.glob("results_*.json"))
        if not json_files:
            print("No results files found!")
            return
        
        filepath = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent file: {filepath}")
    else:
        filepath = args.filepath
    
    analyze_file(str(filepath))


if __name__ == "__main__":
    main()
