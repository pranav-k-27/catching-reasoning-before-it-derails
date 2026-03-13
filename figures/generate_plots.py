# figures/generate_plots.py
# Generate plots for paper figures

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Run: pip install matplotlib")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_token_comparison(results: List[Dict], output_path: str = "figures/token_comparison.png"):
    """
    Generate token comparison bar chart (Figure 2 in paper).
    
    Shows token usage comparison between student-only and supervised inference.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return
    
    # Aggregate by category
    categories = ["business", "logic_trap", "ethical", "paradox"]
    display_names = ["Business\nStrategy", "Logic\nTraps", "Ethical\nDilemmas", "Logical\nParadoxes"]
    
    student_tokens = []
    supervised_tokens = []
    
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            student_tokens.append(0)
            supervised_tokens.append(0)
            continue
        
        # Student tokens (always full generation)
        avg_student = sum(r["student_tokens"] for r in cat_results) / len(cat_results)
        student_tokens.append(avg_student)
        
        # Supervised tokens (stopped early if STOP)
        supervised = []
        for r in cat_results:
            if r["observer_decision"] == "STOP":
                supervised.append(r.get("oip", 150))
            else:
                supervised.append(r["student_tokens"])
        avg_supervised = sum(supervised) / len(supervised)
        supervised_tokens.append(avg_supervised)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, student_tokens, width, label='Student-only', 
                   color='#ff6b6b', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, supervised_tokens, width, label='Process Supervised',
                   color='#4ecdc4', edgecolor='black', linewidth=1)
    
    # Customize
    ax.set_xlabel('Task Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tokens Used (avg)', fontsize=12, fontweight='bold')
    ax.set_title('Token Usage: Student-only vs Process-Supervised Inference', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(student_tokens) * 1.2)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Add token savings annotation
    total_student = sum(student_tokens)
    total_supervised = sum(supervised_tokens)
    savings = 100 * (total_student - total_supervised) / total_student
    
    ax.text(0.98, 0.95, f'Overall Token Savings: {savings:.1f}%',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_ablation_results(ablation_data: Dict[str, Any], output_path: str = "figures/ablation_plot.png"):
    """
    Generate OIP ablation plot showing precision/recall tradeoff.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return
    
    summary = ablation_data.get("ablation_summary", {})
    if not summary:
        print("No ablation data found")
        return
    
    oip_values = sorted([int(k) for k in summary.keys()])
    precision = [summary[str(oip)]["precision"] for oip in oip_values]
    recall = [summary[str(oip)]["recall"] for oip in oip_values]
    f1 = [summary[str(oip)]["f1_score"] for oip in oip_values]
    tokens_saved = [summary[str(oip)]["tokens_saved_percent"] for oip in oip_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Precision, Recall, F1
    ax1.plot(oip_values, precision, 'o-', label='Precision', linewidth=2, markersize=8, color='#e74c3c')
    ax1.plot(oip_values, recall, 's-', label='Recall', linewidth=2, markersize=8, color='#3498db')
    ax1.plot(oip_values, f1, '^-', label='F1 Score', linewidth=2, markersize=8, color='#2ecc71')
    
    ax1.set_xlabel('Optimal Intervention Point (tokens)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Detection Performance vs OIP', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(oip_values)
    
    # Plot 2: Tokens Saved
    bars = ax2.bar(oip_values, tokens_saved, width=30, color='#9b59b6', edgecolor='black', linewidth=1)
    ax2.set_xlabel('Optimal Intervention Point (tokens)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tokens Saved (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Compute Efficiency vs OIP', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(oip_values)
    
    # Add value labels
    for bar, val in zip(bars, tokens_saved):
        ax2.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def generate_all_plots(results_file: str = None, ablation_file: str = None):
    """Generate all plots from results files."""
    
    results_dir = Path("results")
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    # Find most recent results file if not specified
    if results_file is None:
        json_files = list(results_dir.glob("results_*.json"))
        if json_files:
            results_file = str(max(json_files, key=lambda p: p.stat().st_mtime))
    
    # Find most recent ablation file if not specified
    if ablation_file is None:
        ablation_files = list(results_dir.glob("ablation_*.json"))
        if ablation_files:
            ablation_file = str(max(ablation_files, key=lambda p: p.stat().st_mtime))
    
    # Generate plots
    if results_file:
        print(f"\nGenerating plots from: {results_file}")
        data = load_results(results_file)
        results = data.get("results", [])
        if results:
            plot_token_comparison(results)
    else:
        print("No results file found for token comparison plot")
    
    if ablation_file:
        print(f"Generating ablation plot from: {ablation_file}")
        ablation_data = load_results(ablation_file)
        plot_ablation_results(ablation_data)
    else:
        print("No ablation file found for ablation plot")
    
    print("\n✓ Plot generation complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots for paper")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--ablation", type=str, help="Path to ablation JSON file")
    
    args = parser.parse_args()
    
    generate_all_plots(args.results, args.ablation)


if __name__ == "__main__":
    main()
