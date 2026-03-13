#!/usr/bin/env python3
# run.py
# Main entry point for running experiments

"""
Catching Reasoning Before It Derails
Inference-Time Process Supervision for Large Language Models

Usage:
    python run.py                      # Run all experiments with default settings
    python run.py --category business  # Run single category
    python run.py --ablation           # Run OIP ablation study
    python run.py --analyze            # Analyze most recent results
    python run.py --test               # Quick test with mock models
"""

import os
import sys
import argparse
from pathlib import Path

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import ModelConfig, ExperimentConfig
from prompts import get_all_prompts, get_categories


def run_experiments(args):
    """Run main experiments."""
    from experiments.run_experiments import ExperimentRunner
    
    print("\n" + "="*60)
    print("RUNNING PROCESS SUPERVISION EXPERIMENTS")
    print("="*60)
    
    config = ModelConfig()
    exp_config = ExperimentConfig()
    
    # Check for API token
    if config.hf_api_token:
        print(f"✓ HF API Token found")
    else:
        print("⚠ No HF API Token - using mock models for testing")
        print("  Set HF_API_TOKEN in .env file for real experiments")
    
    runner = ExperimentRunner(config, exp_config)
    
    if args.category:
        runner.run_category(args.category, oip=args.oip)
    else:
        runner.run_all(oip=args.oip)
    
    runner.save_results()
    runner.print_summary()
    
    print("\n✓ Experiments complete!")
    print("  Run 'python run.py --analyze' to generate paper tables")


def run_ablation(args):
    """Run OIP ablation study."""
    from experiments.run_ablation import AblationRunner
    
    print("\n" + "="*60)
    print("RUNNING OIP ABLATION STUDY")
    print("="*60)
    
    config = ModelConfig()
    exp_config = ExperimentConfig()
    
    runner = AblationRunner(config, exp_config)
    runner.run_ablation(args.oip_values)
    runner.save_results()
    runner.print_summary_table()
    
    print("\n✓ Ablation study complete!")


def run_analysis(args):
    """Analyze results and generate tables."""
    from analyze_results import analyze_file
    
    results_dir = Path("results")
    
    if args.file:
        filepath = args.file
    else:
        # Find most recent results file
        json_files = list(results_dir.glob("results_*.json"))
        if not json_files:
            print("No results files found! Run experiments first.")
            return
        filepath = max(json_files, key=lambda p: p.stat().st_mtime)
    
    analyze_file(str(filepath))


def run_test(args):
    """Quick test run with mock models."""
    from experiments.run_experiments import ExperimentRunner
    from prompts import get_prompts_by_category
    
    print("\n" + "="*60)
    print("TEST RUN (Mock Models)")
    print("="*60)
    
    config = ModelConfig()
    config.hf_api_token = None  # Force mock mode
    
    exp_config = ExperimentConfig()
    
    runner = ExperimentRunner(config, exp_config)
    
    # Run just 2 prompts from each category
    for category in get_categories():
        prompts = get_prompts_by_category(category)[:2]
        for prompt in prompts:
            runner.run_single(prompt, oip=150)
    
    runner.save_results("test_results.json")
    runner.print_summary()
    
    print("\n✓ Test complete!")
    print("  This was a test run with mock models.")
    print("  Set HF_API_TOKEN for real experiments.")


def show_info():
    """Show project info and setup instructions."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Catching Reasoning Before It Derails                        ║
║  Inference-Time Process Supervision for LLMs                 ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
    1. Install dependencies:
       pip install -r requirements.txt
    
    2. Set your Hugging Face API token:
       cp .env.example .env
       # Edit .env and add your token from https://huggingface.co/settings/tokens

USAGE:
    python run.py                      # Run all 40 experiments
    python run.py --category business  # Run single category (10 prompts)
    python run.py --ablation           # Run OIP ablation study
    python run.py --analyze            # Generate paper tables from results
    python run.py --test               # Quick test with mock models

OPTIONS:
    --oip 150                          # Set intervention point (default: 150)
    --category [business|logic_trap|ethical|paradox]
    --ablation --oip-values 100 150 200

OUTPUT:
    results/           # JSON experiment logs
    figures/           # Generated plots
    logs/              # Detailed logs

EXPERIMENT STRUCTURE:
    - 40 prompts total (10 per category)
    - Categories: Business, Logic Traps, Ethical Dilemmas, Paradoxes
    - Metrics: Precision, Recall, F1, Token Savings
""")


def main():
    parser = argparse.ArgumentParser(
        description="Catching Reasoning Before It Derails - Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                      Run all experiments
  python run.py --category business  Run single category  
  python run.py --ablation           Run OIP ablation study
  python run.py --analyze            Generate paper tables
  python run.py --test               Quick test run
  python run.py --info               Show setup instructions
        """
    )
    
    parser.add_argument("--category", type=str, choices=get_categories(),
                        help="Run specific category only")
    parser.add_argument("--oip", type=int, default=150,
                        help="Optimal Intervention Point in tokens (default: 150)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run OIP ablation study")
    parser.add_argument("--oip-values", type=int, nargs="+", default=[100, 150, 200],
                        help="OIP values for ablation (default: 100 150 200)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze results and generate tables")
    parser.add_argument("--file", type=str,
                        help="Specific results file to analyze")
    parser.add_argument("--test", action="store_true",
                        help="Quick test run with mock models")
    parser.add_argument("--info", action="store_true",
                        help="Show project info and setup instructions")
    
    args = parser.parse_args()
    
    # Route to appropriate function
    if args.info:
        show_info()
    elif args.test:
        run_test(args)
    elif args.ablation:
        run_ablation(args)
    elif args.analyze:
        run_analysis(args)
    else:
        run_experiments(args)


if __name__ == "__main__":
    main()
