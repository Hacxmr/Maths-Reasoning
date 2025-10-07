"""
Enhanced implementation for the SED puzzle solver with improved accuracy.

This implementation addresses the main issues identified in the accuracy analysis:
1. String state tracking errors
2. Solution format parsing failures
3. Incomplete search space exploration

Usage:
    python enhanced_implementation.py --model gpt-4o --prompt_type chain_of_thought --max_retries 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advanced_metrics import EnhancedEvaluationMetrics
from src.enhanced_evaluator import EnhancedLLMEvaluator
from src.enhanced_real_llm_evaluator import (EnhancedGPTLLMEvaluator,
                                             EnhancedOpenRouterLLMEvaluator)
from src.schema import Problem
from src.utils import read_problem_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced SED Puzzle Solver Evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                        help="Model to use for evaluation: gpt-4o, claude-3-opus, gemini-1.5-pro")
    parser.add_argument("--prompt_type", type=str, default="chain_of_thought", 
                        choices=["zero_shot", "few_shot", "chain_of_thought", "enhanced"],
                        help="Type of prompt to use for evaluation")
    parser.add_argument("--data_dir", type=str, default="data/puzzles", 
                        help="Directory containing puzzle files")
    parser.add_argument("--max_retries", type=int, default=1,
                        help="Maximum number of retry attempts for failed solutions")
    parser.add_argument("--output_dir", type=str, default="evaluation/enhanced_results",
                        help="Directory to save evaluation results")
    return parser.parse_args()

def create_evaluator(model_name, prompt_type, max_retries):
    """Create the appropriate LLM evaluator based on the model name."""
    
    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Map prompt type to evaluator's format
    orig_prompt_type = {
        "zero_shot": "zero-shot",
        "few_shot": "few-shot",
        "chain_of_thought": "cot",
        "enhanced": "cot"
    }.get(prompt_type, "cot")
    
    # Create empty output directory if it doesn't exist
    os.makedirs("evaluation/enhanced_results", exist_ok=True)
    
    # Set up model-specific parameters
    puzzles_dir = "data/puzzles"
    output_dir = "evaluation/enhanced_results"
    
    if model_name.startswith("gpt"):
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        return EnhancedGPTLLMEvaluator(
            api_key=openai_api_key,
            model=model_name,
            puzzles_dir=puzzles_dir,
            output_dir=output_dir
        )
    elif any(name in model_name for name in ["claude", "gemini", "llama"]):
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        return EnhancedOpenRouterLLMEvaluator(
            api_key=openrouter_api_key,
            model=model_name,
            puzzles_dir=puzzles_dir,
            output_dir=output_dir
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def run_evaluation(args):
    """Run the enhanced evaluation pipeline."""
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load puzzles
    puzzles = read_problem_folder(Path(args.data_dir))
    print(f"Loaded {len(puzzles)} puzzles from {args.data_dir}")
    
    # Create evaluator
    evaluator = create_evaluator(args.model, args.prompt_type, args.max_retries)
    
    # Map prompt type to evaluator's format
    orig_prompt_type = {
        "zero_shot": "zero-shot",
        "few_shot": "few-shot",
        "chain_of_thought": "cot",
        "enhanced": "cot"
    }.get(args.prompt_type, "cot")
    
    # Evaluate puzzles using evaluate_with_real_llm
    print(f"Running evaluation with {args.model} using {args.prompt_type} prompting...")
    results = evaluator.evaluate_with_real_llm(
        prompt_type=orig_prompt_type, 
        num_samples=len(puzzles)
    )
    
    # Save results
    model_short = args.model.replace("-", "_")
    results_file = f"{args.prompt_type}_{model_short}_results.json"
    evaluator.save_results(results, results_file)
    
    print(f"Results saved to {output_dir / results_file}")
    
    # Calculate success rate
    metrics = evaluator.compute_basic_metrics(results)
    success_rate = metrics["success_rate"]
    print(f"Success rate: {metrics['correct']}/{metrics['total']} ({success_rate*100:.2f}%)")
    
    return results, results_file

def run_comparative_analysis(args):
    """Run analysis across different prompt types to compare performance."""
    
    # Setup
    prompt_types = ["zero_shot", "few_shot", "chain_of_thought", "enhanced"]
    model_short = args.model.replace("-", "_")
    output_dir = Path(args.output_dir)
    results_dir = output_dir
    
    # Store results for comparison
    all_results_files = {}
    all_results = {}
    
    # Run evaluation for each prompt type
    for prompt_type in prompt_types:
        modified_args = args
        modified_args.prompt_type = prompt_type
        
        print(f"\n===== Running evaluation with {prompt_type} prompt =====\n")
        results, results_file = run_evaluation(modified_args)
        
        all_results[prompt_type] = results
        all_results_files[prompt_type] = results_file
    
    # Run enhanced analysis
    metrics = EnhancedEvaluationMetrics(results_dir)
    enhanced_report = metrics.generate_enhanced_report(
        all_results_files,
        output_dir=str(output_dir / "reports")
    )
    
    print(f"\n===== Enhanced analysis complete =====")
    print(f"Report saved to {output_dir / 'reports' / 'enhanced_evaluation_report.md'}")
    
    return enhanced_report

if __name__ == "__main__":
    args = parse_args()
    
    if args.prompt_type == "comparative":
        # Run all prompt types and compare
        enhanced_report = run_comparative_analysis(args)
    else:
        # Run a single evaluation
        results, results_file = run_evaluation(args)
        
        # Generate a single report
        metrics = EnhancedEvaluationMetrics(args.output_dir)
        enhanced_report = metrics.generate_enhanced_report(
            {args.prompt_type: results_file},
            output_dir=f"{args.output_dir}/reports"
        )