"""
Comparative testing script for evaluating the performance difference between
original and enhanced SED puzzle solver implementations.

This script runs both the original and enhanced implementations on the same
set of puzzles and generates comparative metrics to demonstrate improvements.

Usage:
    python compare_implementations.py --model gpt-4o --puzzles_count 20 --prompt_type chain_of_thought
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.advanced_metrics import EnhancedEvaluationMetrics
from src.enhanced_evaluator import EnhancedLLMEvaluator
from src.enhanced_real_llm_evaluator import (EnhancedGPTLLMEvaluator,
                                             EnhancedOpenRouterLLMEvaluator)
from src.evaluator import LLMEvaluator
from src.real_llm_evaluator import GPTLLMEvaluator, OpenRouterLLMEvaluator
# Import our modules
from src.schema import Problem
from src.utils import read_problem_folder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare original vs enhanced SED puzzle solver implementations")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                        help="Model to use: gpt-4o, claude-3-opus, gemini-1.5-pro")
    parser.add_argument("--prompt_type", type=str, default="chain_of_thought", 
                        choices=["zero_shot", "few_shot", "chain_of_thought"],
                        help="Type of prompt to use")
    parser.add_argument("--puzzles_count", type=int, default=10, 
                        help="Number of puzzles to evaluate")
    parser.add_argument("--data_dir", type=str, default="dataset/puzzles", 
                        help="Directory containing puzzle files")
    parser.add_argument("--output_dir", type=str, default="evaluation/comparison",
                        help="Directory to save comparison results")
    parser.add_argument("--use_cached", action="store_true",
                        help="Use cached results if available instead of running new evaluations")
    return parser.parse_args()

def setup_evaluators(model_name: str, prompt_type: str):
    """
    Set up original and enhanced evaluators for the specified model.
    
    Args:
        model_name: Name of the LLM model to use
        prompt_type: Type of prompt to use
    
    Returns:
        Tuple of (original_evaluator, enhanced_evaluator)
    """
    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Define directories
    puzzles_dir = "dataset/puzzles"
    output_dir = "evaluation/comparison"
    
    if model_name.startswith("gpt"):
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Map prompt type to original evaluator's format
        orig_prompt_type = {
            "zero_shot": "zero-shot",
            "few_shot": "few-shot",
            "chain_of_thought": "cot"
        }.get(prompt_type, "cot")
        
        # Setup both evaluators
        original = GPTLLMEvaluator(
            api_key=openai_api_key,
            model=model_name,
            puzzles_dir=puzzles_dir,
            output_dir=output_dir
        )
        
        enhanced = EnhancedGPTLLMEvaluator(
            api_key=openai_api_key,
            model=model_name,
            puzzles_dir=puzzles_dir,
            output_dir=output_dir
        )
        
    elif any(name in model_name for name in ["claude", "gemini", "llama"]):
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Map prompt type to original evaluator's format
        orig_prompt_type = {
            "zero_shot": "zero-shot",
            "few_shot": "few-shot",
            "chain_of_thought": "cot"
        }.get(prompt_type, "cot")
        
        # Setup both evaluators
        original = OpenRouterLLMEvaluator(
            api_key=openrouter_api_key,
            model=model_name,
            puzzles_dir=puzzles_dir,
            output_dir=output_dir
        )
        
        enhanced = EnhancedOpenRouterLLMEvaluator(
            api_key=openrouter_api_key,
            model=model_name,
            puzzles_dir=puzzles_dir,
            output_dir=output_dir
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return original, enhanced

def run_comparison(args):
    """
    Run a comparison between original and enhanced implementations.
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load puzzles
    puzzles = read_problem_folder(Path(args.data_dir))
    logging.info(f"Loaded {len(puzzles)} puzzles from {args.data_dir}")
    
    # Sample puzzles for evaluation
    if len(puzzles) > args.puzzles_count:
        puzzle_ids = random.sample(list(puzzles.keys()), args.puzzles_count)
        selected_puzzles = {pid: puzzles[pid] for pid in puzzle_ids}
    else:
        selected_puzzles = puzzles
    
    logging.info(f"Selected {len(selected_puzzles)} puzzles for evaluation")
    
    # Check for cached results
    model_name_safe = args.model.replace("-", "_")
    original_results_file = output_dir / f"original_{args.prompt_type}_{model_name_safe}_results.json"
    enhanced_results_file = output_dir / f"enhanced_{args.prompt_type}_{model_name_safe}_results.json"
    
    original_results = {}
    enhanced_results = {}
    
    if args.use_cached and original_results_file.exists() and enhanced_results_file.exists():
        logging.info("Loading cached results...")
        with open(original_results_file, 'r') as f:
            original_results = json.load(f)
        with open(enhanced_results_file, 'r') as f:
            enhanced_results = json.load(f)
    else:
        # Setup evaluators
        original_evaluator, enhanced_evaluator = setup_evaluators(args.model, args.prompt_type)
        
        # Run evaluations
        logging.info(f"Running original implementation with {args.model} and {args.prompt_type} prompting...")
        
        # Map prompt type to original evaluator's format
        orig_prompt_type = {
            "zero_shot": "zero-shot",
            "few_shot": "few-shot",
            "chain_of_thought": "cot"
        }.get(args.prompt_type, "cot")
        
        for i, (puzzle_id, puzzle) in enumerate(selected_puzzles.items()):
            logging.info(f"[Original] Evaluating puzzle {i+1}/{len(selected_puzzles)}: {puzzle_id}...")
            
            # Temporarily add the puzzle to the original_evaluator's puzzles dictionary
            # This is a workaround since original implementation evaluates from its puzzles dict
            puzzle_id_str = str(puzzle_id)
            original_evaluator.puzzles = {puzzle_id_str: puzzle}
            
            # Evaluate with original implementation
            results = original_evaluator.evaluate_with_real_llm(
                prompt_type=orig_prompt_type,
                num_samples=1
            )
            
            # Extract the result for this puzzle
            original_result = next(iter(results.values()))
            
            original_results[puzzle_id] = original_result
            
            logging.info(f"Original: Correct: {original_result['is_correct']}")
            
            # Small delay to avoid API rate limits
            time.sleep(1)
        
        # Save original results
        with open(original_results_file, 'w') as f:
            json.dump(original_results, f, indent=2)
        
        logging.info(f"Running enhanced implementation with {args.model} and {args.prompt_type} prompting...")
        
        for i, (puzzle_id, puzzle) in enumerate(selected_puzzles.items()):
            logging.info(f"[Enhanced] Evaluating puzzle {i+1}/{len(selected_puzzles)}: {puzzle_id}...")
            
            # Temporarily add the puzzle to the enhanced_evaluator's puzzles dictionary
            # This is a workaround since enhanced implementation evaluates from its puzzles dict
            puzzle_id_str = str(puzzle_id)
            enhanced_evaluator.puzzles = {puzzle_id_str: puzzle}
            
            # Map prompt type to enhanced evaluator's format
            enh_prompt_type = {
                "zero_shot": "zero-shot",
                "few_shot": "few-shot",
                "chain_of_thought": "cot"
            }.get(args.prompt_type, "cot")
            
            # Evaluate with enhanced implementation
            results = enhanced_evaluator.evaluate_with_real_llm(
                prompt_type=enh_prompt_type,
                num_samples=1
            )
            
            # Extract the result for this puzzle
            enhanced_result = next(iter(results.values()))
            
            enhanced_results[puzzle_id] = enhanced_result
            
            logging.info(f"Enhanced: Correct: {enhanced_result['is_correct']} - Attempts: {enhanced_result['attempts']}")
            
            # Small delay to avoid API rate limits
            time.sleep(1)
        
        # Save enhanced results
        with open(enhanced_results_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
    
    # Generate comparative metrics
    generate_comparison_report(original_results, enhanced_results, args, output_dir)

def generate_comparison_report(original_results, enhanced_results, args, output_dir):
    """
    Generate a comparative report between original and enhanced implementations.
    
    Args:
        original_results: Results from original implementation
        enhanced_results: Results from enhanced implementation
        args: Command line arguments
        output_dir: Output directory
    """
    # Create metrics calculator
    metrics = EnhancedEvaluationMetrics(output_dir)
    
    # Calculate metrics
    original_metrics = calculate_metrics(original_results)
    enhanced_metrics = calculate_metrics(enhanced_results)
    
    # Generate comparison report
    report_file = output_dir / f"comparison_{args.prompt_type}_{args.model.replace('-', '_')}_report.md"
    
    with open(report_file, 'w') as f:
        f.write(f"# SED Puzzle Solver Implementation Comparison\n\n")
        
        f.write(f"## Overview\n\n")
        f.write(f"- **Model**: {args.model}\n")
        f.write(f"- **Prompt Type**: {args.prompt_type}\n")
        f.write(f"- **Puzzles Evaluated**: {len(original_results)}\n\n")
        
        f.write(f"## Success Rate Comparison\n\n")
        f.write(f"| Implementation | Success Rate | First Attempt Success | Retry Success |\n")
        f.write(f"|----------------|--------------|----------------------|--------------|\n")
        
        original_success_rate = original_metrics["success_rate"] * 100
        
        enhanced_success_rate = enhanced_metrics["success_rate"] * 100
        enhanced_first_attempt = enhanced_metrics.get("first_attempt_rate", 0) * 100
        enhanced_retry_rate = enhanced_metrics.get("retry_success_rate", 0) * 100
        
        f.write(f"| Original | {original_success_rate:.1f}% | N/A | N/A |\n")
        f.write(f"| Enhanced | {enhanced_success_rate:.1f}% | {enhanced_first_attempt:.1f}% | {enhanced_retry_rate:.1f}% |\n\n")
        
        # Calculate and display improvement
        absolute_improvement = enhanced_success_rate - original_success_rate
        relative_improvement = ((enhanced_success_rate / original_success_rate) - 1) * 100 if original_success_rate > 0 else float('inf')
        
        f.write(f"**Absolute Improvement**: {absolute_improvement:.1f} percentage points\n")
        
        if original_success_rate > 0:
            f.write(f"**Relative Improvement**: {relative_improvement:.1f}%\n\n")
        else:
            f.write(f"**Relative Improvement**: N/A (original success rate was 0%)\n\n")
        
        # Failure analysis
        f.write(f"## Failure Analysis\n\n")
        
        # Original implementation failure analysis
        original_failures = analyze_failures(original_results)
        f.write(f"### Original Implementation Failure Distribution\n\n")
        f.write(f"| Failure Type | Count | Percentage |\n")
        f.write(f"|--------------|-------|------------|\n")
        
        for failure_type, count in original_failures["counts"].items():
            percentage = original_failures["percentages"][failure_type]
            f.write(f"| {failure_type} | {count} | {percentage:.1f}% |\n")
        
        f.write(f"\nTotal failures: {original_failures['total']}\n\n")
        
        # Enhanced implementation failure analysis
        enhanced_failures = analyze_failures(enhanced_results)
        f.write(f"### Enhanced Implementation Failure Distribution\n\n")
        f.write(f"| Failure Type | Count | Percentage |\n")
        f.write(f"|--------------|-------|------------|\n")
        
        for failure_type, count in enhanced_failures["counts"].items():
            percentage = enhanced_failures["percentages"][failure_type]
            f.write(f"| {failure_type} | {count} | {percentage:.1f}% |\n")
        
        f.write(f"\nTotal failures: {enhanced_failures['total']}\n\n")
        
        # Key improvements section
        f.write(f"## Key Improvements\n\n")
        
        # Identify most significant improvements
        improvements = {}
        for failure_type in original_failures["counts"].keys():
            orig_count = original_failures["counts"].get(failure_type, 0)
            enh_count = enhanced_failures["counts"].get(failure_type, 0)
            
            if orig_count > 0:
                reduction = orig_count - enh_count
                reduction_percent = (reduction / orig_count) * 100 if orig_count > 0 else 0
                improvements[failure_type] = {
                    "reduction": reduction,
                    "reduction_percent": reduction_percent
                }
        
        # Sort improvements by percentage reduction
        sorted_improvements = sorted(
            improvements.items(), 
            key=lambda x: x[1]["reduction_percent"], 
            reverse=True
        )
        
        for failure_type, data in sorted_improvements:
            if data["reduction"] > 0:
                f.write(f"### {failure_type.replace('_', ' ').title()} Reduction\n\n")
                f.write(f"- **Reduced by**: {data['reduction']} instances ({data['reduction_percent']:.1f}%)\n")
                
                if failure_type == "parse_failure":
                    f.write("- **Improvements**: Enhanced output format instructions and robust parsing with multiple fallback strategies\n\n")
                elif failure_type == "state_tracking_error":
                    f.write("- **Improvements**: Explicit state tracking prompts and step-by-step verification\n\n")
                elif failure_type == "invalid_rule_index":
                    f.write("- **Improvements**: Better rule index validation and clearer rule indexing in prompts\n\n")
                elif failure_type == "incomplete_solution":
                    f.write("- **Improvements**: Enhanced verification steps to check for complete transformations\n\n")
                else:
                    f.write("- **Improvements**: Multiple enhanced techniques and verification steps\n\n")
        
        # Conclusion
        f.write(f"## Conclusion\n\n")
        
        if enhanced_success_rate > original_success_rate:
            f.write("The enhanced implementation shows significant improvements over the original implementation. ")
            f.write(f"With a {absolute_improvement:.1f} percentage point increase in success rate")
            
            if original_success_rate > 0:
                f.write(f" ({relative_improvement:.1f}% relative improvement)")
            
            f.write(", the enhancements demonstrate effective solutions to the key accuracy issues.\n\n")
            
            # Highlight most effective improvement
            if sorted_improvements and sorted_improvements[0][1]["reduction"] > 0:
                top_improvement = sorted_improvements[0][0].replace('_', ' ').title()
                f.write(f"The most significant improvement was in reducing {top_improvement} errors, ")
                f.write(f"with a {sorted_improvements[0][1]['reduction_percent']:.1f}% reduction in this type of failure.\n")
        else:
            f.write("The enhanced implementation did not show improvement over the original implementation for this specific ")
            f.write(f"combination of model ({args.model}) and prompt type ({args.prompt_type}). ")
            f.write("Further analysis and refinement of the enhanced implementation may be necessary.\n")
    
    logging.info(f"Comparison report generated at {report_file}")
    return original_metrics, enhanced_metrics

def calculate_metrics(results):
    """
    Calculate metrics from evaluation results.
    
    Args:
        results: Evaluation results
        
    Returns:
        Dictionary of metrics
    """
    total = len(results)
    correct = sum(1 for result in results.values() if result["is_correct"])
    success_rate = correct / total if total > 0 else 0
    
    # Add metrics for retry success if attempt data is available
    if "attempts" in next(iter(results.values()), {}):
        first_attempt_correct = sum(1 for result in results.values() 
                                  if result["is_correct"] and result["attempts"] == 1)
        retry_success = sum(1 for result in results.values() 
                          if result["is_correct"] and result["attempts"] > 1)
        retry_rate = retry_success / (total - first_attempt_correct) if (total - first_attempt_correct) > 0 else 0
        
        metrics = {
            "total": total,
            "correct": correct,
            "success_rate": success_rate,
            "first_attempt_correct": first_attempt_correct,
            "first_attempt_rate": first_attempt_correct / total if total > 0 else 0,
            "retry_success": retry_success,
            "retry_success_rate": retry_rate
        }
    else:
        metrics = {
            "total": total,
            "correct": correct,
            "success_rate": success_rate
        }
    
    return metrics

def analyze_failures(results):
    """
    Analyze patterns in failed solutions.
    
    Args:
        results: Dictionary of evaluation results
        
    Returns:
        dict: Analysis of failure patterns
    """
    failures = {
        "parse_failure": 0,
        "state_tracking_error": 0,
        "invalid_rule_index": 0, 
        "incomplete_solution": 0,
        "other_failure": 0
    }
    
    for result in results.values():
        if not result["is_correct"]:
            if result.get("solution") is None:
                failures["parse_failure"] += 1
            elif "verification_details" in result:
                details = " ".join(result["verification_details"])
                
                if "pattern not found" in details or "Cannot apply" in details:
                    failures["state_tracking_error"] += 1
                elif "does not exist" in details:
                    failures["invalid_rule_index"] += 1
                elif "Final string is not empty" in details:
                    failures["incomplete_solution"] += 1
                else:
                    failures["other_failure"] += 1
            else:
                failures["other_failure"] += 1
    
    total_failures = sum(failures.values())
    
    # Calculate percentages
    failure_percentages = {}
    for failure_type, count in failures.items():
        failure_percentages[failure_type] = (count / total_failures * 100) if total_failures > 0 else 0
    
    return {
        "counts": failures,
        "percentages": failure_percentages,
        "total": total_failures
    }

if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)