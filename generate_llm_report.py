"""
Generate a detailed report of LLM evaluations on SED puzzles.
This script processes the results from run_real_llm_eval.py and creates
visualizations and comparison metrics.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

# Import the metrics module
from src.metrics import EvaluationMetrics


def generate_llm_comparison_chart(results, output_file):
    """Generate a bar chart comparing LLM performance across prompt types."""
    models = list(results.keys())
    prompt_types = list(results[models[0]].keys())
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Width of each bar group
    width = 0.8 / len(prompt_types)
    
    # Position of bar groups
    indices = np.arange(len(models))
    
    for i, prompt_type in enumerate(prompt_types):
        # Extract success rates for this prompt type
        success_rates = [results[model][prompt_type]["success_rate"] * 100 for model in models]
        
        # Plot the bars
        bars = ax.bar(indices + (i - len(prompt_types)/2 + 0.5) * width, 
                      success_rates, 
                      width, 
                      label=prompt_type)
        
        # Add value labels above each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Add labels and legend
    ax.set_xlabel('LLM Models')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('LLM Performance Comparison Across Prompt Types')
    ax.set_xticks(indices)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    # Directory containing the evaluation results
    results_dir = Path("evaluation/results")
    reports_dir = Path("evaluation/reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Initialize metrics calculator
    metrics = EvaluationMetrics(results_dir=results_dir)
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Find all result files
    result_files = list(results_dir.glob("*.json"))
    
    # Group result files by model and prompt type
    model_results = {}
    
    for file_path in result_files:
        filename = file_path.name
        
        # Skip non-LLM result files
        if not any(model in filename for model in ["gemini", "claude", "gpt"]):
            continue
        
        # Extract model and prompt type from filename
        parts = filename.split("_")
        model_name = parts[0]
        prompt_type = "_".join(parts[1:-1])  # Handle prompt types with underscores
        
        # Load the results
        results = metrics.load_results(filename)
        
        # Compute metrics
        basic_metrics = metrics.compute_basic_metrics(results)
        length_metrics = metrics.compute_solution_length_metrics(results)
        
        # Store metrics
        if model_name not in model_results:
            model_results[model_name] = {}
            
        model_results[model_name][prompt_type] = {
            "basic_metrics": basic_metrics,
            "length_metrics": length_metrics
        }
        
        print(f"Processed {filename}: {basic_metrics['correct']}/{basic_metrics['total']} correct ({basic_metrics['success_rate']*100:.2f}%)")
    
    # Generate comparison chart
    if model_results:
        print("\nGenerating comparison chart...")
        
        # Extract success rates for each model and prompt type
        comparison_data = {}
        for model, prompt_data in model_results.items():
            comparison_data[model] = {}
            for prompt_type, metrics_data in prompt_data.items():
                comparison_data[model][prompt_type] = metrics_data["basic_metrics"]["success_rate"]
        
        generate_llm_comparison_chart(comparison_data, reports_dir / "llm_comparison.png")
        
        # Save all metrics to a JSON file
        with open(reports_dir / "llm_metrics.json", 'w') as f:
            json.dump(model_results, f, indent=2)
        
        print(f"Report and visualizations saved to {reports_dir}")
    else:
        print("No LLM evaluation results found. Run run_real_llm_eval.py first.")

if __name__ == "__main__":
    main()