"""
Script to run the evaluation metrics on the test results.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

# Import our metrics class
from src.metrics import EvaluationMetrics

if __name__ == "__main__":
    # Create reports directory
    reports_dir = root_dir / "evaluation" / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Initialize the metrics evaluator
    metrics = EvaluationMetrics(results_dir="evaluation/results")
    
    # Define result files for different prompt types
    results_files = {
        "zero-shot": "mock_zero-shot_results.json",
        "few-shot": "mock_few-shot_results.json",
        "cot": "mock_cot_results.json"
    }
    
    # Generate the report
    print("Generating evaluation report...")
    report = metrics.generate_report(results_files, output_dir=str(reports_dir))
    
    print(f"\nReport and visualizations saved to {reports_dir}")