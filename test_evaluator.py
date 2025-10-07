"""
Script to test the LLM evaluator with a mock LLM.
"""

import sys
from pathlib import Path

# Get the absolute path to the project root
root_dir = Path(__file__).resolve().parent

# Add the project root to the Python path
sys.path.insert(0, str(root_dir))

# Now import and run the evaluator
from src.evaluator import LLMEvaluator

if __name__ == "__main__":
    print("Initializing evaluator...")
    evaluator = LLMEvaluator(
        puzzles_dir=root_dir / "dataset" / "puzzles",
        output_dir=root_dir / "evaluation" / "results"
    )
    
    print("Testing with mock LLM...")
    # Test with a small number of samples first
    for prompt_type in ["zero-shot", "few-shot", "cot"]:
        print(f"Evaluating with {prompt_type} prompting...")
        results = evaluator.evaluate_with_mock_llm(prompt_type, num_samples=5)
        evaluator.save_results(results, f"mock_llm_{prompt_type}_results.json")
        metrics = evaluator.compute_metrics(results)
        print(f"Results for {prompt_type}: Success rate = {metrics['success_rate']*100:.2f}%")
    
    print("Evaluation complete!")