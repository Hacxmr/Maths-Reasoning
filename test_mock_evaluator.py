"""
Simple test script for the mock LLM evaluator.
"""

import json
import logging
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Helper function to read a puzzle from a file
def read_puzzle(file_path):
    with open(file_path, 'r') as f:
        puzzle_data = json.load(f)
    return puzzle_data

# Mock baseline BFS solver
def mock_solve(puzzle):
    """Mock solver that just returns a simple solution for testing"""
    # Return a random sequence of up to 5 transition indices
    num_transitions = len(puzzle["transitions"])
    if num_transitions == 0:
        return []
    
    # For testing, just return 2-3 random indices
    solution_length = min(num_transitions, random.randint(2, 3))
    return [random.randint(0, num_transitions - 1) for _ in range(solution_length)]

def test_mock_evaluator():
    """Test the evaluator with mock solutions"""
    # Define paths
    puzzles_dir = Path("dataset/puzzles")
    results_dir = Path("evaluation/results")
    results_dir.mkdir(exist_ok=True)
    
    # Get a list of puzzle files
    puzzle_files = list(puzzles_dir.glob("*.json"))
    logging.info(f"Found {len(puzzle_files)} puzzle files")
    
    # Sample a few puzzles for testing
    sample_size = min(5, len(puzzle_files))
    sampled_files = random.sample(puzzle_files, sample_size)
    
    # Simulate different prompting techniques
    prompt_types = ["zero-shot", "few-shot", "cot"]
    
    for prompt_type in prompt_types:
        logging.info(f"Testing with {prompt_type} prompting...")
        
        results = {}
        for puzzle_file in sampled_files:
            puzzle = read_puzzle(puzzle_file)
            problem_id = puzzle["problem_id"]
            
            # Get mock solution
            solution = mock_solve(puzzle)
            
            # In a real scenario, we'd verify the solution
            is_correct = random.choice([True, False])  # Mock correctness for testing
            
            results[problem_id] = {
                "prompt_type": prompt_type,
                "solution": solution,
                "is_correct": is_correct
            }
            
            logging.info(f"Problem {problem_id} - Solution: {solution} - Correct: {is_correct}")
        
        # Save results
        output_file = results_dir / f"mock_{prompt_type}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Compute metrics
        total = len(results)
        correct = sum(1 for result in results.values() if result["is_correct"])
        success_rate = correct / total if total > 0 else 0
        
        metrics = {
            "total": total,
            "correct": correct,
            "success_rate": success_rate
        }
        
        logging.info(f"Metrics for {prompt_type}: {metrics}")

if __name__ == "__main__":
    test_mock_evaluator()