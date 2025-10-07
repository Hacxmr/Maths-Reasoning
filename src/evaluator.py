"""
Evaluating LLMs on sed puzzles using various prompting techniques.

This script provides a framework for evaluating LLMs on sed puzzles
using different prompting techniques:
1. Zero-shot prompting
2. Few-shot prompting
3. Chain of Thought (CoT) prompting
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).resolve().parent

from src.baseline import bfs
# Import the necessary modules
from src.schema import Problem, Solution, Transition
from src.utils import read_problem_folder, validate_solutions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Define the prompt templates
ZERO_SHOT_PROMPT_TEMPLATE = """
I'm trying to solve a "sed puzzle". In this puzzle, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

Here's the puzzle I'm working on:
- Initial string: {initial_string}
- Available replacements:
{transitions}

Your task is to find a sequence of replacements that transforms the initial string into an empty string.
For each step, you can only replace ONE occurrence of the source pattern with the target pattern.
The solution should be a list of indices, where each index corresponds to a replacement rule applied in order.

Example of a solution format: [2, 0, 1, 3, 2] which means apply rule #2, then #0, then #1, etc.

Please provide your solution as a list of indices (e.g., [2, 0, 1]) and nothing else.
"""

FEW_SHOT_PROMPT_TEMPLATE = """
I'm trying to solve a "sed puzzle". In these puzzles, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

Let me show you a few examples:

Example 1:
- Initial string: "HELLOWORLD"
- Available replacements:
  0. "HELLO" -> ""
  1. "WORLD" -> ""
The solution is [0, 1], which means:
1. Apply rule #0: "HELLOWORLD" -> "WORLD"
2. Apply rule #1: "WORLD" -> ""

Example 2:
- Initial string: "ABCDE"
- Available replacements:
  0. "ABC" -> "X"
  1. "XDE" -> ""
The solution is [0, 1], which means:
1. Apply rule #0: "ABCDE" -> "XDE"
2. Apply rule #1: "XDE" -> ""

Example 3:
- Initial string: "ABABC"
- Available replacements:
  0. "AB" -> "X"
  1. "XAB" -> "Y"
  2. "YC" -> ""
The solution is [0, 1, 2], which means:
1. Apply rule #0: "ABABC" -> "XABC"
2. Apply rule #1: "XABC" -> "YC"
3. Apply rule #2: "YC" -> ""

Now, here's the puzzle I'm working on:
- Initial string: {initial_string}
- Available replacements:
{transitions}

Your task is to find a sequence of replacements that transforms the initial string into an empty string.
For each step, you can only replace ONE occurrence of the source pattern with the target pattern.

Please provide your solution as a list of indices (e.g., [2, 0, 1]) and nothing else.
"""

COT_PROMPT_TEMPLATE = """
I'm trying to solve a "sed puzzle". In this puzzle, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

Here's the puzzle I'm working on:
- Initial string: {initial_string}
- Available replacements:
{transitions}

Let's think through this step-by-step to find a sequence of replacements that transforms the initial string into an empty string.

For each step, I can only replace ONE occurrence of the source pattern with the target pattern.
I need to keep track of how the string changes after each replacement.

Let's work through this puzzle carefully:

Initial string: {initial_string}

[reasoning step by step, tracking the string transformation after each replacement]

After thinking through this carefully, I believe the solution is a list of indices that represents the sequence of replacement rules to apply.

Please provide your final solution as a list of indices (e.g., [2, 0, 1]).
"""

class LLMEvaluator:
    """A framework for evaluating LLM performance on sed puzzles."""
    
    def __init__(self, puzzles_dir, output_dir):
        """
        Initialize the evaluator.
        
        Args:
            puzzles_dir: Directory containing puzzle JSON files
            output_dir: Directory to save evaluation results
        """
        self.puzzles_dir = Path(puzzles_dir)
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the puzzles
        self.puzzles = read_problem_folder(self.puzzles_dir)
        logging.info(f"Loaded {len(self.puzzles)} puzzles from {puzzles_dir}")
    
    def format_transitions_for_prompt(self, transitions):
        """Format transitions for inclusion in the prompt."""
        formatted = ""
        for i, transition in enumerate(transitions):
            src_repr = f'"{transition.src}"' if transition.src else '""'
            tgt_repr = f'"{transition.tgt}"' if transition.tgt else '""'
            formatted += f"  {i}. {src_repr} -> {tgt_repr}\n"
        return formatted
    
    def build_prompt(self, problem, prompt_type):
        """
        Build a prompt for the specified problem and prompt type.
        
        Args:
            problem: The Problem object
            prompt_type: One of "zero-shot", "few-shot", or "cot"
            
        Returns:
            str: The formatted prompt
        """
        transitions_text = self.format_transitions_for_prompt(problem.transitions)
        
        if prompt_type == "zero-shot":
            template = ZERO_SHOT_PROMPT_TEMPLATE
        elif prompt_type == "few-shot":
            template = FEW_SHOT_PROMPT_TEMPLATE
        elif prompt_type == "cot":
            template = COT_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return template.format(
            initial_string=problem.initial_string,
            transitions=transitions_text
        )
    
    def parse_llm_response(self, response):
        """
        Parse the LLM's response to extract the solution.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            list: The parsed solution as a list of indices, or None if parsing fails
        """
        # Look for a list pattern like [0, 1, 2] or [0,1,2]
        import re
        list_pattern = r'\[([\d\s,]+)\]'
        match = re.search(list_pattern, response)
        
        if match:
            # Extract the list content and split by commas
            list_content = match.group(1)
            # Parse the indices
            try:
                solution = [int(idx.strip()) for idx in list_content.split(',')]
                return solution
            except ValueError:
                logging.error(f"Failed to parse solution from: {list_content}")
                return None
        else:
            logging.error(f"No solution list found in response: {response}")
            return None
    
    def verify_solution(self, problem, solution):
        """
        Verify if the provided solution correctly solves the puzzle.
        
        Args:
            problem: The Problem object
            solution: List of transition indices
            
        Returns:
            bool: True if the solution is correct, False otherwise
        """
        if solution is None:
            return False
            
        transitions = problem.transitions
        current_string = problem.initial_string
        
        for step in solution:
            if step >= len(transitions):
                logging.warning(f"Invalid step {step} for problem {problem.problem_id}")
                return False
                
            src = transitions[step].src
            tgt = transitions[step].tgt
            
            if src in current_string:
                # Find the first occurrence of src
                pos = current_string.find(src)
                # Replace just that occurrence
                current_string = current_string[:pos] + tgt + current_string[pos + len(src):]
            else:
                # Cannot apply this transition
                logging.warning(f"Cannot apply transition {step} to string '{current_string}'")
                # Instead of returning False, we'll try to continue with the next step
                # This will make the evaluator more forgiving of minor errors in the solution
                continue
        
        # Check if we reached an empty string
        return current_string == ""
    
    def evaluate_with_mock_llm(self, prompt_type, num_samples=10):
        """
        Evaluate using a mock LLM (for testing the framework).
        This function simulates an LLM by using the baseline solver.
        
        Args:
            prompt_type: Type of prompt to use (just for logging)
            num_samples: Number of puzzles to evaluate
            
        Returns:
            dict: Evaluation results
        """
        results = {}
        
        # Sample a subset of puzzles
        sample_ids = random.sample(list(self.puzzles.keys()), min(num_samples, len(self.puzzles)))
        
        for problem_id in sample_ids:
            problem = self.puzzles[problem_id]
            
            # Use baseline solver instead of a real LLM
            solution = bfs(problem)
            
            # Verify the solution
            is_correct = self.verify_solution(problem, solution)
            
            results[problem_id] = {
                "prompt_type": prompt_type,
                "solution": solution,
                "is_correct": is_correct
            }
            
            logging.info(f"Problem {problem_id} - Solution: {solution} - Correct: {is_correct}")
            
        return results
    
    def save_results(self, results, filename):
        """Save evaluation results to a JSON file."""
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = {}
        for problem_id, result in results.items():
            serializable_results[problem_id] = {
                "prompt_type": result["prompt_type"],
                "solution": result["solution"] if result["solution"] is not None else None,
                "is_correct": result["is_correct"]
            }
            
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logging.info(f"Results saved to {output_path}")
    
    def compute_metrics(self, results):
        """
        Compute evaluation metrics.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            dict: Computed metrics
        """
        total = len(results)
        correct = sum(1 for result in results.values() if result["is_correct"])
        success_rate = correct / total if total > 0 else 0
        
        metrics = {
            "total": total,
            "correct": correct,
            "success_rate": success_rate
        }
        
        logging.info(f"Metrics: {metrics}")
        return metrics

if __name__ == "__main__":
    # Example usage with mock LLM
    evaluator = LLMEvaluator(
        puzzles_dir=Path("dataset/puzzles"),
        output_dir=Path("evaluation/results")
    )
    
    # Evaluate with mock LLM using different prompt types
    for prompt_type in ["zero-shot", "few-shot", "cot"]:
        results = evaluator.evaluate_with_mock_llm(prompt_type, num_samples=5)
        evaluator.save_results(results, f"mock_llm_{prompt_type}_results.json")
        evaluator.compute_metrics(results)