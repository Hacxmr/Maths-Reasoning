# Implementing SED Puzzle Solver with Improved Accuracy

This implementation guide provides code improvements to address the accuracy issues in the SED Puzzle Solver. These enhancements focus on string state tracking, solution verification, and improved prompt design.

## 1. Enhanced Evaluator Implementation

```python
"""
Enhanced evaluator module with improved accuracy features.
"""

import os
import json
import re
import time
import random
import logging
from pathlib import Path

from src.schema import Problem, Transition, Solution
from src.baseline import bfs
from src.utils import read_problem_folder, validate_solutions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Enhanced prompt templates
ENHANCED_ZERO_SHOT_PROMPT_TEMPLATE = """
I'm trying to solve a "sed puzzle". In this puzzle, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

Here's the puzzle I'm working on:
- Initial string: {initial_string}
- Available replacements:
{transitions}

Your task is to find a sequence of replacements that transforms the initial string into an empty string.
For each step, you can only replace ONE occurrence of the source pattern with the target pattern.

IMPORTANT OUTPUT FORMAT: You must provide your solution as a list of indices in the exact format [a, b, c] where each index corresponds to a replacement rule applied in order.

For example, a solution like [2, 0, 1] means:
1. Apply rule #2 first
2. Then apply rule #0
3. Then apply rule #1

Remember to track the EXACT state of the string after each replacement.

AFTER working out your solution, make sure your final answer is ONLY a list of indices and nothing else.
"""

ENHANCED_FEW_SHOT_PROMPT_TEMPLATE = """
I'm trying to solve a "sed puzzle". In these puzzles, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

Let me show you a few examples:

Example 1:
- Initial string: "HELLOWORLD"
- Available replacements:
  0. "HELLO" -> ""
  1. "WORLD" -> ""
The solution is [0, 1], which means:
1. Apply rule #0: "HELLOWORLD" -> "WORLD" (replaced "HELLO" with "")
2. Apply rule #1: "WORLD" -> "" (replaced "WORLD" with "")

Example 2:
- Initial string: "ABCDE"
- Available replacements:
  0. "ABC" -> "X"
  1. "XDE" -> ""
The solution is [0, 1], which means:
1. Apply rule #0: "ABCDE" -> "XDE" (replaced "ABC" with "X")
2. Apply rule #1: "XDE" -> "" (replaced "XDE" with "")

Example 3:
- Initial string: "ABABC"
- Available replacements:
  0. "AB" -> "X"
  1. "XAB" -> "Y"
  2. "YC" -> ""
The solution is [0, 1, 2], which means:
1. Apply rule #0: "ABABC" -> "XABC" (replaced first "AB" with "X")
2. Apply rule #1: "XABC" -> "YC" (replaced "XAB" with "Y")
3. Apply rule #2: "YC" -> "" (replaced "YC" with "")

Now, here's the puzzle I'm working on:
- Initial string: {initial_string}
- Available replacements:
{transitions}

Your task is to find a sequence of replacements that transforms the initial string into an empty string.
For each step, you can only replace ONE occurrence of the source pattern with the target pattern.

IMPORTANT: Always track the EXACT state of the string after each replacement.
Your final answer must be ONLY a list of indices (e.g., [2, 0, 1]) that shows the rules to apply in order.
"""

ENHANCED_COT_PROMPT_TEMPLATE = """
I'm trying to solve a "sed puzzle". In this puzzle, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

Here's the puzzle I'm working on:
- Initial string: {initial_string}
- Available replacements:
{transitions}

I'll solve this step-by-step, carefully tracking how the string changes after each replacement.

Initial string: {initial_string}

Approach:
1. For each step, I'll look for a rule that can be applied to the current string
2. I'll show exactly where the pattern appears and how the string changes
3. I'll continue until I reach an empty string

Working through the solution:
[Current string: {initial_string}]

Step 1:
[Analyze which rules can be applied to the current string]
[Show the exact position where the rule is applied and the resulting string]

Step 2:
[Continue this pattern, showing the exact string after each transformation]

[Continue steps until reaching an empty string]

Verification:
Let me verify my solution by tracing through each step again:
- Initial string: {initial_string}
[Trace through each step to ensure correctness]

Final solution (as a list of rule indices): [a, b, c, ...]

IMPORTANT: The final answer must be just a list of indices in the format [0, 1, 2] representing the rules to apply in order.
"""

class EnhancedLLMEvaluator:
    """An enhanced framework for evaluating LLM performance on sed puzzles with improved accuracy."""
    
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
            template = ENHANCED_ZERO_SHOT_PROMPT_TEMPLATE
        elif prompt_type == "few-shot":
            template = ENHANCED_FEW_SHOT_PROMPT_TEMPLATE
        elif prompt_type == "cot":
            template = ENHANCED_COT_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return template.format(
            initial_string=problem.initial_string,
            transitions=transitions_text
        )
    
    def parse_llm_response_with_retry(self, response):
        """
        Enhanced parser for LLM responses with fallback strategies.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            list: The parsed solution as a list of indices, or None if parsing fails
        """
        # First attempt: Look for standard list pattern [0, 1, 2]
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
                logging.warning(f"Failed to parse solution from: {list_content}")
        
        # Second attempt: Look for numbers preceded by "rule" or "step" with "#" or numbers
        rule_pattern = r'(?:rule|step)\s*(?:#|number|:)?\s*(\d+)'
        rule_matches = re.findall(rule_pattern, response.lower())
        
        if rule_matches:
            try:
                solution = [int(idx) for idx in rule_matches]
                logging.info(f"Extracted solution from rule references: {solution}")
                return solution
            except ValueError:
                logging.warning("Failed to parse solution from rule references")
        
        # Third attempt: Just extract any sequence of digits that might be indices
        digit_pattern = r'\b(\d+)\b'
        digit_matches = re.findall(digit_pattern, response)
        
        if digit_matches:
            # Filter out likely non-indices (very large numbers)
            filtered_digits = [d for d in digit_matches if len(d) < 3]  # Exclude long numbers
            
            if filtered_digits:
                try:
                    solution = [int(idx) for idx in filtered_digits]
                    logging.info(f"Extracted solution from digit sequences: {solution}")
                    return solution
                except ValueError:
                    pass
        
        logging.error(f"No solution list found in response: {response[:100]}...")
        return None
    
    def verify_solution_with_details(self, problem, solution):
        """
        Verify if the provided solution correctly solves the puzzle and provide detailed feedback.
        
        Args:
            problem: The Problem object
            solution: List of transition indices
            
        Returns:
            tuple: (is_correct, verification_details)
        """
        if solution is None:
            return False, ["No solution provided"]
            
        transitions = problem.transitions
        current_string = problem.initial_string
        verification_steps = [f"Initial string: '{current_string}'"]
        
        for step_idx, rule_idx in enumerate(solution):
            if rule_idx >= len(transitions):
                error_msg = f"Step {step_idx+1}: ERROR - Rule {rule_idx} does not exist (max index is {len(transitions)-1})"
                verification_steps.append(error_msg)
                return False, verification_steps
                
            src = transitions[rule_idx].src
            tgt = transitions[rule_idx].tgt
            
            if src in current_string:
                # Find the first occurrence of src
                pos = current_string.find(src)
                # Replace just that occurrence
                new_string = current_string[:pos] + tgt + current_string[pos + len(src):]
                verification_steps.append(f"Step {step_idx+1}: Apply Rule {rule_idx} ({src} → {tgt}) to '{current_string}' → '{new_string}'")
                current_string = new_string
            else:
                # Cannot apply this transition
                error_msg = f"Step {step_idx+1}: ERROR - Cannot apply Rule {rule_idx} ({src} → {tgt}) to '{current_string}' because pattern not found"
                verification_steps.append(error_msg)
                return False, verification_steps
        
        # Check if we reached an empty string
        if current_string == "":
            verification_steps.append("Solution is valid! Final string is empty.")
            return True, verification_steps
        else:
            error_msg = f"ERROR - Final string is not empty: '{current_string}'"
            verification_steps.append(error_msg)
            return False, verification_steps
    
    def evaluate_with_retry(self, llm_client, prompt_type, problem, max_retries=2):
        """
        Evaluate a problem with retries if the solution is incorrect.
        
        Args:
            llm_client: Function that calls the LLM API
            prompt_type: Type of prompt to use
            problem: Problem to solve
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: Evaluation results
        """
        prompt = self.build_prompt(problem, prompt_type)
        response = llm_client(prompt)
        solution = self.parse_llm_response_with_retry(response)
        is_correct, verification_details = self.verify_solution_with_details(problem, solution)
        
        # If solution is incorrect and we have retries left, try again with feedback
        attempt = 1
        while not is_correct and attempt <= max_retries:
            retry_prompt = self._build_retry_prompt(problem, prompt_type, response, verification_details)
            response = llm_client(retry_prompt)
            solution = self.parse_llm_response_with_retry(response)
            is_correct, verification_details = self.verify_solution_with_details(problem, solution)
            attempt += 1
        
        return {
            "prompt_type": prompt_type,
            "solution": solution,
            "is_correct": is_correct,
            "verification_details": verification_details,
            "response": response,
            "attempts": attempt
        }
    
    def _build_retry_prompt(self, problem, prompt_type, previous_response, verification_details):
        """
        Build a retry prompt with feedback on the previous attempt.
        
        Args:
            problem: The Problem object
            prompt_type: Type of prompt to use
            previous_response: The model's previous response
            verification_details: Verification steps from the previous attempt
            
        Returns:
            str: The retry prompt
        """
        # Extract the base prompt
        base_prompt = self.build_prompt(problem, prompt_type)
        
        # Add feedback based on verification details
        feedback = "\n\nYour previous solution was incorrect. Here's what happened:\n"
        for step in verification_details:
            feedback += f"- {step}\n"
        
        feedback += "\nPlease try again and provide the correct solution as a list of indices [a, b, c].\n"
        feedback += "Make sure to carefully track how the string changes after each transformation.\n"
        
        return base_prompt + feedback
    
    def analyze_failure_patterns(self, results):
        """
        Analyze common patterns in failed solutions.
        
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
                if result["solution"] is None:
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
        
        return failures
```

## 2. Enhanced Real LLM Evaluator

```python
"""
Enhanced module for evaluating sed puzzles using real LLM APIs with retry mechanisms.
"""

import os
import json
import time
from pathlib import Path
import logging
import sys
import requests
from typing import Dict, Any, List, Optional

# Add project root to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

# Import our modules
from src.enhanced_evaluator import EnhancedLLMEvaluator
from src.schema import Solution

class EnhancedGPTLLMEvaluator(EnhancedLLMEvaluator):
    """Enhanced class for evaluating sed puzzles using OpenAI's GPT API with retries."""
    
    def __init__(self, api_key, model="gpt-4o", temperature=0.2, **kwargs):
        """
        Initialize the GPT evaluator with enhanced features.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4o)
            temperature: Temperature for generation (lower = more deterministic)
            **kwargs: Additional arguments to pass to EnhancedLLMEvaluator
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def call_gpt_api_with_backoff(self, prompt, max_retries=3, initial_backoff=2):
        """
        Call the OpenAI API with exponential backoff for rate limits.
        
        Args:
            prompt: The prompt to send to the API
            max_retries: Maximum number of retries for rate limit errors
            initial_backoff: Initial backoff time in seconds
            
        Returns:
            str: The model's response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 1024
        }
        
        backoff = initial_backoff
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=data,
                    timeout=30
                )
                
                # Check for rate limit errors
                if response.status_code == 429:
                    if attempt < max_retries:
                        logging.warning(f"Rate limit hit. Backing off for {backoff} seconds...")
                        time.sleep(backoff)
                        backoff *= 2  # Exponential backoff
                        continue
                    else:
                        logging.error("Rate limit error persisted after max retries")
                        return "Error: API rate limit exceeded after multiple retries"
                
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    logging.error(f"Unexpected API response format: {result}")
                    return ""
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    logging.warning(f"API request error: {e}. Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                else:
                    logging.error(f"API request failed after {max_retries} retries: {e}")
                    return f"Error: API request failed - {str(e)}"
        
        return "Error: Maximum retries exceeded"
    
    def evaluate_with_real_llm(self, prompt_type, num_samples=10):
        """
        Evaluate the puzzles using the GPT API with enhanced retry logic.
        
        Args:
            prompt_type: Type of prompt to use ("zero-shot", "few-shot", or "cot")
            num_samples: Number of puzzles to evaluate
            
        Returns:
            dict: Evaluation results
        """
        import random
        results = {}
        
        # Sample a subset of puzzles
        sample_ids = random.sample(list(self.puzzles.keys()), min(num_samples, len(self.puzzles)))
        
        for i, problem_id in enumerate(sample_ids):
            problem = self.puzzles[problem_id]
            
            # Call evaluate with retry
            logging.info(f"[{i+1}/{len(sample_ids)}] Evaluating problem {problem_id} with {prompt_type} prompting...")
            result = self.evaluate_with_retry(
                llm_client=self.call_gpt_api_with_backoff,
                prompt_type=prompt_type,
                problem=problem,
                max_retries=1  # One additional attempt if first solution is incorrect
            )
            
            results[problem_id] = result
            
            logging.info(f"Problem {problem_id} - Solution: {result['solution']} - Correct: {result['is_correct']} - Attempts: {result['attempts']}")
            
            # Add a small delay to avoid API rate limits
            time.sleep(1)
            
        return results
```

## 3. Advanced Metrics for Analysis

```python
"""
Advanced evaluation metrics for analyzing SED puzzle performance.
"""

import json
import os
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
from collections import defaultdict

class EnhancedEvaluationMetrics:
    """
    An enhanced class for computing and visualizing evaluation metrics for sed puzzle solutions.
    """
    
    def __init__(self, results_dir):
        """
        Initialize the metrics calculator.
        
        Args:
            results_dir: Directory containing evaluation result JSON files
        """
        self.results_dir = Path(results_dir)
        
    def load_results(self, results_file):
        """Load results from a JSON file."""
        file_path = self.results_dir / results_file
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def compute_basic_metrics(self, results):
        """Compute basic success rate metrics for a set of results."""
        total = len(results)
        correct = sum(1 for result in results.values() if result["is_correct"])
        success_rate = correct / total if total > 0 else 0
        
        # Add metrics for retry success
        if "attempts" in next(iter(results.values()), {}):
            first_attempt_correct = sum(1 for result in results.values() 
                                       if result["is_correct"] and result["attempts"] == 1)
            retry_success = sum(1 for result in results.values() 
                              if result["is_correct"] and result["attempts"] > 1)
            retry_rate = retry_success / (total - first_attempt_correct) if (total - first_attempt_correct) > 0 else 0
            
            return {
                "total": total,
                "correct": correct,
                "success_rate": success_rate,
                "first_attempt_correct": first_attempt_correct,
                "first_attempt_rate": first_attempt_correct / total if total > 0 else 0,
                "retry_success": retry_success,
                "retry_success_rate": retry_rate
            }
        
        return {
            "total": total,
            "correct": correct,
            "success_rate": success_rate
        }
    
    def analyze_failure_patterns(self, results):
        """Analyze patterns in failed solutions."""
        failures = {
            "parse_failure": 0,
            "state_tracking_error": 0,
            "invalid_rule_index": 0, 
            "incomplete_solution": 0,
            "other_failure": 0
        }
        
        for result in results.values():
            if not result["is_correct"]:
                if result["solution"] is None:
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
    
    def visualize_failure_patterns(self, failure_analysis, output_file):
        """Create a pie chart of failure patterns."""
        if failure_analysis["total"] == 0:
            logging.info("No failures to visualize")
            return
            
        labels = []
        sizes = []
        
        for failure_type, percentage in failure_analysis["percentages"].items():
            if percentage > 0:
                labels.append(f"{failure_type} ({failure_analysis['counts'][failure_type]})")
                sizes.append(percentage)
        
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Failure Pattern Distribution')
        
        plt.savefig(output_file)
        plt.close()
    
    def generate_enhanced_report(self, results_files, output_dir="evaluation/enhanced_reports"):
        """
        Generate a comprehensive report with enhanced metrics.
        
        Args:
            results_files: Dictionary mapping prompt types to result files
            output_dir: Directory to save the report and visualizations
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load results for each prompt type
        all_results = {}
        basic_metrics = {}
        failure_analyses = {}
        
        for prompt_type, file_name in results_files.items():
            all_results[prompt_type] = self.load_results(file_name)
            basic_metrics[prompt_type] = self.compute_basic_metrics(all_results[prompt_type])
            failure_analyses[prompt_type] = self.analyze_failure_patterns(all_results[prompt_type])
            
            # Visualize failure patterns
            self.visualize_failure_patterns(
                failure_analyses[prompt_type], 
                f"{output_dir}/{prompt_type}_failure_patterns.png"
            )
        
        # Generate markdown report
        with open(f"{output_dir}/enhanced_evaluation_report.md", 'w') as f:
            f.write("# Enhanced SED Puzzle Solver Evaluation Report\n\n")
            
            # Basic metrics section
            f.write("## Success Rates\n\n")
            f.write("| Prompt Type | Total | Correct | Success Rate | First Attempt | Retry Success |\n")
            f.write("|------------|-------|---------|--------------|---------------|---------------|\n")
            
            for prompt_type, metrics in basic_metrics.items():
                if "first_attempt_rate" in metrics:
                    f.write(f"| {prompt_type} | {metrics['total']} | {metrics['correct']} | "
                           f"{metrics['success_rate']*100:.1f}% | {metrics['first_attempt_rate']*100:.1f}% | "
                           f"{metrics['retry_success_rate']*100:.1f}% |\n")
                else:
                    f.write(f"| {prompt_type} | {metrics['total']} | {metrics['correct']} | "
                           f"{metrics['success_rate']*100:.1f}% | N/A | N/A |\n")
            
            f.write("\n## Failure Analysis\n\n")
            for prompt_type, analysis in failure_analyses.items():
                f.write(f"### {prompt_type} Failure Distribution\n\n")
                f.write(f"Total failures: {analysis['total']}\n\n")
                f.write("| Failure Type | Count | Percentage |\n")
                f.write("|-------------|-------|------------|\n")
                
                for failure_type, count in analysis["counts"].items():
                    percentage = analysis["percentages"][failure_type]
                    f.write(f"| {failure_type} | {count} | {percentage:.1f}% |\n")
                
                f.write(f"\n![{prompt_type} Failure Patterns]({prompt_type}_failure_patterns.png)\n\n")
            
            # Recommendations based on analysis
            f.write("## Recommendations\n\n")
            
            # Calculate overall most common failure
            all_failures = defaultdict(int)
            for analysis in failure_analyses.values():
                for failure_type, count in analysis["counts"].items():
                    all_failures[failure_type] += count
            
            most_common_failure = max(all_failures.items(), key=lambda x: x[1]) if all_failures else (None, 0)
            
            if most_common_failure[0] == "state_tracking_error":
                f.write("### Key Focus Areas for Improvement\n\n")
                f.write("1. **Improve String State Tracking**\n")
                f.write("   - Enhance prompts to emphasize accurate string state after each transformation\n")
                f.write("   - Add explicit verification steps in the Chain of Thought process\n")
                f.write("   - Implement visual tracking of string transformations in prompts\n\n")
            elif most_common_failure[0] == "parse_failure":
                f.write("### Key Focus Areas for Improvement\n\n")
                f.write("1. **Enhance Output Format Instructions**\n")
                f.write("   - Make output format requirements more prominent in prompts\n")
                f.write("   - Add explicit examples of correctly formatted solutions\n")
                f.write("   - Implement more robust parsing of model outputs\n\n")
            
            # Best performing prompt type
            best_prompt = max(basic_metrics.items(), key=lambda x: x[1]["success_rate"])
            f.write(f"The **{best_prompt[0]}** prompting strategy shows the best performance with a "
                   f"{best_prompt[1]['success_rate']*100:.1f}% success rate and should be prioritized.\n\n")
        
        # Save JSON data
        enhanced_report = {
            "basic_metrics": basic_metrics,
            "failure_analyses": failure_analyses
        }
        
        with open(f"{output_dir}/enhanced_metrics_report.json", 'w') as f:
            json.dump(enhanced_report, f, indent=2)
        
        return enhanced_report
```

## 4. Enhanced Run Script with Improved Features

```python
"""
Enhanced script to run evaluations with real LLMs using improved accuracy techniques.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

# Import the enhanced evaluator modules
from src.enhanced_evaluator import EnhancedLLMEvaluator
from src.enhanced_real_llm_evaluator import EnhancedGPTLLMEvaluator
from src.advanced_metrics import EnhancedEvaluationMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SED puzzles using enhanced accuracy techniques")
    
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of puzzles to evaluate per LLM/prompt combo")
    
    parser.add_argument("--puzzles_dir", type=str, default="dataset/puzzles",
                        help="Directory containing puzzle JSON files")
    
    parser.add_argument("--results_dir", type=str, default="evaluation/enhanced_results",
                        help="Directory to save evaluation results")
    
    parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="API key for OpenAI's GPT (or set OPENAI_API_KEY env var)")
    
    parser.add_argument("--models", type=str, default="gpt-4o",
                        help="Comma-separated list of models to evaluate (gpt-4o,gpt-4,gpt-3.5-turbo)")
    
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature setting for LLM generation (lower = more deterministic)")
    
    parser.add_argument("--max_retries", type=int, default=1,
                        help="Maximum number of solution retry attempts")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Verify API key
    if not args.openai_api_key:
        print("ERROR: OpenAI API key is required.")
        print("You can provide it as a command line argument or environment variable:")
        print("  --openai_api_key / OPENAI_API_KEY")
        sys.exit(1)
    
    # Parse requested models
    requested_models = [m.strip() for m in args.models.split(',')]
    
    # Ensure output directory exists
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a dictionary to hold results from all models and prompt types
    all_results = {}
    
    # Evaluate each model with each prompt type
    for model in requested_models:
        all_results[model] = {}
        
        print(f"\n=== Evaluating {model} ===")
        
        # Initialize the evaluator
        evaluator = EnhancedGPTLLMEvaluator(
            api_key=args.openai_api_key,
            model=model,
            temperature=args.temperature,
            puzzles_dir=args.puzzles_dir,
            output_dir=args.results_dir
        )
        
        # Evaluate with each prompt type
        for prompt_type in ["zero-shot", "few-shot", "cot"]:
            print(f"\nEvaluating {model} with {prompt_type} prompting...")
            
            results = evaluator.evaluate_with_real_llm(prompt_type, num_samples=args.num_samples)
            all_results[model][prompt_type] = results
            
            # Save results
            output_file = f"{model}_{prompt_type}_enhanced_results.json"
            evaluator.save_results(results, output_file)
            
            # Compute basic metrics
            metrics = evaluator.compute_basic_metrics(results)
            print(f"Success rate: {metrics['correct']}/{metrics['total']} ({metrics['success_rate']*100:.1f}%)")
    
    # Generate enhanced evaluation report
    print("\n=== Generating Enhanced Evaluation Report ===")
    metrics_calculator = EnhancedEvaluationMetrics(results_dir=args.results_dir)
    
    for model in requested_models:
        results_files = {
            "zero-shot": f"{model}_zero-shot_enhanced_results.json",
            "few-shot": f"{model}_few-shot_enhanced_results.json",
            "cot": f"{model}_cot_enhanced_results.json"
        }
        
        report = metrics_calculator.generate_enhanced_report(
            results_files, 
            output_dir=f"{args.results_dir}/{model}_analysis"
        )
        
        print(f"\nEnhanced evaluation report for {model} generated.")
        print(f"Report saved to {args.results_dir}/{model}_analysis/")

if __name__ == "__main__":
    main()
```

## 5. Improved Testing Script

```python
"""
Improved testing script for the SED puzzle solver.
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Add project root to path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from src.schema import Problem, Solution, Transition
from src.enhanced_evaluator import EnhancedLLMEvaluator

def test_string_state_tracking():
    """Test the string state tracking verification logic."""
    # Create a simple test problem
    problem = Problem(
        problem_id="test_001",
        initial_string="ABCDEF",
        transitions=[
            Transition(src="ABC", tgt="X"),
            Transition(src="XD", tgt="Y"),
            Transition(src="YEF", tgt="")
        ]
    )
    
    # Create an evaluator instance
    evaluator = EnhancedLLMEvaluator(puzzles_dir=".", output_dir=".")
    
    # Test correct solution
    correct_solution = [0, 1, 2]  # ABC->X, XD->Y, YEF->""
    is_correct, details = evaluator.verify_solution_with_details(problem, correct_solution)
    
    logging.info("=== Testing Correct Solution ===")
    for step in details:
        logging.info(step)
    logging.info(f"Is solution correct? {is_correct}\n")
    
    # Test incorrect solution with state tracking error
    incorrect_solution = [0, 2]  # This fails because after step 0, we have "XDEF" and rule 2 can't be applied
    is_correct, details = evaluator.verify_solution_with_details(problem, incorrect_solution)
    
    logging.info("=== Testing Incorrect Solution (State Tracking Error) ===")
    for step in details:
        logging.info(step)
    logging.info(f"Is solution correct? {is_correct}\n")
    
    # Test solution with invalid rule index
    invalid_solution = [0, 1, 3]  # Rule 3 doesn't exist
    is_correct, details = evaluator.verify_solution_with_details(problem, invalid_solution)
    
    logging.info("=== Testing Solution with Invalid Rule Index ===")
    for step in details:
        logging.info(step)
    logging.info(f"Is solution correct? {is_correct}\n")
    
    # Test incomplete solution
    incomplete_solution = [0, 1]  # Doesn't reach empty string
    is_correct, details = evaluator.verify_solution_with_details(problem, incomplete_solution)
    
    logging.info("=== Testing Incomplete Solution ===")
    for step in details:
        logging.info(step)
    logging.info(f"Is solution correct? {is_correct}\n")

def test_response_parsing():
    """Test the enhanced response parsing logic."""
    evaluator = EnhancedLLMEvaluator(puzzles_dir=".", output_dir=".")
    
    # Test standard format
    standard_response = "The solution is [0, 1, 2]."
    solution = evaluator.parse_llm_response_with_retry(standard_response)
    logging.info(f"Standard format: {solution}")
    
    # Test rule reference format
    rule_response = "I'll apply rule #0, then rule #1, and finally rule #2."
    solution = evaluator.parse_llm_response_with_retry(rule_response)
    logging.info(f"Rule reference format: {solution}")
    
    # Test step format
    step_response = "Step 1: Apply rule 2. Step 2: Apply rule 0. Step 3: Apply rule 1."
    solution = evaluator.parse_llm_response_with_retry(step_response)
    logging.info(f"Step format: {solution}")
    
    # Test difficult format
    difficult_response = "First we need to use transformation number 3, followed by number 0, and then number 2."
    solution = evaluator.parse_llm_response_with_retry(difficult_response)
    logging.info(f"Difficult format: {solution}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the enhanced SED puzzle solver")
    parser.add_argument("--test", choices=["tracking", "parsing", "all"], default="all",
                      help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "tracking" or args.test == "all":
        test_string_state_tracking()
    
    if args.test == "parsing" or args.test == "all":
        test_response_parsing()
```

These implementations address the key issues affecting accuracy in the SED Puzzle Solver:

1. **String State Tracking**: Enhanced prompts emphasize precise tracking of string state, with explicit verification steps.
2. **Output Format Parsing**: More robust parsing with multiple fallback strategies to extract solutions.
3. **Retry Mechanisms**: Solution retry with detailed feedback when errors occur.
4. **Enhanced Analytics**: Comprehensive failure analysis to identify and address specific issues.

By implementing these improvements, the accuracy of the SED Puzzle Solver should increase significantly, particularly for the most common failure modes identified in the evaluation results.