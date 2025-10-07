"""
Enhanced evaluator module with improved accuracy features for the SED Puzzle Solver.
"""

import json
import logging
import os
import random
import re
import time
from pathlib import Path

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

I'll solve this step-by-step, EXTREMELY CAREFULLY tracking how the string changes after each replacement.

Initial string: {initial_string}

CRITICAL APPROACH:
1. For each step, I'll carefully examine ALL rules to find one that can be applied to the EXACT current string
2. I'll show the PRECISE position where the pattern appears and the resulting string with the replacement applied
3. I'll double-check each transformation by visually inspecting the string before and after
4. I'll continue until I reach an empty string, verifying each step

Working through the solution:
[Current string: {initial_string}]

Step 1:
- I'll examine each rule carefully to find one where the LEFT side pattern EXACTLY matches a substring in the current string
- I'll mark the EXACT positions where the match occurs using | | markers
- After finding a match, I'll apply the rule and show the EXACT resulting string

Step 2:
[Continue this pattern, showing the exact string after each transformation]

[Continue steps until reaching an empty string]

Detailed Verification:
After reaching the empty string, I'll verify my solution by tracing through each step again:
- Initial string: {initial_string}
[Trace through each step one more time, checking that each rule application is valid]

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
        
        # Import here to avoid circular imports
        from src.utils import read_problem_folder

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
        Build a retry prompt with detailed feedback on the previous attempt.
        
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
        
        # Determine the specific error type from verification details
        error_type = "general"
        error_message = ""
        
        for step in verification_details:
            if "ERROR" in step:
                if "pattern not found" in step:
                    error_type = "pattern_not_found"
                    # Extract the rule and pattern that failed
                    import re
                    match = re.search(r"Rule (\d+) \(([^→]+) → ([^)]+)\)", step)
                    if match:
                        rule_num, src, tgt = match.groups()
                        error_message = f"Rule {rule_num} failed because '{src.strip()}' was not found in the string."
                elif "does not exist" in step:
                    error_type = "invalid_rule"
                    match = re.search(r"Rule (\d+)", step)
                    if match:
                        rule_num = match.group(1)
                        error_message = f"Rule {rule_num} does not exist in the available rules."
        
        # Add feedback based on verification details
        feedback = "\n\nYour previous solution was incorrect. Here's what happened:\n"
        for step in verification_details:
            feedback += f"- {step}\n"
        
        feedback += "\n"
        
        # Add specific guidance based on error type
        if error_type == "pattern_not_found":
            feedback += f"IMPORTANT ERROR DETAIL: {error_message}\n"
            feedback += "You need to ensure that each rule is applied only when its pattern EXACTLY matches the current string.\n"
            feedback += "Review the available rules very carefully and ensure that at each step you're applying a rule that matches the current state of the string.\n"
        elif error_type == "invalid_rule":
            feedback += f"IMPORTANT ERROR DETAIL: {error_message}\n"
            feedback += "Make sure you're only using valid rule indices from the available rules.\n"
        else:
            feedback += "Make sure you're tracking the string transformation precisely at each step.\n"
            
        feedback += "\nPlease try again and provide the correct solution as a list of indices [a, b, c].\n"
        feedback += "IMPORTANT: For each step, verify that the pattern on the left side of the rule EXACTLY matches a portion of your current string before applying the rule.\n"
        
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
        
    def save_results(self, results, filename):
        """Save evaluation results to a JSON file."""
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = {}
        for problem_id, result in results.items():
            serializable_results[problem_id] = {
                "prompt_type": result["prompt_type"],
                "solution": result["solution"] if result["solution"] is not None else None,
                "is_correct": result["is_correct"],
                "attempts": result.get("attempts", 1)
            }
            
            # Add verification details if available
            if "verification_details" in result:
                serializable_results[problem_id]["verification_details"] = result["verification_details"]
                
            # Add truncated response (to save space)
            if "response" in result:
                response = result["response"]
                serializable_results[problem_id]["response_excerpt"] = response[:500] + "..." if len(response) > 500 else response
            
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logging.info(f"Results saved to {output_path}")
    
    def compute_basic_metrics(self, results):
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
        
        logging.info(f"Metrics: {metrics}")
        return metrics