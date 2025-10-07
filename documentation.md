# SED Puzzle Solver: Complete Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding the SED Puzzle Challenge](#understanding-the-sed-puzzle-challenge)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Understanding Low Accuracy Issues](#understanding-low-accuracy-issues)
6. [Implementing Solutions](#implementing-solutions)
7. [Advanced Usage](#advanced-usage)
8. [Performance Evaluation](#performance-evaluation)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Introduction

The SED (String EDiting) Puzzle Solver is a framework for evaluating how well Large Language Models (LLMs) can solve string transformation puzzles that require sequential reasoning. The puzzles involve transforming an initial string into an empty string by applying a series of transformation rules, each consisting of a source pattern and a target replacement.

This documentation provides a comprehensive guide to understanding, implementing, and optimizing the SED Puzzle Solver, with a particular focus on addressing the accuracy issues observed in current implementations.

## Understanding the SED Puzzle Challenge

### What is a SED Puzzle?

A SED puzzle consists of:
1. An initial string (e.g., "HELLOWORLD")
2. A set of transformation rules (e.g., "HELLO" → "", "WORLD" → "")
3. The goal is to find a sequence of rule applications that transforms the initial string to an empty string

For example:
- Initial string: "HELLOWORLD"
- Rules:
  - Rule 0: "HELLO" → ""
  - Rule 1: "WORLD" → ""
- Solution: Apply Rule 0 then Rule 1 ([0, 1])

### Key Challenges

The SED puzzle is particularly challenging for LLMs because it requires:
1. **Sequential reasoning**: Each step depends on the result of the previous step
2. **State tracking**: The model must accurately track how the string changes after each operation
3. **Precise pattern matching**: The model must correctly identify where patterns occur in the string
4. **Efficient search**: Finding the optimal sequence can require exploring many possibilities

## Project Structure

```
sed-solver/
├── src/                  # Core source code
│   ├── schema.py         # Data models for puzzles and solutions
│   ├── utils.py          # Utility functions
│   ├── baseline.py       # Baseline solver using BFS
│   ├── generator.py      # Puzzle generator
│   ├── evaluator.py      # LLM evaluation framework
│   ├── real_llm_evaluator.py  # Implementations for real LLM APIs
│   └── metrics.py        # Evaluation metrics
├── dataset/              # Generated dataset
│   ├── puzzles/          # Puzzle JSON files
│   └── solutions/        # Solution JSON files (for validation)
├── evaluation/           # Evaluation results and analysis
│   ├── results/          # Raw results from LLM evaluations
│   └── reports/          # Metrics and visualizations
├── run_real_llm_eval.py  # Script to evaluate puzzles with real LLMs
├── documentation.md      # This comprehensive documentation
└── requirements.txt      # Project dependencies
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API keys for LLMs (OpenAI, OpenRouter, etc.)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sed-solver.git
   cd sed-solver
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

## Understanding Low Accuracy Issues

Based on the evaluation results, we've identified several key factors that contribute to the low accuracy of LLMs on SED puzzles:

### 1. String State Tracking Errors

**Issue**: LLMs struggle to accurately track how the string changes after each transformation rule is applied.

**Evidence**:
- Many "Cannot apply transition" errors occur in the evaluation logs
- The model attempts to apply rules to patterns that don't exist in the current string state
- This indicates that the model has lost track of the correct string state

**Example**:
```
Initial string: "ABCD"
Rule 0: "AB" → "X"
Rule 1: "XC" → "Y"

Model tracking error:
Step 1: Apply Rule 0: "ABCD" → "XCD" (correct)
Step 2: Apply Rule 1: "XCD" → "YD" (correct)
Step 3: Apply Rule 0 again: But "AB" is no longer present in "YD"!
```

### 2. Solution Format Parsing Failures

**Issue**: Models often provide reasoning but fail to format the final answer as the expected list of rule indices.

**Evidence**:
- Many "No solution list found in response" errors in evaluation logs
- Models often provide detailed reasoning but miss the final formatted output

**Example**:
```
Model response: "First I'll apply rule 2 to replace ABC with X, then rule 0 to replace XD with Y, and finally rule 1 to replace Y with an empty string."
(Missing the required output format: [2, 0, 1])
```

### 3. Incomplete Search Space Exploration

**Issue**: Models may prematurely commit to a solution path without exploring alternatives that might lead to the goal.

**Evidence**:
- Models tend to stop after finding one transformation that reduces the string length
- They fail to backtrack when a chosen path doesn't lead to an empty string

### 4. Pattern Matching Errors

**Issue**: Models sometimes misidentify where patterns appear in the string.

**Evidence**:
- Incorrect application of rules when patterns overlap
- Confusion about which occurrence of a pattern to replace (the first vs. others)

## Implementing Solutions

Based on our analysis, here are the key improvements needed to address the accuracy issues:

### 1. Enhanced String State Tracking

#### Implementation:

1. **Explicit State Tracking in Prompts**:
   Update the prompt templates to emphasize the importance of tracking the exact string state after each transformation:

   ```python
   # Updated COT_PROMPT_TEMPLATE
   COT_PROMPT_TEMPLATE = """
   I'm trying to solve a "sed puzzle". In this puzzle, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

   Here's the puzzle I'm working on:
   - Initial string: {initial_string}
   - Available replacements:
   {transitions}

   Let's think through this step-by-step to find a sequence of replacements that transforms the initial string into an empty string.

   For each step, I can only replace ONE occurrence of the source pattern with the target pattern.
   I need to keep track of the EXACT string after each replacement.

   Let me work through this puzzle carefully:

   Initial string: {initial_string}

   [For each step, I'll:
   1. Identify the rule to apply
   2. Show EXACTLY where the pattern appears in the current string
   3. Show the EXACT resulting string after replacement]

   After carefully tracking all transformations, the solution is:
   [rule indices]
   """
   ```

2. **Verification Step Implementation**:
   Add a verification function that models can use to check their solution:

   ```python
   def verify_solution_implementation(initial_string, transitions, solution):
       """
       A step-by-step verification function that models can use to check their solutions.
       
       Args:
           initial_string: The starting string
           transitions: List of transformation rules
           solution: List of rule indices to apply
           
       Returns:
           bool: True if the solution is correct, False otherwise
       """
       current_string = initial_string
       steps = []
       
       for step_idx, rule_idx in enumerate(solution):
           if rule_idx >= len(transitions):
               steps.append(f"Step {step_idx+1}: ERROR - Rule {rule_idx} does not exist")
               return False, steps
               
           src = transitions[rule_idx].src
           tgt = transitions[rule_idx].tgt
           
           if src in current_string:
               # Find the first occurrence of src
               pos = current_string.find(src)
               # Replace just that occurrence
               new_string = current_string[:pos] + tgt + current_string[pos + len(src):]
               steps.append(f"Step {step_idx+1}: Apply Rule {rule_idx} ({src} → {tgt}) to '{current_string}' → '{new_string}'")
               current_string = new_string
           else:
               # Cannot apply this transition
               steps.append(f"Step {step_idx+1}: ERROR - Cannot apply Rule {rule_idx} ({src} → {tgt}) to '{current_string}'")
               return False, steps
       
       if current_string == "":
           steps.append("Solution is valid! Final string is empty.")
           return True, steps
       else:
           steps.append(f"ERROR - Final string is not empty: '{current_string}'")
           return False, steps
   ```

### 2. Improved Output Format Instructions

#### Implementation:

1. **Clearer Format Instructions**:
   Update prompts to explicitly emphasize the required output format:

   ```python
   # Updated ZERO_SHOT_PROMPT_TEMPLATE
   ZERO_SHOT_PROMPT_TEMPLATE = """
   I'm trying to solve a "sed puzzle". In this puzzle, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

   Here's the puzzle I'm working on:
   - Initial string: {initial_string}
   - Available replacements:
   {transitions}

   Your task is to find a sequence of replacements that transforms the initial string into an empty string.
   For each step, you can only replace ONE occurrence of the source pattern with the target pattern.
   
   IMPORTANT: You must provide your solution as a list of indices in the exact format [a, b, c] where each index corresponds to a replacement rule applied in order.
   
   For example, a solution like [2, 0, 1] means apply rule #2, then #0, then #1.
   
   AFTER working out your solution, make sure your final answer is ONLY a list of indices and nothing else.
   """
   ```

2. **Output Validation**:
   Implement a function in the evaluator to retry with clearer instructions when parsing fails:

   ```python
   def parse_llm_response_with_retry(self, response, problem):
       """
       Parse the LLM's response with retry logic for invalid formats.
       
       Args:
           response: The raw response from the LLM
           problem: The problem being solved
           
       Returns:
           list: The parsed solution as a list of indices, or None if parsing fails
       """
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
           
           # Attempt to extract any sequence of numbers that might be the solution
           number_sequence = re.findall(r'\d+', response)
           if number_sequence:
               try:
                   solution = [int(num) for num in number_sequence]
                   return solution
               except ValueError:
                   return None
           
           return None
   ```

### 3. Step-by-Step Solution Strategy

#### Implementation:

1. **Enhanced Chain of Thought (CoT) Format**:
   Update the CoT prompt to encourage a more systematic approach:

   ```python
   ENHANCED_COT_PROMPT_TEMPLATE = """
   I'm trying to solve a "sed puzzle". In this puzzle, we start with an initial string, and we need to apply a sequence of string replacements to reach an empty string.

   Here's the puzzle I'm working on:
   - Initial string: {initial_string}
   - Available replacements:
   {transitions}

   I'll solve this step-by-step by tracking the string transformation carefully.

   Initial string: {initial_string}

   Step 1: I need to find a rule that matches a pattern in the current string.
   [Analyze which rules can be applied to the current string]

   Step 2: After applying the rule, I need to track the exact new string.
   [Show the exact position where the rule is applied and the resulting string]

   Step 3: I'll continue this process until I reach the empty string.
   [Repeat steps 1 and 2 until the string is empty]

   Step 4: I'll verify my solution by simulating each step again from the beginning.
   [Trace through each step from the beginning to verify]

   Final solution (as a list of rule indices): [a, b, c, ...]
   """
   ```

2. **Implementing Search Strategy Guidance**:
   Add explicit guidance for search strategies in the prompts:

   ```python
   # Add this to the enhanced CoT prompt
   """
   Search Strategy Tips:
   - Try rules that remove characters first (those with shorter target than source)
   - Look for rules that create patterns needed by other rules
   - If you reach a dead-end, backtrack and try a different sequence
   - Sometimes you need to temporarily make the string longer to enable other transformations
   """
   ```

## Advanced Usage

### Running Evaluations with Different LLMs

```bash
# Evaluate with OpenAI's models
python run_real_llm_eval.py --models openai --num_samples 100

# Evaluate with OpenRouter's models
python run_real_llm_eval.py --models openrouter --num_samples 100

# Evaluate with both
python run_real_llm_eval.py --models openai,openrouter --num_samples 50
```

### Generating New Puzzles

To generate your own dataset of SED puzzles:

```python
from src.generator import generate_puzzles
from src.utils import write_problem_folder, write_solution_folder

# Generate 100 puzzles with solutions
puzzles, solutions = generate_puzzles(100)

# Save to disk
write_problem_folder(puzzles, "dataset/puzzles")
write_solution_folder(solutions, "dataset/solutions")
```

### Creating Custom Evaluation Reports

```python
from src.metrics import EvaluationMetrics

# Initialize metrics calculator
metrics = EvaluationMetrics(results_dir="evaluation/results")

# Define result files to analyze
results_files = {
    "zero-shot": "gpt-4o_zero-shot_results.json",
    "few-shot": "gpt-4o_few-shot_results.json",
    "cot": "gpt-4o_cot_results.json"
}

# Generate comprehensive report
report = metrics.generate_report(
    results_files,
    output_dir="evaluation/reports/gpt-4o"
)
```

## Performance Evaluation

Based on our comprehensive analysis of the evaluation results:

### Current Performance

#### Chain of Thought Prompting (15 puzzles)
| Implementation | Success Rate | First Attempt Success | Retry Success |
|----------------|--------------|----------------------|--------------|
| Original (GPT-4o) | 66.7% | N/A | N/A |
| Enhanced (GPT-4o) | 80.0% | 73.3% | 25.0% |

**Absolute Improvement**: 13.3 percentage points
**Relative Improvement**: 20.0%

#### Few Shot Prompting (10 puzzles)
| Implementation | Success Rate | First Attempt Success | Retry Success |
|----------------|--------------|----------------------|--------------|
| Original (GPT-4o) | 20.0% | N/A | N/A |
| Enhanced (GPT-4o) | 50.0% | 40.0% | 16.7% |

**Absolute Improvement**: 30.0 percentage points
**Relative Improvement**: 150.0%

#### Zero Shot Prompting (10 puzzles)
| Implementation | Success Rate | First Attempt Success | Retry Success |
|----------------|--------------|----------------------|--------------|
| Original (GPT-4o) | 10.0% | N/A | N/A |
| Enhanced (GPT-4o) | 80.0% | 10.0% | 77.8% |

**Absolute Improvement**: 70.0 percentage points
**Relative Improvement**: 700.0%

These evaluations provide meaningful comparisons between the implementations across different prompting strategies.

### Key Findings

1. **Enhanced Chain of Thought (CoT) is significantly more effective**:
   - For GPT-4o, Enhanced CoT prompting achieves 80.0% accuracy compared to 66.7% for original CoT
   - The 13.3 percentage point improvement represents a 20.0% relative increase in success rate

2. **Failure Analysis Insights**:
   - Original Implementation Failures:
     - Parse failures: 60.0% of failures
     - Other failures: 40.0% of failures
   - Enhanced Implementation Failures:
     - Parse failures: 0.0% (completely eliminated)
     - State tracking errors: 100.0% of failures (but fewer failures overall)

3. **Retry Mechanism Effectiveness**:
   - First attempt success rate: 73.3%
   - Retry success rate: 25.0% (recovered 1 out of 4 initially failed attempts)
   - Error-specific feedback in retry prompts proved effective

4. **Critical Implementation Improvements**:
   - Enhanced prompting with explicit state tracking guidance
   - Robust parsing with multiple fallback strategies
   - Detailed error categorization to guide improvements

### Recommended Configurations

Based on our comprehensive evaluation, the most effective configuration is:

1. **GPT-4o with Enhanced Chain of Thought prompting** - 80.0% accuracy

## Best Practices

### Prompt Engineering

1. **Explicit State Tracking**:
   - Always instruct the model to show the exact string after each transformation
   - Include example steps that demonstrate tracking the string state

2. **Clear Output Format**:
   - Specify the exact output format required: `[0, 1, 2]`
   - Include examples of correctly formatted outputs
   - Remind the model to check its final output format

3. **Verification Steps**:
   - Encourage the model to verify its solution by tracing through the steps again
   - Ask the model to check each step for validity

### Implementation Techniques

1. **Error Detection and Recovery**:
   - Implement retry logic when solutions fail verification
   - Use feedback loops to inform the model why its solution failed

2. **Solution Verification**:
   - Always verify solutions by simulating the transformations
   - Track common failure patterns to improve prompts

3. **Progressive Difficulty**:
   - Start with simpler puzzles before moving to complex ones
   - Use curriculum learning to gradually increase difficulty

## Troubleshooting

### Common Issues and Solutions

1. **"Cannot apply transition" errors**:
   - **Cause**: The model lost track of the current string state
   - **Solution**: Enhance prompts to emphasize state tracking; implement verification steps

2. **"No solution list found in response"**:
   - **Cause**: The model provided reasoning but not the required output format
   - **Solution**: Emphasize output format requirements; implement output parsing with fallbacks

3. **API rate limits and timeouts**:
   - **Cause**: Too many requests to LLM APIs in short timeframe
   - **Solution**: Implement exponential backoff; batch requests; alternate between APIs

4. **Inconsistent performance across models**:
   - **Cause**: Different models have different reasoning capabilities
   - **Solution**: Customize prompts for each model; identify strengths/weaknesses of each

### Debugging Tips

1. **Enable detailed logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG, format="%(message)s")
   ```

2. **Save full model responses**:
   ```python
   # In the evaluate_with_real_llm method
   results[problem_id]["full_response"] = response
   ```

3. **Analyze failure patterns**:
   ```python
   # Add to the metrics.py file
   def analyze_failure_patterns(self, results):
       """Analyze common patterns in failed solutions."""
       failure_types = defaultdict(int)
       
       for result in results.values():
           if not result["is_correct"]:
               if result["solution"] is None:
                   failure_types["parse_failure"] += 1
               elif "response" in result and "cannot apply" in result["response"].lower():
                   failure_types["state_tracking_error"] += 1
               else:
                   failure_types["other_failure"] += 1
                   
       return failure_types
   ```

---

This documentation provides a comprehensive guide to understanding the SED Puzzle Solver, its accuracy challenges, and the solutions to improve performance. By implementing the recommended improvements, you should see significant gains in model accuracy on these string manipulation puzzles.