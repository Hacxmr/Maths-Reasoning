# s/Math + AI/ Reasoning in LLMs

![a demonstration of the sed puzzle](imgs/image.png)

This project investigates the reasoning capabilities of Large Language Models (LLMs) in solving string replacement puzzles inspired by the Unix `sed` command. These puzzles require the application of a sequence of string replacements to transform an initial string into an empty string.

## Executive Summary

Our enhanced SED puzzle solver implementation significantly outperformed the original implementation across all prompting strategies:

- **Chain of Thought**: 80.0% success rate (vs. 66.7% original) - 20.0% relative improvement
- **Few Shot**: 50.0% success rate (vs. 20.0% original) - 150.0% relative improvement 
- **Zero Shot**: 80.0% success rate (vs. 10.0% original) - 700.0% relative improvement

These impressive gains were achieved through:

1. Advanced prompt engineering with explicit state tracking guidance
2. Robust solution parsing and verification
3. Intelligent retry mechanisms for failed attempts
4. Comprehensive error analysis and categorization

This evaluation was conducted on multiple sets of puzzles, providing a meaningful comparison between the implementations across different prompting strategies.

## Project Structure

```
sed-solver/
├── src/                  # Source code
│   ├── schema.py         # Data models for puzzles and solutions
│   ├── utils.py          # Utility functions
│   ├── baseline.py       # Baseline solver using BFS
│   ├── generator.py      # Puzzle generator
│   ├── evaluator.py      # Original LLM evaluation framework
│   ├── enhanced_evaluator.py  # Enhanced evaluation with improved prompts
│   └── metrics.py        # Evaluation metrics
├── dataset/              # Generated dataset
│   ├── puzzles/          # Puzzle JSON files
│   └── solutions/        # Solution JSON files (for validation)
├── evaluation/           # Evaluation results and analysis
│   ├── results/          # Raw results from LLM evaluations
│   ├── reports/          # Metrics and visualizations
│   └── report.md         # Comprehensive findings report
├── sample-data/          # Example puzzles provided in the original repo
├── generate_100_puzzles.py  # Script to generate the dataset
├── compare_implementations.py  # Script to compare original vs enhanced implementations
├── test_evaluator.py     # Script to test the evaluator
├── report.md             # Comprehensive project report
└── run_metrics.py        # Script to compute evaluation metrics
```

## Core Components

### Data Structures (`schema.py`)
We define the data schema using [Pydantic](https://docs.pydantic.dev/latest/) to ensure data validation. Key models include:
- `Problem`: Represents an SED puzzle with initial string and transitions
- `Solution`: Contains the sequence of rule indices to solve a puzzle
- `LLMResponse`: Captures the response from an LLM evaluation

### Utilities (`utils.py`)
- `read_problem_folder()`/`read_solution_folder()`: Load puzzles/solutions from JSON files
- `write_problem_folder()`/`write_solution_folder()`: Save puzzles/solutions to JSON files
- `validate_solutions()`: Verify solution correctness for given problems

### Baseline Solver (`baseline.py`)
A breadth-first search implementation that systematically explores the solution space until reaching the empty string, providing a traditional algorithmic benchmark.

## Enhanced Implementation

### Prompt Engineering

We developed and tested multiple prompting strategies:

1. **Zero-shot**: Direct instruction without examples
   ```
   I'm trying to solve a "sed puzzle"...
   ```

2. **Few-shot**: Includes examples of solved puzzles
   ```
   I'm trying to solve a "sed puzzle". Let me show you a few examples...
   ```

3. **Chain of Thought (CoT)**: Encourages step-by-step reasoning
   ```
   I'll solve this step-by-step, tracking how the string changes...
   ```

4. **Enhanced CoT**: Our improved implementation with precise guidance
   ```
   I'll solve this step-by-step, EXTREMELY CAREFULLY tracking how the string changes after each replacement.

   CRITICAL APPROACH:
   1. For each step, I'll examine ALL rules to find one that can be applied
   2. I'll show the PRECISE position where the pattern appears
   3. I'll double-check each transformation
   ```

### Solution Parsing

The enhanced implementation includes robust parsing strategies:
- Multiple regex patterns to extract solution sequences
- Fallback mechanisms for different response formats
- Detailed error reporting for failed parsing

### Retry Mechanism

Our implementation includes an intelligent retry system that:
1. Analyzes failure types (parsing errors vs. state tracking errors)
2. Provides targeted feedback in retry prompts
3. Suggests specific corrections based on identified issues

## Dataset Generation

We generated 100 puzzles with controlled difficulty levels:

| Difficulty | Count | Characteristics |
|------------|-------|-----------------|
| Easy       | 25    | Few rules, simple patterns |
| Medium     | 40    | Medium-length strings, moderate complexity |
| Hard       | 30    | Many rules, complex interdependencies |
| Very Hard  | 5     | Long strings, challenging rule interactions |

Generate your own dataset with:
```bash
python generate_100_puzzles.py
```

## Evaluation

### Comparing Implementations

Compare the original and enhanced implementations with:
```bash
python compare_implementations.py
```

### Results Summary

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

### Failure Analysis

| Failure Type | Original | Enhanced |
|--------------|----------|----------|
| Parse failures | 60.0% of failures | 0.0% (eliminated) |
| State tracking | 0.0% of failures | 100.0% of failures |
| Other failures | 40.0% of failures | 0.0% of failures |

## Using the Framework

### Setting Up

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set API keys as environment variables:
```bash
export OPENAI_API_KEY=your_openai_api_key
export OPENROUTER_API_KEY=your_openrouter_api_key
```

### Running Evaluations

With OpenAI's GPT-4o (default):
```bash
python run_real_llm_eval.py --num_samples 100
```

With OpenRouter models:
```bash
python run_real_llm_eval.py --num_samples 100 --models openrouter
```

### Generating Reports

Create comparative reports:
```bash
python generate_llm_report.py
```

## Comprehensive Report

For detailed analysis, methodologies, and findings, see the [full report](report.md).

## Prerequisites

- Python 3.8+
- Required packages:
  - pydantic
  - numpy
  - matplotlib
  - openai
  - tiktoken (for OpenAI token counting)
  - tqdm (for progress bars)

