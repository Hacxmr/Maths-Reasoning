"""
Script to run evaluations with real LLMs using various prompting techniques.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

# Import the real LLM evaluator
from src.real_llm_evaluator import evaluate_with_multiple_llms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SED puzzles using real LLMs")
    
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of puzzles to evaluate per LLM/prompt combo")
    
    parser.add_argument("--puzzles_dir", type=str, default="dataset/puzzles",
                        help="Directory containing puzzle JSON files")
    
    parser.add_argument("--results_dir", type=str, default="evaluation/results",
                        help="Directory to save evaluation results")
    
    parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="API key for OpenAI's GPT (or set OPENAI_API_KEY env var)")
    
    parser.add_argument("--openrouter_api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY"),
                        help="API key for OpenRouter (or set OPENROUTER_API_KEY env var)")
    
    parser.add_argument("--models", type=str, default="openai",
                        help="Comma-separated list of models to evaluate (openai,openrouter)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if we have any API keys
    api_keys = {}
    
    # Parse requested models if specified
    requested_models = []
    if args.models:
        requested_models = [m.strip().lower() for m in args.models.split(',')]
    
    if args.openai_api_key and (not requested_models or 'openai' in requested_models or 'gpt' in requested_models):
        api_keys["openai"] = args.openai_api_key
        
    if args.openrouter_api_key and (not requested_models or 'openrouter' in requested_models):
        api_keys["openrouter"] = args.openrouter_api_key
    
    if not api_keys:
        print("ERROR: No API keys provided or no requested models are available.")
        print("You can provide API keys as command line arguments or environment variables:")
        print("  --openai_api_key / OPENAI_API_KEY")
        print("  --openrouter_api_key / OPENROUTER_API_KEY")
        print("And you can specify which models to use with --models (comma-separated):")
        print("  --models openai,openrouter")
        sys.exit(1)
    
    # Run the evaluations
    evaluate_with_multiple_llms(
        puzzles_dir=args.puzzles_dir,
        results_dir=args.results_dir,
        api_keys=api_keys,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()