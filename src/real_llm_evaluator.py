"""
Module for evaluating sed puzzles using real LLM APIs.
Supports Gemini API (with free tier) and can be extended for other LLMs.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Add project root to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

# Import our modules
from src.evaluator import LLMEvaluator
from src.schema import Solution

# Removed GeminiLLMEvaluator and AnthropicLLMEvaluator as they're no longer needed


class GPTLLMEvaluator(LLMEvaluator):
    """Class for evaluating sed puzzles using OpenAI's GPT API."""
    
    def __init__(self, api_key, model="gpt-4o", **kwargs):
        """
        Initialize the GPT evaluator.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4o)
            **kwargs: Additional arguments to pass to LLMEvaluator
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def call_gpt_api(self, prompt):
        """
        Call the OpenAI API with the given prompt.
        
        Args:
            prompt: The prompt to send to the API
            
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
            "temperature": 0.1,  # Low temperature for more deterministic responses
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logging.error(f"Unexpected API response format: {result}")
                return ""
                
        except Exception as e:
            logging.error(f"API call error: {e}")
            return ""
    
    def evaluate_with_real_llm(self, prompt_type, num_samples=10):
        """
        Evaluate the puzzles using the GPT API.
        
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
            
            # Build the prompt
            prompt = self.build_prompt(problem, prompt_type)
            
            # Call the API
            logging.info(f"[{i+1}/{len(sample_ids)}] Evaluating problem {problem_id} with {prompt_type} prompting...")
            response = self.call_gpt_api(prompt)
            
            # Parse the response
            solution = self.parse_llm_response(response)
            
            # Verify the solution
            is_correct = self.verify_solution(problem, solution)
            
            results[problem_id] = {
                "prompt_type": prompt_type,
                "solution": solution,
                "is_correct": is_correct,
                "response": response  # Store the full response for analysis
            }
            
            logging.info(f"Problem {problem_id} - Solution: {solution} - Correct: {is_correct}")
            
            # Add a small delay to avoid API rate limits
            time.sleep(1)
            
        return results


class OpenRouterLLMEvaluator(LLMEvaluator):
    """Class for evaluating sed puzzles using OpenRouter API, which provides access to multiple LLMs."""
    
    def __init__(self, api_key, model, **kwargs):
        """
        Initialize the OpenRouter evaluator.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier to use (e.g. "anthropic/claude-3-sonnet")
            **kwargs: Additional arguments to pass to LLMEvaluator
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def call_api(self, prompt):
        """
        Call the OpenRouter API with the given prompt.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The model's response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://sed-solver.com",  # Required by OpenRouter - using a dummy domain
            "X-Title": "SED Puzzle Solver",  # Optional but good practice
            "User-Agent": "sed-solver-evaluator/1.0"  # OpenRouter sometimes checks User-Agent
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temperature for more deterministic responses
            "max_tokens": 1024,
            "stream": False,
            # Try to add some OpenRouter-specific parameters
            "transforms": ["middle-out"],
            "route": "fallback"
        }
        
        try:
            # Debug log the API request
            logging.info(f"Calling OpenRouter API with model: {self.model}")
            logging.info(f"OpenRouter API URL: {self.api_url}")
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            # Even if status code is not 200, try to get response body for debugging
            try:
                result_text = response.text
                logging.info(f"OpenRouter API response text: {result_text[:200]}...")
                result = response.json()
            except Exception as parse_err:
                logging.error(f"Failed to parse API response: {parse_err}")
                if response.text:
                    logging.error(f"Response text: {response.text[:500]}")
                return ""
                
            # Now raise for status to check HTTP errors
            response.raise_for_status()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logging.error(f"Unexpected API response format: {result}")
                return ""
                
        except Exception as e:
            logging.error(f"API call error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"Response status: {e.response.status_code}")
                logging.error(f"Response text: {e.response.text[:500]}")
            return ""
    
    def evaluate_with_real_llm(self, prompt_type, num_samples=10):
        """
        Evaluate the puzzles using the OpenRouter API.
        
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
            
            # Build the prompt
            prompt = self.build_prompt(problem, prompt_type)
            
            # Call the API
            logging.info(f"[{i+1}/{len(sample_ids)}] Evaluating problem {problem_id} with {prompt_type} prompting...")
            response = self.call_api(prompt)
            
            # Parse the response
            solution = self.parse_llm_response(response)
            
            # Verify the solution
            is_correct = self.verify_solution(problem, solution)
            
            results[problem_id] = {
                "prompt_type": prompt_type,
                "solution": solution,
                "is_correct": is_correct,
                "response": response  # Store the full response for analysis
            }
            
            logging.info(f"Problem {problem_id} - Solution: {solution} - Correct: {is_correct}")
            
            # Add a small delay to avoid API rate limits
            time.sleep(1)
            
        return results
        
    def call_gpt_api(self, prompt):
        """
        Call the OpenAI GPT API with the given prompt.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            str: The model's response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.1  # Low temperature for more deterministic responses
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logging.error(f"Unexpected API response format: {result}")
                return ""
                
        except Exception as e:
            logging.error(f"API call error: {e}")
            return ""
    
    def evaluate_with_real_llm(self, prompt_type, num_samples=10):
        """
        Evaluate the puzzles using the GPT API.
        
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
            
            # Build the prompt
            prompt = self.build_prompt(problem, prompt_type)
            
            # Call the API
            logging.info(f"[{i+1}/{len(sample_ids)}] Evaluating problem {problem_id} with {prompt_type} prompting...")
            response = self.call_gpt_api(prompt)
            
            # Parse the response
            solution = self.parse_llm_response(response)
            
            # Verify the solution
            is_correct = self.verify_solution(problem, solution)
            
            results[problem_id] = {
                "prompt_type": prompt_type,
                "solution": solution,
                "is_correct": is_correct,
                "response": response  # Store the full response for analysis
            }
            
            logging.info(f"Problem {problem_id} - Solution: {solution} - Correct: {is_correct}")
            
            # Add a small delay to avoid API rate limits
            time.sleep(1)
            
        return results


# Function to evaluate puzzles with different LLMs
def evaluate_with_multiple_llms(puzzles_dir, results_dir, api_keys, num_samples=10):
    """
    Evaluate puzzles using multiple LLMs and prompting techniques.
    
    Args:
        puzzles_dir: Directory containing puzzle JSON files
        results_dir: Directory to save results
        api_keys: Dictionary of API keys for different providers
        num_samples: Number of puzzles to evaluate per LLM/prompt combo
    """
    Path(results_dir).mkdir(exist_ok=True)
    
    # Prompt types to evaluate
    prompt_types = ["zero-shot", "few-shot", "cot"]
    
    # Define the LLMs to evaluate
    llm_evaluators = []
    
    # Add GPT evaluator if API key provided
    if "openai" in api_keys:
        llm_evaluators.append(
            (
                "gpt-4o",
                GPTLLMEvaluator(
                    api_key=api_keys["openai"],
                    model="gpt-4o",
                    puzzles_dir=puzzles_dir,
                    output_dir=results_dir
                )
            )
        )
        
    # Add OpenRouter evaluators if API key provided
    if "openrouter" in api_keys:
        # Add OpenAI GPT-4o via OpenRouter
        llm_evaluators.append(
            (
                "openrouter-gpt-4o",
                OpenRouterLLMEvaluator(
                    api_key=api_keys["openrouter"],
                    model="openai/gpt-4o",
                    puzzles_dir=puzzles_dir,
                    output_dir=results_dir
                )
            )
        )
        
        # Add Mixtral-8x7b via OpenRouter (open source model)
        llm_evaluators.append(
            (
                "openrouter-mixtral-8x7b",
                OpenRouterLLMEvaluator(
                    api_key=api_keys["openrouter"],
                    model="mistralai/mixtral-8x7b-instruct",
                    puzzles_dir=puzzles_dir,
                    output_dir=results_dir
                )
            )
        )
        
        # Add Llama-3-70b via OpenRouter (open source model)
        llm_evaluators.append(
            (
                "openrouter-llama-3-70b",
                OpenRouterLLMEvaluator(
                    api_key=api_keys["openrouter"],
                    model="meta-llama/llama-3-70b-instruct",
                    puzzles_dir=puzzles_dir,
                    output_dir=results_dir
                )
            )
        )
    
    if not llm_evaluators:
        logging.error("No API keys provided. Cannot evaluate with real LLMs.")
        return
    
    # Evaluate with each LLM and prompt type
    all_results = {}
    for model_name, evaluator in llm_evaluators:
        all_results[model_name] = {}
        
        for prompt_type in prompt_types:
            logging.info(f"Evaluating {model_name} with {prompt_type} prompting...")
            
            results = evaluator.evaluate_with_real_llm(prompt_type, num_samples=num_samples)
            
            # Save results
            evaluator.save_results(results, f"{model_name}_{prompt_type}_results.json")
            
            # Compute metrics
            metrics = evaluator.compute_metrics(results)
            all_results[model_name][prompt_type] = metrics
            
    # Log overall results
    logging.info("\n===== EVALUATION RESULTS =====")
    for model_name in all_results:
        logging.info(f"\nResults for {model_name}:")
        for prompt_type, metrics in all_results[model_name].items():
            logging.info(f"  {prompt_type}: {metrics['correct']}/{metrics['total']} correct ({metrics['success_rate']*100:.2f}%)")