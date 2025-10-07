"""
Enhanced module for evaluating sed puzzles using real LLM APIs with retry mechanisms.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
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


class EnhancedOpenRouterLLMEvaluator(EnhancedLLMEvaluator):
    """Enhanced class for evaluating sed puzzles using OpenRouter API with retries."""
    
    def __init__(self, api_key, model, temperature=0.2, **kwargs):
        """
        Initialize the OpenRouter evaluator with enhanced features.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier to use (e.g. "anthropic/claude-3-sonnet")
            temperature: Temperature for generation (lower = more deterministic)
            **kwargs: Additional arguments to pass to EnhancedLLMEvaluator
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def call_api_with_backoff(self, prompt, max_retries=3, initial_backoff=2):
        """
        Call the OpenRouter API with exponential backoff for rate limits.
        
        Args:
            prompt: The prompt to send to the API
            max_retries: Maximum number of retries for rate limit errors
            initial_backoff: Initial backoff time in seconds
            
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
            "temperature": self.temperature,  # Lower temperature for more deterministic responses
            "max_tokens": 1024,
            "stream": False,
            # Try to add some OpenRouter-specific parameters
            "transforms": ["middle-out"],
            "route": "fallback"
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
                    if hasattr(e, 'response') and e.response is not None:
                        logging.error(f"Response status: {e.response.status_code}")
                        logging.error(f"Response text: {e.response.text[:500]}")
                    return f"Error: API request failed - {str(e)}"
        
        return "Error: Maximum retries exceeded"
    
    def evaluate_with_real_llm(self, prompt_type, num_samples=10):
        """
        Evaluate the puzzles using the OpenRouter API with enhanced retry logic.
        
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
                llm_client=self.call_api_with_backoff,
                prompt_type=prompt_type,
                problem=problem,
                max_retries=1  # One additional attempt if first solution is incorrect
            )
            
            results[problem_id] = result
            
            logging.info(f"Problem {problem_id} - Solution: {result['solution']} - Correct: {result['is_correct']} - Attempts: {result['attempts']}")
            
            # Add a small delay to avoid API rate limits
            time.sleep(1)
            
        return results


# Function to evaluate puzzles with different LLMs using enhanced features
def evaluate_with_enhanced_llms(puzzles_dir, results_dir, api_keys, num_samples=10, temperature=0.2):
    """
    Evaluate puzzles using multiple LLMs and prompting techniques with enhanced accuracy features.
    
    Args:
        puzzles_dir: Directory containing puzzle JSON files
        results_dir: Directory to save results
        api_keys: Dictionary of API keys for different providers
        num_samples: Number of puzzles to evaluate per LLM/prompt combo
        temperature: Temperature setting for generation (lower = more deterministic)
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
                EnhancedGPTLLMEvaluator(
                    api_key=api_keys["openai"],
                    model="gpt-4o",
                    temperature=temperature,
                    puzzles_dir=puzzles_dir,
                    output_dir=results_dir
                )
            )
        )
        
    # Add OpenRouter evaluators if API key provided
    if "openrouter" in api_keys:
        # Add Claude-3-Sonnet via OpenRouter
        llm_evaluators.append(
            (
                "openrouter-claude-3-sonnet",
                EnhancedOpenRouterLLMEvaluator(
                    api_key=api_keys["openrouter"],
                    model="anthropic/claude-3-sonnet",
                    temperature=temperature,
                    puzzles_dir=puzzles_dir,
                    output_dir=results_dir
                )
            )
        )
        
        # Add Gemini-1.5-Pro via OpenRouter
        llm_evaluators.append(
            (
                "openrouter-gemini-1.5-pro",
                EnhancedOpenRouterLLMEvaluator(
                    api_key=api_keys["openrouter"],
                    model="google/gemini-1.5-pro",
                    temperature=temperature,
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
            evaluator.save_results(results, f"{model_name}_{prompt_type}_enhanced_results.json")
            
            # Compute metrics
            metrics = evaluator.compute_basic_metrics(results)
            all_results[model_name][prompt_type] = metrics
            
    # Log overall results
    logging.info("\n===== ENHANCED EVALUATION RESULTS =====")
    for model_name in all_results:
        logging.info(f"\nResults for {model_name}:")
        for prompt_type, metrics in all_results[model_name].items():
            if "first_attempt_rate" in metrics:
                logging.info(f"  {prompt_type}: {metrics['correct']}/{metrics['total']} correct ({metrics['success_rate']*100:.1f}%) - "
                           f"First attempt: {metrics['first_attempt_rate']*100:.1f}%, "
                           f"Retry success: {metrics['retry_success_rate']*100:.1f}%")
            else:
                logging.info(f"  {prompt_type}: {metrics['correct']}/{metrics['total']} correct ({metrics['success_rate']*100:.1f}%)")
    
    return all_results