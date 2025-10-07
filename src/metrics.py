"""
Evaluation metrics for comparing LLM performance on sed puzzles.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class EvaluationMetrics:
    """
    A class for computing and visualizing evaluation metrics for sed puzzle solutions.
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
        """
        Compute basic success rate metrics for a set of results.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            dict: Computed metrics
        """
        total = len(results)
        correct = sum(1 for result in results.values() if result["is_correct"])
        success_rate = correct / total if total > 0 else 0
        
        return {
            "total": total,
            "correct": correct,
            "success_rate": success_rate
        }
    
    def compute_difficulty_based_metrics(self, results, puzzles):
        """
        Compute metrics based on puzzle difficulty.
        
        Args:
            results: Dictionary of evaluation results
            puzzles: Dictionary of puzzles with difficulty information
            
        Returns:
            dict: Metrics grouped by difficulty level
        """
        difficulty_metrics = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for problem_id, result in results.items():
            if problem_id in puzzles:
                # For our dataset, we'll determine difficulty based on:
                # 1. Length of the initial string
                # 2. Number of transitions
                # 3. Complexity of transitions (e.g., transitions with empty source/target)
                puzzle = puzzles[problem_id]
                
                # Simplified difficulty calculation for demonstration
                initial_length = len(puzzle["initial_string"])
                num_transitions = len(puzzle["transitions"])
                
                # Determine difficulty level (1-5)
                if initial_length <= 5 and num_transitions <= 3:
                    difficulty = 1  # Very easy
                elif initial_length <= 10 and num_transitions <= 5:
                    difficulty = 2  # Easy
                elif initial_length <= 15 and num_transitions <= 7:
                    difficulty = 3  # Medium
                elif initial_length <= 20 and num_transitions <= 10:
                    difficulty = 4  # Hard
                else:
                    difficulty = 5  # Very hard
                
                # Update metrics
                difficulty_metrics[difficulty]["total"] += 1
                if result["is_correct"]:
                    difficulty_metrics[difficulty]["correct"] += 1
        
        # Calculate success rates
        for difficulty, metrics in difficulty_metrics.items():
            metrics["success_rate"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        
        return difficulty_metrics
    
    def compute_solution_length_metrics(self, results):
        """
        Analyze solution length distribution.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            dict: Metrics related to solution length
        """
        solution_lengths = []
        correct_solution_lengths = []
        incorrect_solution_lengths = []
        
        for result in results.values():
            solution = result["solution"]
            if solution is not None:
                length = len(solution)
                solution_lengths.append(length)
                
                if result["is_correct"]:
                    correct_solution_lengths.append(length)
                else:
                    incorrect_solution_lengths.append(length)
        
        # Calculate statistics
        return {
            "avg_length": np.mean(solution_lengths) if solution_lengths else 0,
            "avg_correct_length": np.mean(correct_solution_lengths) if correct_solution_lengths else 0,
            "avg_incorrect_length": np.mean(incorrect_solution_lengths) if incorrect_solution_lengths else 0,
            "min_length": min(solution_lengths) if solution_lengths else 0,
            "max_length": max(solution_lengths) if solution_lengths else 0
        }
    
    def compute_advanced_metrics(self, correct_results, optimal_results):
        """
        Compare LLM solutions with optimal (shortest) solutions.
        
        Args:
            correct_results: Dictionary of correct LLM solutions
            optimal_results: Dictionary of optimal (shortest) solutions
            
        Returns:
            dict: Advanced metrics comparing solution quality
        """
        optimality_ratios = []
        
        for problem_id, result in correct_results.items():
            if problem_id in optimal_results and result["is_correct"]:
                llm_solution = result["solution"]
                optimal_solution = optimal_results[problem_id]["solution"]
                
                if optimal_solution and llm_solution:
                    # Calculate optimality ratio (optimal length / LLM length)
                    # Lower is better, 1.0 means the LLM found the optimal solution
                    optimality_ratio = len(optimal_solution) / len(llm_solution)
                    optimality_ratios.append(optimality_ratio)
        
        return {
            "avg_optimality_ratio": np.mean(optimality_ratios) if optimality_ratios else 0,
            "median_optimality_ratio": np.median(optimality_ratios) if optimality_ratios else 0,
            "min_optimality_ratio": min(optimality_ratios) if optimality_ratios else 0,
            "max_optimality_ratio": max(optimality_ratios) if optimality_ratios else 0
        }
    
    def visualize_success_rates(self, metrics, output_file):
        """
        Create a bar chart of success rates across prompt types.
        
        Args:
            metrics: Dictionary mapping prompt types to metrics
            output_file: Path to save the visualization
        """
        prompt_types = list(metrics.keys())
        success_rates = [metrics[pt]["success_rate"] * 100 for pt in prompt_types]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(prompt_types, success_rates)
        
        plt.title('Success Rates by Prompt Type')
        plt.xlabel('Prompt Type')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 100)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.savefig(output_file)
        plt.close()
    
    def visualize_difficulty_metrics(self, difficulty_metrics, output_file):
        """
        Create a line chart showing success rates across difficulty levels.
        
        Args:
            difficulty_metrics: Dictionary mapping prompt types to difficulty metrics
            output_file: Path to save the visualization
        """
        plt.figure(figsize=(12, 7))
        
        for prompt_type, metrics in difficulty_metrics.items():
            difficulties = sorted(metrics.keys())
            success_rates = [metrics[diff]["success_rate"] * 100 for diff in difficulties]
            
            plt.plot(difficulties, success_rates, marker='o', label=prompt_type)
        
        plt.title('Success Rates by Difficulty Level')
        plt.xlabel('Difficulty Level (1=Easiest, 5=Hardest)')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 100)
        plt.xticks(range(1, 6))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(output_file)
        plt.close()
    
    def calculate_human_machine_gap(self, human_results, machine_results):
        """
        Calculate the performance gap between human and machine.
        
        Args:
            human_results: Dictionary of human solution results
            machine_results: Dictionary of machine solution results
            
        Returns:
            dict: Metrics showing human vs machine performance
        """
        # For puzzles solved by both human and machine
        common_problems = set(human_results.keys()) & set(machine_results.keys())
        
        human_only = set(human_results.keys()) - set(machine_results.keys())
        machine_only = set(machine_results.keys()) - set(human_results.keys())
        
        human_correct = sum(1 for pid in common_problems if human_results[pid]["is_correct"])
        machine_correct = sum(1 for pid in common_problems if machine_results[pid]["is_correct"])
        
        return {
            "common_problems": len(common_problems),
            "human_only_solved": len(human_only),
            "machine_only_solved": len(machine_only),
            "human_correct_rate": human_correct / len(common_problems) if common_problems else 0,
            "machine_correct_rate": machine_correct / len(common_problems) if common_problems else 0,
            "performance_gap": (human_correct - machine_correct) / len(common_problems) if common_problems else 0
        }
    
    def generate_report(self, results_files, puzzles_file=None, output_dir="evaluation/reports"):
        """
        Generate a comprehensive report of evaluation metrics.
        
        Args:
            results_files: Dictionary mapping prompt types to result files
            puzzles_file: Optional file containing puzzle data (for difficulty analysis)
            output_dir: Directory to save the report and visualizations
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load results for each prompt type
        all_results = {}
        basic_metrics = {}
        
        for prompt_type, file_name in results_files.items():
            all_results[prompt_type] = self.load_results(file_name)
            basic_metrics[prompt_type] = self.compute_basic_metrics(all_results[prompt_type])
        
        # Generate success rate visualization
        self.visualize_success_rates(basic_metrics, f"{output_dir}/success_rates.png")
        
        # Calculate solution length metrics
        length_metrics = {}
        for prompt_type, results in all_results.items():
            length_metrics[prompt_type] = self.compute_solution_length_metrics(results)
        
        # Load puzzles for difficulty analysis if provided
        if puzzles_file:
            with open(puzzles_file, 'r') as f:
                puzzles = json.load(f)
                
            difficulty_metrics = {}
            for prompt_type, results in all_results.items():
                difficulty_metrics[prompt_type] = self.compute_difficulty_based_metrics(results, puzzles)
                
            self.visualize_difficulty_metrics(difficulty_metrics, f"{output_dir}/difficulty_metrics.png")
        
        # Generate report
        report = {
            "basic_metrics": basic_metrics,
            "length_metrics": length_metrics
        }
        
        if puzzles_file:
            report["difficulty_metrics"] = difficulty_metrics
        
        with open(f"{output_dir}/metrics_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nEvaluation Report Summary:")
        print("-------------------------")
        for prompt_type, metrics in basic_metrics.items():
            print(f"{prompt_type}: {metrics['correct']}/{metrics['total']} correct ({metrics['success_rate']*100:.2f}%)")
        
        return report

if __name__ == "__main__":
    # Example usage
    metrics = EvaluationMetrics(results_dir="evaluation/results")
    
    results_files = {
        "zero-shot": "mock_zero-shot_results.json",
        "few-shot": "mock_few-shot_results.json",
        "cot": "mock_cot_results.json"
    }
    
    # Generate report
    report = metrics.generate_report(results_files)