"""
Advanced evaluation metrics for analyzing SED puzzle performance.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        
    def visualize_success_rates(self, basic_metrics, output_file):
        """Create a bar chart comparing success rates across different prompt types."""
        prompt_types = list(basic_metrics.keys())
        success_rates = [basic_metrics[pt]["success_rate"] * 100 for pt in prompt_types]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(prompt_types, success_rates)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Success Rates by Prompt Type')
        plt.xlabel('Prompt Type')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(output_file)
        plt.close()
        
    def visualize_retry_effectiveness(self, basic_metrics, output_file):
        """Create a bar chart showing the effectiveness of retry mechanism."""
        prompt_types = []
        first_attempt_rates = []
        final_success_rates = []
        
        for prompt_type, metrics in basic_metrics.items():
            if "first_attempt_rate" in metrics:
                prompt_types.append(prompt_type)
                first_attempt_rates.append(metrics["first_attempt_rate"] * 100)
                final_success_rates.append(metrics["success_rate"] * 100)
        
        if not prompt_types:
            logging.info("No retry data available to visualize")
            return
            
        x = range(len(prompt_types))
        width = 0.35
        
        plt.figure(figsize=(12, 7))
        plt.bar(x, first_attempt_rates, width, label='First Attempt Success')
        plt.bar([i + width for i in x], final_success_rates, width, label='Final Success (With Retries)')
        
        plt.xlabel('Prompt Type')
        plt.ylabel('Success Rate (%)')
        plt.title('Effectiveness of Retry Mechanism')
        plt.xticks([i + width/2 for i in x], prompt_types)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
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
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
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
        
        # Create overall success rate visualization
        self.visualize_success_rates(
            basic_metrics,
            f"{output_dir}/success_rates.png"
        )
        
        # Create retry effectiveness visualization if data available
        if "first_attempt_rate" in next(iter(basic_metrics.values()), {}):
            self.visualize_retry_effectiveness(
                basic_metrics,
                f"{output_dir}/retry_effectiveness.png"
            )
        
        # Generate markdown report
        with open(f"{output_dir}/enhanced_evaluation_report.md", 'w') as f:
            f.write("# Enhanced SED Puzzle Solver Evaluation Report\n\n")
            
            # Overall success rate visualization
            f.write("## Overall Success Rates\n\n")
            f.write(f"![Success Rates](success_rates.png)\n\n")
            
            # Basic metrics section
            f.write("## Detailed Success Metrics\n\n")
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
            
            # Retry effectiveness section if available
            if "first_attempt_rate" in next(iter(basic_metrics.values()), {}):
                f.write("\n## Retry Effectiveness\n\n")
                f.write("The following chart shows the improvement in success rate achieved by the retry mechanism:\n\n")
                f.write(f"![Retry Effectiveness](retry_effectiveness.png)\n\n")
            
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
            
            f.write("### Key Focus Areas for Improvement\n\n")
            
            if most_common_failure[0] == "state_tracking_error":
                f.write("1. **Improve String State Tracking**\n")
                f.write("   - Enhance prompts to emphasize accurate string state after each transformation\n")
                f.write("   - Add explicit verification steps in the Chain of Thought process\n")
                f.write("   - Implement visual tracking of string transformations in prompts\n\n")
            elif most_common_failure[0] == "parse_failure":
                f.write("1. **Enhance Output Format Instructions**\n")
                f.write("   - Make output format requirements more prominent in prompts\n")
                f.write("   - Add explicit examples of correctly formatted solutions\n")
                f.write("   - Implement more robust parsing of model outputs\n\n")
            elif most_common_failure[0] == "invalid_rule_index":
                f.write("1. **Improve Rule Index Understanding**\n")
                f.write("   - Emphasize valid rule index range in prompts\n")
                f.write("   - Add explicit warnings about out-of-range indices\n")
                f.write("   - Include verification steps to check rule index validity\n\n")
            elif most_common_failure[0] == "incomplete_solution":
                f.write("1. **Address Incomplete Solutions**\n")
                f.write("   - Emphasize that solutions must reach an empty string\n")
                f.write("   - Add verification steps to confirm the final string is empty\n")
                f.write("   - Provide examples of complete vs. incomplete solutions\n\n")
            
            # Add second most common failure if it's significant
            if len(all_failures) > 1:
                failures_sorted = sorted(all_failures.items(), key=lambda x: x[1], reverse=True)
                second_failure = failures_sorted[1]
                if second_failure[1] > 0:
                    if second_failure[0] == "state_tracking_error":
                        f.write("2. **Improve String State Tracking**\n")
                        f.write("   - Enhance prompts to emphasize accurate string state after each transformation\n")
                        f.write("   - Add explicit verification steps in the Chain of Thought process\n\n")
                    elif second_failure[0] == "parse_failure":
                        f.write("2. **Enhance Output Format Instructions**\n")
                        f.write("   - Make output format requirements more prominent in prompts\n")
                        f.write("   - Implement more robust parsing of model outputs\n\n")
            
            # Best performing prompt type
            best_prompt = max(basic_metrics.items(), key=lambda x: x[1]["success_rate"])
            f.write(f"The **{best_prompt[0]}** prompting strategy shows the best performance with a "
                   f"{best_prompt[1]['success_rate']*100:.1f}% success rate and should be prioritized.\n\n")
            
            # Effectiveness of retry mechanism
            if "first_attempt_rate" in next(iter(basic_metrics.values()), {}):
                # Calculate average improvement from retry
                retry_improvements = []
                for metrics in basic_metrics.values():
                    if "first_attempt_rate" in metrics and "success_rate" in metrics:
                        improvement = metrics["success_rate"] - metrics["first_attempt_rate"]
                        retry_improvements.append(improvement)
                
                avg_improvement = sum(retry_improvements) / len(retry_improvements) if retry_improvements else 0
                f.write(f"The retry mechanism improved success rates by an average of {avg_improvement*100:.1f}%. ")
                
                if avg_improvement > 0.1:  # More than 10% improvement
                    f.write("This substantial improvement suggests that further refining the retry feedback could yield even better results.\n\n")
                elif avg_improvement > 0:
                    f.write("While the improvement is modest, refinements to the retry feedback could yield better results.\n\n")
                else:
                    f.write("The retry mechanism did not significantly improve results, suggesting deeper issues that require a different approach.\n\n")
        
        # Save JSON data
        enhanced_report = {
            "basic_metrics": basic_metrics,
            "failure_analyses": failure_analyses
        }
        
        with open(f"{output_dir}/enhanced_metrics_report.json", 'w') as f:
            json.dump(enhanced_report, f, indent=2)
        
        return enhanced_report