import json
import os
from pathlib import Path

# Try to import visualization libraries, but continue with text-based report if not available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib and/or numpy not installed. Skipping visualization.")
    VISUALIZATION_AVAILABLE = False

def load_results(results_dir):
    """
    Load evaluation results from json files.
    
    Args:
        results_dir: Directory containing result json files
        
    Returns:
        Dictionary mapping model/prompt to metrics
    """
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            parts = filename.replace("_results.json", "").split("_")
            if len(parts) < 2:
                continue
                
            model = parts[0]
            prompt_type = "_".join(parts[1:])
            
            try:
                with open(os.path.join(results_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Check if this is a direct results file with individual problems
                    if isinstance(data, dict) and not 'metrics' in data:
                        # Calculate metrics from individual results
                        total = len(data)
                        correct = sum(1 for problem_id, result in data.items() if result.get('is_correct', False))
                        success_rate = correct / total if total > 0 else 0
                        
                        if model not in results:
                            results[model] = {}
                        
                        results[model][prompt_type] = {
                            'success_rate': success_rate,
                            'total': total,
                            'correct': correct
                        }
                        
                        print(f"Calculated metrics for {model} {prompt_type}: {correct}/{total} = {success_rate:.2f}")
                    # Check if metrics are directly included in the data
                    elif 'metrics' in data:
                        success_rate = data['metrics'].get('success_rate', 0)
                        total = data['metrics'].get('total', 0)
                        correct = data['metrics'].get('correct', 0)
                        
                        if model not in results:
                            results[model] = {}
                        
                        results[model][prompt_type] = {
                            'success_rate': success_rate,
                            'total': total,
                            'correct': correct
                        }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return results

def plot_results(results):
    """
    Plot the results as a bar chart.
    
    Args:
        results: Results dictionary from load_results
    """
    models = list(results.keys())
    prompt_types = ['zero-shot', 'few-shot', 'cot']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.25
    opacity = 0.8
    index = np.arange(len(models))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, prompt_type in enumerate(prompt_types):
        success_rates = []
        for model in models:
            if prompt_type in results[model]:
                success_rates.append(results[model][prompt_type]['success_rate'] * 100)
            else:
                success_rates.append(0)
        
        plt.bar(index + i * bar_width, 
                success_rates,
                bar_width,
                alpha=opacity,
                color=colors[i],
                label=f'{prompt_type}')
    
    plt.xlabel('Models')
    plt.ylabel('Success Rate (%)')
    plt.title('SED Puzzle Success Rate by Model and Prompting Strategy')
    plt.xticks(index + bar_width, models)
    plt.legend()
    plt.tight_layout()
    
    # Add value labels on top of each bar
    for i, prompt_type in enumerate(prompt_types):
        for j, model in enumerate(models):
            if prompt_type in results[model]:
                success_rate = results[model][prompt_type]['success_rate'] * 100
                correct = results[model][prompt_type]['correct']
                total = results[model][prompt_type]['total']
                ax.text(j + i * bar_width, success_rate + 1, 
                      f'{success_rate:.1f}%\n({correct}/{total})', 
                      ha='center', va='bottom')
    
    # Save the figure
    output_dir = Path('evaluation/figures')
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / 'success_rate_comparison.png', dpi=300)
    plt.savefig(output_dir / 'success_rate_comparison.pdf')
    
    print(f"Saved plot to {output_dir}/success_rate_comparison.png and .pdf")
    
    # Close the figure to free memory
    plt.close()

def generate_markdown_report(results):
    """
    Generate a markdown report from the results.
    
    Args:
        results: Results dictionary from load_results
    """
    markdown = "# SED Puzzle Evaluation Results\n\n"
    markdown += "## Success Rates\n\n"
    
    markdown += "| Model | Zero-shot | Few-shot | Chain of Thought |\n"
    markdown += "|-------|-----------|----------|------------------|\n"
    
    for model in results:
        zero_shot = results[model].get('zero-shot', {})
        few_shot = results[model].get('few-shot', {})
        cot = results[model].get('cot', {})
        
        zero_shot_str = f"{zero_shot.get('correct', 0)}/{zero_shot.get('total', 0)} ({zero_shot.get('success_rate', 0)*100:.1f}%)"
        few_shot_str = f"{few_shot.get('correct', 0)}/{few_shot.get('total', 0)} ({few_shot.get('success_rate', 0)*100:.1f}%)"
        cot_str = f"{cot.get('correct', 0)}/{cot.get('total', 0)} ({cot.get('success_rate', 0)*100:.1f}%)"
        
        markdown += f"| {model} | {zero_shot_str} | {few_shot_str} | {cot_str} |\n"
    
    markdown += "\n## Analysis\n\n"
    
    # Add some analysis based on the results
    best_model = None
    best_rate = -1
    best_prompt = None
    
    for model in results:
        for prompt_type in results[model]:
            rate = results[model][prompt_type].get('success_rate', 0)
            if rate > best_rate:
                best_rate = rate
                best_model = model
                best_prompt = prompt_type
    
    markdown += f"### Key Findings\n\n"
    
    if best_model:
        markdown += f"- The best performing configuration was **{best_model}** with **{best_prompt}** prompting, achieving a {best_rate*100:.1f}% success rate.\n"
    
    # Add comparative analysis for prompt types
    markdown += "- **Prompt Type Analysis**:\n"
    
    for model in results:
        if len(results[model]) > 1:
            prompt_rates = [(p, results[model][p]['success_rate']) for p in results[model]]
            prompt_rates.sort(key=lambda x: x[1], reverse=True)
            best_prompt, best_prompt_rate = prompt_rates[0]
            
            markdown += f"  - For {model}, {best_prompt} prompting was most effective ({best_prompt_rate*100:.1f}%).\n"
    
    # Save the markdown file
    output_dir = Path('evaluation/reports')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'evaluation_report.md'
    
    with open(output_path, 'w') as f:
        f.write(markdown)
    
    print(f"Saved markdown report to {output_path}")

def main():
    results_dir = os.path.join("evaluation", "results")
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    results = load_results(results_dir)
    if not results:
        print("No results found.")
        return
    
    print("Loaded results:", results)
    
    # Generate visualizations if libraries are available
    if VISUALIZATION_AVAILABLE:
        plot_results(results)
    else:
        print("Skipping visualization due to missing libraries.")
    
    # Generate markdown report
    generate_markdown_report(results)
    
if __name__ == "__main__":
    main()