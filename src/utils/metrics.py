import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

def plot_training_curves(metrics: Dict[str, List[float]], save_path: str):
    """Plot training and validation curves."""
    plt.figure(figsize=(12, 6))
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()

def save_benchmark_results(results: Dict[str, Dict[str, float]], save_path: str):
    """Save benchmark results to JSON file."""
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

def create_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """Create a markdown table comparing model performances."""
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    
    for model_name, metrics in results.items():
        row = [
            model_name.split('/')[-1],
            f"{metrics['test_acc']:.4f}",
            f"{metrics['test_precision']:.4f}",
            f"{metrics['test_recall']:.4f}",
            f"{metrics['test_f1']:.4f}"
        ]
        table.append("| " + " | ".join(row) + " |")
    
    return "\n".join(table)

def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: str):
    """Create a bar plot comparing model performances."""
    metrics = ['test_acc', 'test_precision', 'test_recall', 'test_f1']
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels([model.split('/')[-1] for model in models], rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_results(results_dir: str):
    """Analyze benchmark results and generate visualizations."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_path / 'results.json', 'r') as f:
        results = json.load(f)
    
    # Generate comparison table
    table = create_comparison_table(results)
    with open(results_path / 'comparison.md', 'w') as f:
        f.write(table)
    
    # Create visualization
    plot_model_comparison(results, str(results_path / 'comparison.png'))
