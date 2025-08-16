#!/usr/bin/env python3
"""
Script to plot success rates by domain, model, and iteration.
Creates comparative plots showing success rate progression across iterations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data(json_file):
    """Load plotting data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def clean_data(data):
    """Clean and process the data for plotting."""
    cleaned_data = {}
    
    for domain, models in data.items():
        cleaned_data[domain] = {}
        
        for model, model_data in models.items():
            iterations = model_data['iterations']
            success_rates = model_data['success_rates']
            
            # Create a mapping of iteration to best success rate at that iteration
            iteration_to_rate = {}
            for i, rate in zip(iterations, success_rates):
                if i not in iteration_to_rate:
                    iteration_to_rate[i] = rate
                else:
                    # Keep the best rate for this iteration
                    iteration_to_rate[i] = max(iteration_to_rate[i], rate)
            
            # Sort by iteration number
            sorted_iterations = sorted(iteration_to_rate.keys())
            sorted_rates = [iteration_to_rate[i] for i in sorted_iterations]
            
            cleaned_data[domain][model] = {
                'iterations': sorted_iterations,
                'success_rates': sorted_rates
            }
    
    return cleaned_data

def plot_by_domain(data, output_dir='.'):
    """Create separate plots for each domain."""
    
    # Color mapping for models
    model_colors = {
        'claude': '#FF6B6B',
        'gemini': '#4ECDC4', 
        'gpt': '#45B7D1'
    }
    
    # Create subplots
    n_domains = len(data)
    n_cols = 3
    n_rows = (n_domains + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    
    # Flatten axes for easy iteration
    axes_flat = [ax for row in axes for ax in row]
    
    for i, (domain, models) in enumerate(data.items()):
        ax = axes_flat[i]
        
        for model, model_data in models.items():
            iterations = model_data['iterations']
            success_rates = model_data['success_rates']
            
            if iterations and success_rates:
                color = model_colors.get(model, '#888888')
                ax.plot(iterations, success_rates, 
                       marker='o', linewidth=2, markersize=6,
                       label=model.upper(), color=color, alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Success Rate')
        ax.set_title(f'{domain.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Set x-axis to show integer ticks
        if any(models.values()):
            max_iter = max(max(m['iterations']) for m in models.values() if m['iterations'])
            ax.set_xticks(range(1, max_iter + 1))
    
    # Hide unused subplots
    for i in range(len(data), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/success_rates_by_domain.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(data, output_dir='.'):
    """Create a comparison plot showing all models across all domains."""
    
    model_colors = {
        'claude': '#FF6B6B',
        'gemini': '#4ECDC4', 
        'gpt': '#45B7D1'
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect all data points for each model
    model_data = defaultdict(lambda: {'iterations': [], 'success_rates': []})
    
    for domain, models in data.items():
        for model, model_info in models.items():
            model_data[model]['iterations'].extend(model_info['iterations'])
            model_data[model]['success_rates'].extend(model_info['success_rates'])
    
    # Plot each model
    for model, info in model_data.items():
        if info['iterations']:
            # Calculate average success rate by iteration
            iter_rates = defaultdict(list)
            for i, rate in zip(info['iterations'], info['success_rates']):
                iter_rates[i].append(rate)
            
            avg_iterations = sorted(iter_rates.keys())
            avg_rates = [np.mean(iter_rates[i]) for i in avg_iterations]
            std_rates = [np.std(iter_rates[i]) for i in avg_iterations]
            
            color = model_colors.get(model, '#888888')
            ax.plot(avg_iterations, avg_rates, 
                   marker='o', linewidth=3, markersize=8,
                   label=f'{model.upper()} (avg)', color=color, alpha=0.8)
            
            # Add error bars
            ax.fill_between(avg_iterations, 
                           np.array(avg_rates) - np.array(std_rates),
                           np.array(avg_rates) + np.array(std_rates),
                           color=color, alpha=0.2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Model Performance Comparison (Average Across All Domains)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(data):
    """Create a summary table of final success rates."""
    
    print("\n" + "="*60)
    print("FINAL SUCCESS RATES SUMMARY")
    print("="*60)
    
    # Collect final success rates
    summary = defaultdict(dict)
    
    for domain, models in data.items():
        for model, model_info in models.items():
            if model_info['success_rates']:
                final_rate = model_info['success_rates'][-1]
                max_iter = model_info['iterations'][-1]
                summary[domain][model] = {
                    'final_rate': final_rate,
                    'max_iteration': max_iter,
                    'num_points': len(model_info['success_rates'])
                }
    
    # Print table
    print(f"{'Domain':<25} {'Model':<8} {'Final Rate':<12} {'Max Iter':<10} {'Data Points':<12}")
    print("-" * 60)
    
    for domain, models in summary.items():
        for i, (model, info) in enumerate(models.items()):
            domain_name = domain.replace('_', ' ').title() if i == 0 else ''
            print(f"{domain_name:<25} {model.upper():<8} {info['final_rate']:<12.3f} "
                  f"{info['max_iteration']:<10} {info['num_points']:<12}")
    
    print("="*60)

def main():
    """Main plotting function."""
    
    # Load data
    data_file = '/home/omer/projects/AML_VIA_LLM/plotting_data.json'
    output_dir = '/home/omer/projects/AML_VIA_LLM'
    
    try:
        data = load_data(data_file)
        cleaned_data = clean_data(data)
        
        print("Creating plots...")
        
        # Create individual domain plots
        plot_by_domain(cleaned_data, output_dir)
        
        # Create model comparison plot
        plot_model_comparison(cleaned_data, output_dir)
        
        # Print summary table
        create_summary_table(cleaned_data)
        
        print(f"\nPlots saved to: {output_dir}/")
        print("- success_rates_by_domain.png")
        print("- model_comparison.png")
        
    except FileNotFoundError:
        print(f"Error: Could not find data file {data_file}")
        print("Please run collect_plotting_data.py first to generate the data.")
    except Exception as e:
        print(f"Error creating plots: {e}")

if __name__ == "__main__":
    main()