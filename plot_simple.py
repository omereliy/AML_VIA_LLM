#!/usr/bin/env python3
"""
Simple plotting script that creates ASCII-style plots and generates CSV data for external plotting.
"""

import json
import csv
from collections import defaultdict

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

def create_ascii_plot(iterations, success_rates, title, width=60, height=20):
    """Create a simple ASCII plot."""
    if not iterations or not success_rates:
        return f"{title}\nNo data available\n"
    
    min_iter = min(iterations)
    max_iter = max(iterations)
    min_rate = 0.0
    max_rate = 1.0
    
    # Create plot grid
    plot = []
    for _ in range(height):
        plot.append([' '] * width)
    
    # Plot points
    for i, rate in zip(iterations, success_rates):
        x = int((i - min_iter) / (max_iter - min_iter) * (width - 1)) if max_iter > min_iter else 0
        y = int((1 - rate) * (height - 1))  # Invert y-axis
        if 0 <= x < width and 0 <= y < height:
            plot[y][x] = '*'
    
    # Convert to string
    result = f"{title}\n"
    result += "1.0 |" + "".join(plot[0]) + "\n"
    for row in plot[1:-1]:
        result += "    |" + "".join(row) + "\n"
    result += "0.0 |" + "".join(plot[-1]) + "\n"
    result += "    +" + "-" * width + "\n"
    result += f"    {min_iter:<{width//2}}{max_iter:>{width//2}}\n"
    result += "    Iterations\n\n"
    
    return result

def export_to_csv(data, output_file):
    """Export data to CSV format for external plotting tools."""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Domain', 'Model', 'Iteration', 'Success_Rate'])
        
        for domain, models in data.items():
            for model, model_data in models.items():
                iterations = model_data['iterations']
                success_rates = model_data['success_rates']
                
                for iteration, rate in zip(iterations, success_rates):
                    writer.writerow([domain, model, iteration, rate])

def create_summary_table(data):
    """Create a summary table of final success rates."""
    
    print("\n" + "="*80)
    print("FINAL SUCCESS RATES SUMMARY")
    print("="*80)
    
    # Collect final success rates
    summary = defaultdict(dict)
    
    for domain, models in data.items():
        for model, model_info in models.items():
            if model_info['success_rates']:
                final_rate = model_info['success_rates'][-1]
                max_iter = model_info['iterations'][-1]
                avg_rate = sum(model_info['success_rates']) / len(model_info['success_rates'])
                summary[domain][model] = {
                    'final_rate': final_rate,
                    'avg_rate': avg_rate,
                    'max_iteration': max_iter,
                    'num_points': len(model_info['success_rates'])
                }
    
    # Print table
    print(f"{'Domain':<25} {'Model':<8} {'Final Rate':<12} {'Avg Rate':<12} {'Max Iter':<10} {'Points':<8}")
    print("-" * 80)
    
    for domain, models in summary.items():
        for i, (model, info) in enumerate(models.items()):
            domain_name = domain.replace('_', ' ').title() if i == 0 else ''
            print(f"{domain_name:<25} {model.upper():<8} {info['final_rate']:<12.3f} "
                  f"{info['avg_rate']:<12.3f} {info['max_iteration']:<10} {info['num_points']:<8}")
    
    print("="*80)

def create_ascii_plots(data):
    """Create ASCII plots for each domain."""
    
    print("\n" + "="*80)
    print("ASCII PLOTS BY DOMAIN")
    print("="*80)
    
    for domain, models in data.items():
        print(f"\n{domain.replace('_', ' ').title().upper()}")
        print("-" * 60)
        
        for model, model_data in models.items():
            title = f"{model.upper()} - Success Rate by Iteration"
            plot = create_ascii_plot(
                model_data['iterations'], 
                model_data['success_rates'], 
                title
            )
            print(plot)

def main():
    """Main function."""
    
    # Load data
    data_file = '/home/omer/projects/AML_VIA_LLM/plotting_data.json'
    csv_file = '/home/omer/projects/AML_VIA_LLM/success_rates_data.csv'
    
    try:
        data = load_data(data_file)
        cleaned_data = clean_data(data)
        
        print("Processing experiment data...")
        
        # Create summary table
        create_summary_table(cleaned_data)
        
        # Create ASCII plots
        create_ascii_plots(cleaned_data)
        
        # Export to CSV
        export_to_csv(cleaned_data, csv_file)
        
        print(f"\nData exported to: {csv_file}")
        print("You can import this CSV into Excel, Google Sheets, or any plotting tool.")
        print("\nColumns: Domain, Model, Iteration, Success_Rate")
        print("Recommended plots:")
        print("- Line chart: X=Iteration, Y=Success_Rate, Series=Model, Filter by Domain")
        print("- Scatter plot: X=Iteration, Y=Success_Rate, Color=Model, Facet by Domain")
        
    except FileNotFoundError:
        print(f"Error: Could not find data file {data_file}")
        print("Please run collect_plotting_data.py first to generate the data.")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()