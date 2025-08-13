#!/usr/bin/env python3
"""
Script to collect data for plotting success rates by domain, model, and iteration.
This script extracts success rate data from experiment logs for visualization.
"""

import os
import re
import json
from collections import defaultdict
from pathlib import Path


def parse_success_rates_from_log(log_path):
    """Extract success rates and iterations from conversation log."""
    success_rates = []
    iterations = []
    
    if not os.path.exists(log_path):
        return iterations, success_rates
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all iterations
        iteration_matches = re.findall(r'--- ITERATION (\d+) ---', content)
        
        # Find all success rates
        success_rate_matches = re.findall(r'SUCCESS_RATE = ([\d.]+)', content)
        
        # Convert to float and pair with iterations
        for i, rate in enumerate(success_rate_matches):
            if i < len(iteration_matches):
                iterations.append(int(iteration_matches[i]))
            else:
                # If we have more success rates than iteration markers, 
                # assume they belong to the last iteration
                iterations.append(int(iteration_matches[-1]) if iteration_matches else 1)
            success_rates.append(float(rate))
    
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return [], []
    
    return iterations, success_rates


def get_experiment_info(dir_name):
    """Extract domain and model from directory name."""
    # Handle both formats: domain_model_timestamp and model|domain
    if '|' in dir_name:
        parts = dir_name.split('|')
        model = parts[0]
        domain = parts[1]
    else:
        # Extract from format: domain_model_timestamp
        parts = dir_name.split('_')
        if len(parts) >= 3:
            # Find the model part (usually second to last or third to last)
            model_candidates = ['claude', 'gemini', 'gpt4', 'gpt']
            model = None
            for part in parts:
                if any(candidate in part.lower() for candidate in model_candidates):
                    model = part.lower()
                    break
            
            if model:
                # Domain is everything before the model
                domain_parts = []
                for part in parts:
                    if any(candidate in part.lower() for candidate in model_candidates):
                        break
                    domain_parts.append(part)
                domain = '_'.join(domain_parts)
            else:
                # Fallback: assume first part is domain, second is model
                domain = parts[0]
                model = parts[1] if len(parts) > 1 else 'unknown'
        else:
            domain = 'unknown'
            model = 'unknown'
    
    # Normalize model names
    if 'claude' in model.lower():
        model = 'claude'
    elif 'gemini' in model.lower():
        model = 'gemini'
    elif 'gpt' in model.lower():
        model = 'gpt'
    
    return domain, model


def collect_experiment_data(experiments_dir):
    """Collect all experiment data from the experiments directory."""
    data = defaultdict(lambda: defaultdict(lambda: {'iterations': [], 'success_rates': []}))
    
    if not os.path.exists(experiments_dir):
        print(f"Warning: Experiments directory {experiments_dir} not found")
        return data
    
    for item in os.listdir(experiments_dir):
        item_path = os.path.join(experiments_dir, item)
        
        # Skip files, only process directories
        if not os.path.isdir(item_path):
            continue
        
        # Skip if it's not an experiment directory
        if not any(model in item.lower() for model in ['claude', 'gemini', 'gpt']):
            continue
        
        domain, model = get_experiment_info(item)
        
        # Look for conversation.log
        log_path = os.path.join(item_path, 'conversation.log')
        
        if not os.path.exists(log_path):
            print(f"Warning: No conversation.log found in {item_path}")
            continue
        
        iterations, success_rates = parse_success_rates_from_log(log_path)
        
        if iterations and success_rates:
            # Store the data
            data[domain][model]['iterations'].extend(iterations)
            data[domain][model]['success_rates'].extend(success_rates)
            print(f"Collected {len(success_rates)} data points from {domain}/{model}")
        else:
            print(f"Warning: No success rate data found for {domain}/{model}")
    
    return data


def save_plotting_data(data, output_file):
    """Save collected data to JSON file for plotting."""
    # Convert defaultdict to regular dict for JSON serialization
    regular_data = {}
    for domain, models in data.items():
        regular_data[domain] = {}
        for model, model_data in models.items():
            regular_data[domain][model] = dict(model_data)
    
    with open(output_file, 'w') as f:
        json.dump(regular_data, f, indent=2)
    
    print(f"Data saved to {output_file}")


def print_summary(data):
    """Print a summary of collected data."""
    print("\n=== DATA COLLECTION SUMMARY ===")
    
    if not data:
        print("No data collected.")
        return
    
    for domain, models in data.items():
        print(f"\nDomain: {domain}")
        for model, model_data in models.items():
            iterations = model_data['iterations']
            success_rates = model_data['success_rates']
            if success_rates:
                print(f"  {model}: {len(success_rates)} data points, "
                      f"max iteration: {max(iterations)}, "
                      f"final success rate: {success_rates[-1]:.3f}")
            else:
                print(f"  {model}: No data")


def main():
    """Main function to collect and save plotting data."""
    # Set up paths
    experiments_dir = "/home/omer/projects/AML_VIA_LLM/experiments"
    output_file = "/home/omer/projects/AML_VIA_LLM/plotting_data.json"
    
    print("Collecting experiment data...")
    data = collect_experiment_data(experiments_dir)
    
    print_summary(data)
    
    if data:
        save_plotting_data(data, output_file)
        print(f"\nData ready for plotting! Use the JSON file: {output_file}")
        print("\nExample usage:")
        print("import json")
        print("with open('plotting_data.json') as f:")
        print("    data = json.load(f)")
        print("# Then use data[domain][model]['success_rates'] and data[domain][model]['iterations'] for plotting")
    else:
        print("No data collected. Check experiment directories and log files.")


if __name__ == "__main__":
    main()