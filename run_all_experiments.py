#!/usr/bin/env python3
"""
Comprehensive PDDL Generation Experiment Runner

Runs experiments for all domain descriptions across all available LLM models.
Each experiment is properly named and organized for data analysis.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import concurrent.futures
import threading

from config import LLMProvider, DEFAULT_SUCCESS_THRESHOLD, DEFAULT_MAX_ITERATIONS
from llm_providers import LLMConfig, LLMFactory
from system import PDDLGeneratorSystem
from experiment_logging import setup_experiment_logging


def get_domain_descriptions() -> List[Tuple[str, str]]:
    """Get all domain description files and their content"""
    domain_dir = Path("domain descriptions")
    descriptions = []
    
    if not domain_dir.exists():
        print(f"❌ Domain descriptions directory not found: {domain_dir}")
        return descriptions
    
    for desc_file in domain_dir.glob("*.txt"):
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    descriptions.append((desc_file.stem, content))
                    print(f"📄 Found domain: {desc_file.stem}")
        except Exception as e:
            print(f"⚠️  Could not read {desc_file}: {e}")
    
    return descriptions


def get_model_configurations() -> Dict[str, Dict]:
    """Get all available model configurations for experiments"""
    models = {
        "gpt4": {
            "provider": LLMProvider.OPENAI,
            "model_name": "gpt-4",
            "display_name": "GPT-4"
        },
        "claude": {
            "provider": LLMProvider.ANTHROPIC,
            "model_name": "claude-3-5-sonnet-20241022",
            "display_name": "Claude-3.5-Sonnet"
        },
        "gemini": {
            "provider": LLMProvider.GENAI,
            "model_name": "gemini-1.5-flash",
            "display_name": "Gemini-1.5-Flash"
        }
    }
    
    # Verify which models are actually available by checking API keys
    available_models = {}
    
    for model_key, config in models.items():
        try:
            llm_config = LLMConfig(
                provider=config["provider"],
                model_name=config["model_name"],
                temperature=0.7,
                max_tokens=2000
            )
            # Try to create the provider to verify API key
            LLMFactory.create_provider(llm_config)
            available_models[model_key] = config
            print(f"✅ {config['display_name']} available")
        except Exception as e:
            print(f"❌ {config['display_name']} not available: {str(e)[:50]}...")
    
    return available_models


def create_experiment_name(domain_name: str, model_name: str, timestamp: str) -> str:
    """Create a standardized experiment name for data analysis"""
    # Format: domain_model_timestamp
    return f"{domain_name}_{model_name}_{timestamp}"


def run_single_experiment(domain_name: str, domain_description: str, 
                         model_key: str, model_config: Dict,
                         experiment_params: Dict, thread_id: int = 0) -> Dict:
    """Run a single experiment and return results"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = create_experiment_name(domain_name, model_key, timestamp)
    
    print(f"\n🧪 [Thread-{thread_id}] Starting experiment: {experiment_name}")
    print(f"   Domain: {domain_name}")
    print(f"   Model: {model_config['display_name']}")
    print(f"   Description: {domain_description[:100]}...")
    
    # Set up experiment logging
    exp_logger = setup_experiment_logging(experiment_name, "experiments")
    
    try:
        # Create LLM configuration
        llm_config = LLMConfig(
            provider=model_config["provider"],
            model_name=model_config["model_name"],
            temperature=experiment_params.get("temperature", 0.7),
            max_tokens=experiment_params.get("max_tokens", 2000)
        )
        
        # Create system with single provider
        system = PDDLGeneratorSystem.create_with_single_provider(
            llm_config,
            success_threshold=experiment_params.get("success_threshold", DEFAULT_SUCCESS_THRESHOLD),
            max_iterations=experiment_params.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        )
        
        # Log experiment metadata
        metadata = {
            "domain": domain_name,
            "model": model_config["display_name"],
            "model_key": model_key,
            "provider": model_config["provider"].value,
            "model_name": model_config["model_name"],
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            **experiment_params
        }
        
        exp_logger.log_experiment_start(domain_description, metadata)
        
        # Run the experiment
        start_time = time.time()
        result = system.generate_pddl(domain_description)
        end_time = time.time()
        
        # Log completion
        success = bool(result and result.strip())
        exp_logger.log_experiment_end(result, success)
        
        experiment_result = {
            "experiment_name": experiment_name,
            "domain": domain_name,
            "model": model_config["display_name"],
            "model_key": model_key,
            "success": success,
            "duration_seconds": round(end_time - start_time, 2),
            "result_length": len(result) if result else 0,
            "experiment_dir": str(exp_logger.experiment_dir),
            "timestamp": timestamp,
            **metadata
        }
        
        print(f"✅ [Thread-{thread_id}] Completed: {experiment_name} ({experiment_result['duration_seconds']}s)")
        return experiment_result
        
    except Exception as e:
        print(f"❌ [Thread-{thread_id}] Failed: {experiment_name} - {str(e)}")
        exp_logger.log_experiment_end(f"ERROR: {str(e)}", False)
        
        return {
            "experiment_name": experiment_name,
            "domain": domain_name,
            "model": model_config["display_name"],
            "model_key": model_key,
            "success": False,
            "error": str(e),
            "timestamp": timestamp
        }


def save_experiment_summary(results: List[Dict], output_dir: str = "experiments"):
    """Save a comprehensive summary of all experiments"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(output_dir) / f"experiment_summary_{timestamp}.json"
    
    # Create summary statistics
    summary = {
        "run_timestamp": timestamp,
        "total_experiments": len(results),
        "successful_experiments": sum(1 for r in results if r.get("success", False)),
        "failed_experiments": sum(1 for r in results if not r.get("success", False)),
        "domains_tested": list(set(r["domain"] for r in results)),
        "models_tested": list(set(r["model_key"] for r in results)),
        "experiments": results
    }
    
    # Add per-domain and per-model statistics
    summary["statistics"] = {
        "by_domain": {},
        "by_model": {}
    }
    
    for domain in summary["domains_tested"]:
        domain_results = [r for r in results if r["domain"] == domain]
        summary["statistics"]["by_domain"][domain] = {
            "total": len(domain_results),
            "successful": sum(1 for r in domain_results if r.get("success", False)),
            "success_rate": sum(1 for r in domain_results if r.get("success", False)) / len(domain_results)
        }
    
    for model in summary["models_tested"]:
        model_results = [r for r in results if r["model_key"] == model]
        summary["statistics"]["by_model"][model] = {
            "total": len(model_results),
            "successful": sum(1 for r in model_results if r.get("success", False)),
            "success_rate": sum(1 for r in model_results if r.get("success", False)) / len(model_results),
            "avg_duration": sum(r.get("duration_seconds", 0) for r in model_results) / len(model_results)
        }
    
    # Save summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Experiment summary saved to: {summary_file}")
    return summary_file


def run_experiments_concurrently(domains: List[Tuple[str, str]], 
                                models: Dict[str, Dict], 
                                experiment_params: Dict,
                                max_workers: int = None) -> List[Dict]:
    """Run experiments concurrently using ThreadPoolExecutor"""
    
    if max_workers is None:
        # Use a conservative number of workers to avoid overwhelming APIs
        max_workers = min(4, len(models))
    
    print(f"\n🚀 Using {max_workers} concurrent workers for experiments")
    
    # Create list of all experiment tasks
    experiment_tasks = []
    for domain_name, domain_description in domains:
        for model_key, model_config in models.items():
            experiment_tasks.append((domain_name, domain_description, model_key, model_config))
    
    results = []
    completed_count = 0
    total_experiments = len(experiment_tasks)
    
    # Use a lock for thread-safe printing of progress
    print_lock = threading.Lock()
    
    def run_experiment_wrapper(task_data):
        domain_name, domain_description, model_key, model_config = task_data
        thread_id = threading.current_thread().ident % 1000  # Short thread ID for display
        
        # Small delay to avoid overwhelming APIs with simultaneous requests
        time.sleep(0.1)
        
        try:
            result = run_single_experiment(
                domain_name=domain_name,
                domain_description=domain_description,
                model_key=model_key,
                model_config=model_config,
                experiment_params=experiment_params,
                thread_id=thread_id
            )
            
            # Thread-safe progress update
            nonlocal completed_count
            with print_lock:
                completed_count += 1
                progress = (completed_count / total_experiments) * 100
                print(f"📊 Progress: {completed_count}/{total_experiments} ({progress:.1f}%) completed")
            
            return result
            
        except Exception as e:
            with print_lock:
                completed_count += 1
                print(f"❌ [Thread-{thread_id}] Experiment failed: {domain_name}_{model_key} - {str(e)}")
            
            return {
                "experiment_name": f"{domain_name}_{model_key}_failed",
                "domain": domain_name,
                "model": model_config.get("display_name", model_key),
                "model_key": model_key,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
    
    # Execute experiments concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_experiment_wrapper, task): task 
            for task in experiment_tasks
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = future_to_task[future]
                domain_name, _, model_key, model_config = task
                print(f"❌ Unexpected error in experiment {domain_name}_{model_key}: {e}")
                results.append({
                    "experiment_name": f"{domain_name}_{model_key}_error",
                    "domain": domain_name,
                    "model": model_config.get("display_name", model_key),
                    "model_key": model_key,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                })
    
    return results


def print_experiment_summary(results: List[Dict]):
    """Print a formatted summary of experiment results"""
    print(f"\n{'='*60}")
    print("🧪 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    failed = total - successful
    
    print(f"📊 Total experiments: {total}")
    print(f"✅ Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
    
    # Group by domain
    domains = {}
    for result in results:
        domain = result["domain"]
        if domain not in domains:
            domains[domain] = {"total": 0, "successful": 0, "models": []}
        domains[domain]["total"] += 1
        if result.get("success", False):
            domains[domain]["successful"] += 1
        domains[domain]["models"].append(result["model_key"])
    
    print(f"\n📄 Results by domain:")
    for domain, stats in domains.items():
        success_rate = stats["successful"] / stats["total"] * 100
        print(f"   {domain}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
    
    # Group by model
    models = {}
    for result in results:
        model = result["model_key"]
        if model not in models:
            models[model] = {"total": 0, "successful": 0}
        models[model]["total"] += 1
        if result.get("success", False):
            models[model]["successful"] += 1
    
    print(f"\n🤖 Results by model:")
    for model, stats in models.items():
        success_rate = stats["successful"] / stats["total"] * 100
        print(f"   {model}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")


def main():
    """Main experiment runner"""
    import argparse
    
    # Add command line argument parsing for concurrency control
    parser = argparse.ArgumentParser(description="Run comprehensive PDDL generation experiments")
    parser.add_argument(
        '--max-workers', 
        type=int, 
        default=None,
        help="Maximum number of concurrent workers (default: min(4, number_of_models))"
    )
    parser.add_argument(
        '--sequential', 
        action='store_true',
        help="Run experiments sequentially instead of concurrently"
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=0.5,
        help="Delay between experiment starts in seconds (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    print("🚀 PDDL Generation Comprehensive Experiment Runner")
    print("="*60)
    print(f"🔧 Concurrency: {'Sequential' if args.sequential else 'Concurrent'}")
    if not args.sequential:
        max_workers = args.max_workers or "auto"
        print(f"🔧 Max workers: {max_workers}")
    print(f"🔧 Inter-experiment delay: {args.delay}s")
    
    # Get domain descriptions
    print("\n📁 Loading domain descriptions...")
    domains = get_domain_descriptions()
    if not domains:
        print("❌ No domain descriptions found!")
        sys.exit(1)
    
    print(f"Found {len(domains)} domains: {[d[0] for d in domains]}")
    
    # Get available models
    print("\n🤖 Checking available models...")
    models = get_model_configurations()
    if not models:
        print("❌ No models available! Check your API keys.")
        sys.exit(1)
    
    print(f"Available models: {list(models.keys())}")
    
    # Set experiment parameters
    experiment_params = {
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "success_threshold": DEFAULT_SUCCESS_THRESHOLD,
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    print(f"\n⚙️  Experiment parameters:")
    for key, value in experiment_params.items():
        print(f"   {key}: {value}")
    
    # Calculate total experiments
    total_experiments = len(domains) * len(models)
    print(f"\n🧮 Total experiments to run: {total_experiments}")
    print(f"   {len(domains)} domains × {len(models)} models")
    
    # Confirm before starting
    response = input("\n🚦 Start experiments? (y/N): ").strip().lower()
    if response != 'y':
        print("Experiments cancelled.")
        sys.exit(0)
    
    # Run all experiments
    start_time = time.time()
    
    if args.sequential:
        print(f"\n🏃 Running experiments sequentially...")
        results = []
        
        for i, (domain_name, domain_description) in enumerate(domains, 1):
            print(f"\n📄 Domain {i}/{len(domains)}: {domain_name}")
            
            for j, (model_key, model_config) in enumerate(models.items(), 1):
                print(f"🤖 Model {j}/{len(models)}: {model_config['display_name']}")
                
                result = run_single_experiment(
                    domain_name=domain_name,
                    domain_description=domain_description,
                    model_key=model_key,
                    model_config=model_config,
                    experiment_params=experiment_params,
                    thread_id=0
                )
                
                results.append(result)
                
                # Configurable delay between experiments
                if args.delay > 0:
                    time.sleep(args.delay)
    else:
        print(f"\n🏃 Running experiments concurrently...")
        results = run_experiments_concurrently(
            domains=domains,
            models=models,
            experiment_params=experiment_params,
            max_workers=args.max_workers
        )
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Print and save summary
    print_experiment_summary(results)
    summary_file = save_experiment_summary(results)
    
    print(f"\n🎉 All experiments completed!")
    print(f"⏱️  Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"📊 Average time per experiment: {total_duration/len(results):.1f} seconds")
    print(f"📁 Individual results: experiments/")
    print(f"📊 Summary report: {summary_file}")


if __name__ == "__main__":
    main()