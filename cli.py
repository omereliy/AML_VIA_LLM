"""
Command-line interface for PDDL generation experiments
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import config
from config import LLMProvider, DEFAULT_SUCCESS_THRESHOLD, DEFAULT_MAX_ITERATIONS
from llm_providers import LLMConfig, LLMFactory
from system import PDDLGeneratorSystem
from experiment_logging import setup_experiment_logging


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Run PDDL generation experiments with iterative refinement.",
        epilog="""
Default configurations:
  uniform : All agents use gpt-4.
  mixed   : Formalizer uses gpt-4 (low temperature), Critic uses Claude (medium temperature), 
            Investigators use Gemini (high temperature).
  To use uniform, do not specify '--mixed' or '--config'.
  To use mixed, specify '--mixed' without '--config'.

To use a custom configuration, use '--config' with a path to a JSON file with the following structure:
  {
    "formalizer": {
      "model": "gpt-4",
      "api_key": "your-api-key-here",
      "temperature": 0.2,
      "max_tokens": 2048,
      "timeout": 30
    },
    "critic": {
      "model": "claude-3-sonnet",
      "api_key": "your-api-key-here",
      "temperature": 0.4
    },
    "investigator": {
      "model": "gemini-1.5-pro",
      "api_key": "your-api-key-here",
      "temperature": 0.9
    }
  }

Any file that contains more or less than the above will be treated as an error.

Fields 'temperature', 'max_tokens', and 'timeout' are optional.
Default max_tokens is 2000, temperature is 0.7 for all agents, timeout is 30 seconds.
API keys will default to environment variables if not provided.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        help="Provide a path to a custom JSON config file."
    )
    parser.add_argument(
        '--mixed',
        action='store_true',
        help="Use mixed provider configuration."
    )
    parser.add_argument(
        '--description',
        type=str,
        help="Natural language description as text (overrides file input)"
    )
    parser.add_argument(
        '--description-file',
        type=str,
        default='user_domain_description.txt',
        help="Path to file containing domain description (default: user_domain_description.txt)"
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help="Name for this experiment run (used in output files)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help="Directory to save experiment results (default: experiments)"
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum refinement iterations (default: {DEFAULT_MAX_ITERATIONS})"
    )
    parser.add_argument(
        '--success-threshold',
        type=float,
        default=DEFAULT_SUCCESS_THRESHOLD,
        help=f"Success threshold for critic evaluation (default: {DEFAULT_SUCCESS_THRESHOLD})"
    )
    
    return parser


def create_system_from_config_file(config_file: str) -> PDDLGeneratorSystem:
    """Create PDDL generator system from JSON configuration file"""
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    llm_configs = {}
    for role, config_dict in config_data.get('llm_configs', {}).items():
        llm_configs[role] = LLMFactory.create_from_dict(config_dict)

    return PDDLGeneratorSystem.create_with_mixed_providers(
        default_config=llm_configs.get('default', LLMConfig(
            provider=LLMProvider.OPENAI, 
            model_name="gpt-4", 
            temperature=0.7, 
            max_tokens=2000
        )),
        formalizer_config=llm_configs.get('formalizer', None),
        critic_config=llm_configs.get('critic', None),
        investigator_config=llm_configs.get('investigator', None),
        success_threshold=config_data.get('success_threshold', DEFAULT_SUCCESS_THRESHOLD),
        max_iterations=config_data.get('max_iterations', DEFAULT_MAX_ITERATIONS)
    )


def create_uniform_system() -> PDDLGeneratorSystem:
    """Create system with uniform GPT-4 configuration"""
    print("No configuration specified. Using default uniform configuration with gpt-4.")
    gpt4_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=2000
    )
    return PDDLGeneratorSystem.create_with_single_provider(
        gpt4_config,
        success_threshold=DEFAULT_SUCCESS_THRESHOLD,
        max_iterations=DEFAULT_MAX_ITERATIONS
    )


def create_mixed_system() -> PDDLGeneratorSystem:
    """Create system with mixed providers"""
    print("Using mixed configuration with different providers for each agent.")
    mixed_configs = {
        'formalizer': LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.3  # Lower temperature for more consistent PDDL generation
        ),
        'critic': LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.5  # Moderate temperature for evaluation
        ),
        'investigator': LLMConfig(
            provider=LLMProvider.GENAI,
            model_name="gemini-pro",
            temperature=0.7  # Higher temperature for creative problem finding
        ),
        'default': LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
    }
    return PDDLGeneratorSystem.create_with_mixed_providers(
        default_config=mixed_configs['default'],
        formalizer_config=mixed_configs['formalizer'],
        critic_config=mixed_configs['critic'],
        investigator_config=mixed_configs['investigator'],
        success_threshold=DEFAULT_SUCCESS_THRESHOLD,
        max_iterations=DEFAULT_MAX_ITERATIONS
    )


def get_problem_description(args) -> str:
    """Get problem description from command line text or file"""
    if args.description:
        # Use direct text input from command line
        return args.description.strip()
    
    # Try to read from specified file
    description_file = args.description_file
    if os.path.exists(description_file):
        with open(description_file, 'r') as f:
            content = f.read().strip()
            if content:
                return content
    
    # Fallback to default if file doesn't exist or is empty
    print(f"Warning: {description_file} not found or empty, using default description")
    return """
This is a domain where we have blocks and a table. 
Blocks can be stacked on top of each other or placed on the table. 
The goal is to move blocks from one configuration to another.
We need actions to pick up blocks, put them down, and stack them.
A robot arm can only hold one block at a time.
"""


def save_experiment_result(result: str, experiment_name: str = None, output_dir: str = "experiments") -> str:
    """Save the generated PDDL to an experiment file (final result)"""
    # The experiment folder structure is now handled by ExperimentLogger
    # This function is kept for backward compatibility but may not be needed
    # since final_domain.pddl is saved automatically by the logger
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        folder_name = f"{experiment_name}_{timestamp}"
    else:
        folder_name = f"experiment_{timestamp}"
    
    # Create experiment folder
    base_dir = Path(output_dir)
    exp_dir = base_dir / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as final_domain.pddl in the experiment folder
    file_path = exp_dir / "final_domain.pddl"
    with open(str(file_path), 'w') as f:
        f.write(result)
    
    return str(file_path)


def main():
    """Main CLI entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    load_dotenv()  # Load environment variables from .env file if it exists

    # Validate arguments
    if args.mixed and args.config:
        raise ValueError("Cannot use both --mixed and --config. Choose one configuration method.")

    # Create system based on configuration
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        llm_system = create_system_from_config_file(args.config)
    elif args.mixed:
        llm_system = create_mixed_system()
    else:
        llm_system = create_uniform_system()
    
    # Override system parameters if provided
    if hasattr(llm_system, 'max_iterations'):
        llm_system.max_iterations = args.max_iterations
    if hasattr(llm_system, 'success_threshold'):
        llm_system.success_threshold = args.success_threshold

    # Get problem description
    problem_desc = get_problem_description(args)
    
    # Print experiment configuration
    print("\n=== PDDL Generation Experiment ===")
    print(f"Experiment name: {args.experiment_name or 'unnamed'}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Success threshold: {args.success_threshold}")
    print(f"Output directory: {args.output_dir}")
    print(f"Description source: {'command line' if args.description else args.description_file}")
    print("\nDomain description:")
    print("-" * 40)
    print(problem_desc)
    print("-" * 40)
    
    # Set up experiment logging (archived by timestamp)
    exp_logger = setup_experiment_logging(args.experiment_name, args.output_dir)
    
    try:
        # Log experiment start
        config_info = {
            "max_iterations": args.max_iterations,
            "success_threshold": args.success_threshold,
            "configuration": "mixed" if args.mixed else ("custom" if args.config else "uniform"),
            "description_source": "command line" if args.description else args.description_file
        }
        exp_logger.log_experiment_start(problem_desc, config_info)
        
        print("\nStarting iterative PDDL generation...")
        result = llm_system.generate_pddl(problem_desc)
        
        # Log experiment end
        success = bool(result and result.strip())
        exp_logger.log_experiment_end(result, success)
        
        print("\n=== Final Generated PDDL ===")
        print(result)
        
        print(f"\n✅ Experiment completed! Results archived to:")
        print(f"   Experiment folder: {exp_logger.experiment_dir}")
        print(f"   Conversation log: {exp_logger.log_file}")
        print(f"   Iteration PDDLs: iteration_1.pddl, iteration_2.pddl, ...")
        print(f"   Final PDDL: final_domain.pddl")
        
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        print("\nTroubleshooting:")
        print("1. Check API keys: python validate_api_keys.py")
        print("2. Verify required packages are installed")
        print("3. Check domain description file exists and is readable")


if __name__ == "__main__":
    main()