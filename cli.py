"""
Command-line interface and configuration handling
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


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Choose an LLM configuration for your multi-agent system.",
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
        help="Path to the description of the problem in natural language"
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
    """Get problem description from args or use default"""
    if args.description:
        with open(args.description, 'r') as f:
            return f.read().strip()
    
    # Default problem description
    return """
    This is a domain where we have blocks and a table. 
    Blocks can be stacked on top of each other or placed on the table. 
    The goal is to move blocks from one configuration to another.
    We need actions to pick up blocks, put them down, and stack them.
    A robot arm can only hold one block at a time.
    """


def save_result(result: str) -> str:
    """Save the generated PDDL to a timestamped file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_pddl_{timestamp}.pddl"
    out_dir = Path(config.OUT_DIR) if config.OUT_DIR else Path(".")
    file_path = out_dir / filename
    with open(str(file_path), 'w') as f:
        f.write(result)
    return filename


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

    # Get problem description
    problem_desc = get_problem_description(args)
    
    print("\n=== Generating PDDL with selected system ===")
    
    try:
        result = llm_system.generate_pddl(problem_desc)
        print("Generated PDDL:")
        print(result)
        
        filename = save_result(result)
        print(f"\nResult saved to: {filename}")
        
    except Exception as e:
        print(f"Error generating PDDL: {e}")
        print("Make sure you have:")
        print("1. Required packages installed (pip install openai anthropic google-generativeai cohere requests)")
        print("2. API keys set in environment variables or config")


if __name__ == "__main__":
    main()