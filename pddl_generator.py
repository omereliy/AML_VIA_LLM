"""
Multi-Agent PDDL Generator System

A sophisticated system for converting natural language descriptions into PDDL domains
using multiple specialized LLM agents with support for various providers.

Installation Requirements:
pip install openai anthropic google-generativeai cohere requests

Optional (for specific providers):
- OpenAI: pip install openai
- Anthropic: pip install anthropic  
- Google: pip install google-generativeai
- Cohere: pip install cohere

Environment Variables (optional, can also pass via config):
- OPENAI_API_KEY: Your OpenAI API key
- ANTHROPIC_API_KEY: Your Anthropic API key
- GOOGLE_API_KEY: Your Google AI API key
- COHERE_API_KEY: Your Cohere API key
"""

import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y|%H:%M:%S')

logger.addHandler(console_handler)
console_handler.setFormatter(formatter)

# Import everything from the modular components
from config import *
from llm_providers import *
from pddl_models import *
from agents import *
from system import *

# Import CLI for backward compatibility
if __name__ == "__main__":
    from cli import main
    main()
