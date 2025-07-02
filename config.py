"""
Configuration constants and defaults for the PDDL Generator System
"""
import os
from enum import Enum
from typing import List
from dotenv import load_dotenv

load_dotenv()

# HYPERPARAMETERS
DEFAULT_SUCCESS_THRESHOLD = 0.95
DEFAULT_MAX_ITERATIONS = 3

try:
    OUT_DIR = os.environ.get("OUT_DIR")
except KeyError:
    OUT_DIR=""


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "gpt4"
    ANTHROPIC = "anthropic"
    GENAI = "genai"
    COHERE = "cohere"

    @classmethod
    def get_providers(cls) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in cls]


# Provider to model mappings
PROVIDER_TO_MODEL_LIST = {
    LLMProvider.OPENAI: [
        "gpt-3.5-turbo",                        # Fast and cheap, 16k context by default
        "gpt-4-turbo",                          # GPT-4 Turbo (usually GPT-4.0 backend), 128k context
        "gpt-4o",                               # GPT-4 Omni, multimodal, fastest GPT-4 model
        "gpt-4",                                # Legacy GPT-4, 8k/32k context
    ],
    LLMProvider.ANTHROPIC: [
        "claude-opus-4-20250514",               # Latest Claude 4 model, multimodal
        "claude-sonnet-4-20250514",             # Claude 4 Sonnet, multimodal
        "claude-3-7-sonnet-20250219",           # Claude 3.7 Sonnet, multimodal
        "claude-3-5-haiku-20241022",            # Claude 3.5 Haiku, multimodal
        "claude-3-5-sonnet-20241022",           # Claude 3.5 Sonnet, multimodal
        "claude-3-haiku-20240307",              # Claude 3 Haiku, multimodal
    ],
    LLMProvider.GENAI: [
        "gemini-2.5-pro",                       # Latest Gemini Pro, multimodal
        "gemini-2.5-flash",                     # Gemini Flash, multimodal
        "gemini-2.5-flash-lite-preview-06-17",  # Gemini Flash Lite, multimodal
        "gemini-2.0-flash",                     # Gemini Flash, multimodal
        "gemini-2.0-flash-lite",                # Gemini Flash Lite, multimodal
        "gemini-1.5-flash",                     # Gemini Flash, multimodal
        "gemini-1.5-pro",                       # Gemini Pro, multimodal
    ],
    LLMProvider.COHERE: [
        "command-a-03-2025",                    # Command-A 03, text
        "command-r7b-12-2024",                  # Command-r7b, faster and cheaper
        "command-r-plus",                       # Command-R+, more expensive but better quality
    ],
}