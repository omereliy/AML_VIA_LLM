"""
LLM Provider interfaces, implementations, and configurations
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from config import LLMProvider, PROVIDER_TO_MODEL_LIST

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = field(init=False, default=None)
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    
    # Provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization to set API key from environment, and model name from defaults -  if not provided"""
        if not self.api_key:
            # Map provider enum values to actual environment variable names
            env_key_mapping = {
                "gpt4": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", 
                "genai": "GOOGLE_API_KEY",
                "cohere": "COHERE_API_KEY"
            }
            
            env_var_name = env_key_mapping.get(self.provider.value, f"{self.provider.value.upper()}_API_KEY")
            env_key = os.getenv(env_var_name)
            if env_key:
                self.api_key = env_key
            else:
                raise ValueError(f"API key for {self.provider.value} provider is required but not provided. Expected environment variable: {env_var_name}")

        # Set default model name based on provider
        if not self.model_name:
            self.model_name = PROVIDER_TO_MODEL_LIST.get(self.provider, [""])[0]  # Default to first model in list


class LLMInterface(ABC):
    """Abstract interface for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config: LLMConfig = config
        self._validate_config()
    
    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def _validate_config(self):
        """Validate provider-specific configuration"""
        pass


class OpenAIProvider(LLMInterface):
    """OpenAI (chat-gpt) provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("openai package required for openai models. Install with: pip install openai")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            return response.choices[0].message.content
        except Exception as err:
            logger.error(f"GPT-4 generation failed: {err}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")
        if not self.config.api_key:
            raise ValueError("OpenAI API key required")
        if not self.config.model_name:
            self.config.model_name = "gpt-4-turbo"


class AnthropicProvider(LLMInterface):
    """Anthropic (Claude) provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            message_content = user_prompt
            if system_prompt:
                message_content = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": message_content}]
            )
            return response.content[0].text
        except Exception as err:
            logger.error(f"Anthropic generation failed: {err}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.config.api_key:
            raise ValueError("Anthropic API key required")
        if not self.config.model_name:
            self.config.model_name = "claude-3-sonnet-20240229"


class GenAIProvider(LLMInterface):
    """Google Generative AI provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            self.model = genai.GenerativeModel(config.model_name)
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            full_prompt = user_prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                }
            )
            return response.text
        except Exception as err:
            logger.error(f"GenAI generation failed: {err}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.config.api_key:
            raise ValueError("Google API key required")
        if not self.config.model_name:
            self.config.model_name = "gemini-pro"


class CohereProvider(LLMInterface):
    """Cohere provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import cohere
            self.client = cohere.Client(api_key=config.api_key)
        except ImportError:
            raise ImportError("cohere package required. Install with: pip install cohere")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            full_prompt = user_prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.client.generate(
                model=self.config.model_name,
                prompt=full_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.generations[0].text
        except Exception as err:
            logger.error(f"Cohere generation failed: {err}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("COHERE_API_KEY")
        if not self.config.api_key:
            raise ValueError("Cohere API key required")
        if not self.config.model_name:
            self.config.model_name = "command"


class LLMFactory:
    """Factory for creating LLM providers"""
    
    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.GENAI: GenAIProvider,
        LLMProvider.COHERE: CohereProvider,
    }
    
    @classmethod
    def create_provider(cls, config: LLMConfig) -> LLMInterface:
        """Create an LLM provider instance"""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        return provider_class(config)
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> LLMInterface:
        """Create provider from dictionary configuration"""
        provider_name = config_dict.get("provider")
        if isinstance(provider_name, str):
            provider = LLMProvider(provider_name)
        else:
            provider = provider_name
        
        config = LLMConfig(
            provider=provider,
            model_name=config_dict.get("model_name", ""),
            base_url=config_dict.get("base_url"),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens", 2000),
            timeout=config_dict.get("timeout", 30),
            extra_params=config_dict.get("extra_params", {})
        )
        
        return cls.create_provider(config)


def get_default_configs() -> Dict[str, LLMConfig]:
    """Get default LLM configurations"""
    default_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=2000
    )

    return {
        'default': default_config,
        'formalizer': default_config,
        'critic': default_config,
        'investigator': default_config
    }