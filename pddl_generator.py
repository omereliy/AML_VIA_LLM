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
- Hugging Face: pip install requests (built-in)
- Ollama: pip install requests (built-in) + local Ollama server

Environment Variables (optional, can also pass via config):
- OPENAI_API_KEY: Your OpenAI API key
- ANTHROPIC_API_KEY: Your Anthropic API key
- GOOGLE_API_KEY: Your Google AI API key
- COHERE_API_KEY: Your Cohere API key
- HUGGINGFACE_API_KEY: Your Hugging Face API key
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
import re
from enum import Enum
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary mapping of model names to provider names
model_name_to_provider_name = {
    # OpenAI
    'gpt-4.1': 'OPENAI',
    'gpt-4.1-mini': 'OPENAI',
    'gpt-4.1-nano': 'OPENAI',
    'gpt-4o': 'OPENAI',
    'gpt-4-turbo': 'OPENAI',
    'gpt-4': 'OPENAI',
    'gpt-3.5-turbo': 'OPENAI',
    'text-davinci-003': 'OPENAI',
    'text-davinci-002': 'OPENAI',
    'code-davinci-002': 'OPENAI',
    'davinci': 'OPENAI',
    'curie': 'OPENAI',
    'babbage': 'OPENAI',
    'ada': 'OPENAI',

    # Anthropic
    'claude-4-opus': 'ANTHROPIC',
    'claude-4-sonnet': 'ANTHROPIC',
    'claude-4-haiku': 'ANTHROPIC',
    'claude-3-opus': 'ANTHROPIC',
    'claude-3-sonnet': 'ANTHROPIC',
    'claude-3-haiku': 'ANTHROPIC',

    # Google (GenAI / Gemini)
    'gemini-1.5-pro': 'GENAI',
    'gemini-1.5-flash': 'GENAI',
    'gemini-1.0-pro': 'GENAI',
    'gemini-1.0-ultra': 'GENAI',
    'gemini-1.0-lite': 'GENAI',

    # Cohere
    'command-r': 'COHERE',
    'command-r+': 'COHERE',
    'command-r7b': 'COHERE',
    'command-a': 'COHERE',
    'command': 'COHERE',
    'rerank': 'COHERE',
    'embed': 'COHERE',

    # HuggingFace (sample models)
    'bert-base-uncased': 'HUGGINGFACE',
    'gpt2': 'HUGGINGFACE',
    'roberta-base': 'HUGGINGFACE',
    'facebook/bart-large': 'HUGGINGFACE',
    'google/flan-t5-xl': 'HUGGINGFACE',
    'EleutherAI/gpt-j-6B': 'HUGGINGFACE',
    'mistralai/Mistral-7B-Instruct-v0.1': 'HUGGINGFACE',
    'meta-llama/Llama-2-7b-hf': 'HUGGINGFACE',

    # Ollama (sample models)
    'mistral:7b': 'OLLAMA',
    'qwen2.5:7b': 'OLLAMA',
    'qwen2.5:14b': 'OLLAMA',
    'qwen2.5:32b': 'OLLAMA',
    'qwen2.5:72b': 'OLLAMA',
    'granite3.3:2b': 'OLLAMA',
    'granite3.3:8b': 'OLLAMA',
    'granite3.2-vision:2b': 'OLLAMA',
}

#HYPERPARAMETERS
DEFAULT_SUCCESS_THRESHOLD = 0.8
DEFAULT_MAX_ITERATIONS = 3

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "gpt4"
    ANTHROPIC = "anthropic"
    GENAI = "genai" 
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

    @classmethod
    def get_providers(cls) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in cls]

@dataclass(slots=True)
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    
    # Provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class PDDLDomain:
    """Represents a PDDL domain with all its components"""
    domain_name: str
    types: List[str]
    constants: List[str]
    predicates: List[str]
    functions: List[str]
    actions: List[str]
    raw_text: str

@dataclass(slots=True)
class InvestigationReport:
    """Report from an investigator agent"""
    investigator_type: str
    issues_found: List[str]
    severity_scores: List[float]  # 0-1 scale for each issue
    suggestions: List[str]

@dataclass(slots=True)
class SuccessRateEvaluation:
    """Evaluation from the Success Rate Critic"""
    success_rate: float
    reasoning: str
    passes_threshold: bool
    specific_concerns: List[str]

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
        except Exception as e:
            logger.error(f"GPT-4 generation failed: {e}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")
        if not self.config.api_key:
            raise ValueError("OpenAI API key required") #TODO: might want to add a quick manual on how to get and set the API key
        if not self.config.model_name:
            self.config.model_name = "gpt-4"

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
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.config.api_key:
            raise ValueError("Anthropic API key required") #TODO: might want to add a quick manual on how to get and set the API key
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
        except Exception as e:
            logger.error(f"GenAI generation failed: {e}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.config.api_key:
            raise ValueError("Google API key required") #TODO: might want to add a quick manual on how to get and set the API key
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
        except Exception as e:
            logger.error(f"Cohere generation failed: {e}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("COHERE_API_KEY")
        if not self.config.api_key:
            raise ValueError("Cohere API key required") #TODO: might want to add a quick manual on how to get and set the API key
        if not self.config.model_name:
            self.config.model_name = "command"

class HuggingFaceProvider(LLMInterface):
    """Hugging Face provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import requests
            self.session = requests.Session()
            if config.api_key:
                self.session.headers.update({"Authorization": f"Bearer {config.api_key}"})
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            full_prompt = user_prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            api_url = self.config.base_url or f"https://api-inference.huggingface.co/models/{self.config.model_name}"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "temperature": self.config.temperature,
                    "max_new_tokens": self.config.max_tokens,
                    "return_full_text": False
                }
            }
            
            response = self.session.post(api_url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            raise
    
    def _validate_config(self):
        if not self.config.api_key:
            self.config.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.config.model_name:
            self.config.model_name = "microsoft/DialoGPT-large"

class OllamaProvider(LLMInterface):
    """Ollama local provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            api_url = self.config.base_url or "http://localhost:11434/api/generate"
            
            payload = {
                "model": self.config.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            }
            
            response = self.session.post(api_url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def _validate_config(self):
        if not self.config.model_name:
            self.config.model_name = "llama2"
        if not self.config.base_url:
            self.config.base_url = "http://localhost:11434/api/generate"

class LLMFactory:
    """Factory for creating LLM providers"""
    
    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.GENAI: GenAIProvider,
        LLMProvider.COHERE: CohereProvider,
        LLMProvider.HUGGINGFACE: HuggingFaceProvider,
        LLMProvider.OLLAMA: OllamaProvider,
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
            api_key=config_dict.get("api_key"),
            base_url=config_dict.get("base_url"),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens", 2000),
            timeout=config_dict.get("timeout", 30),
            extra_params=config_dict.get("extra_params", {})
        )
        
        return cls.create_provider(config)

class LLMAgent(ABC):
    """Base class for all LLM agents"""
    
    def __init__(self, name: str, llm_provider: LLMInterface):
        self.name = name
        self.llm = llm_provider
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Process input and return output"""
        pass

class FormalizerAgent(LLMAgent):
    """Agent responsible for formalizing natural language to PDDL"""
    
    def __init__(self, llm_provider: LLMInterface):
        super().__init__("Formalizer", llm_provider)
        self.system_prompt = """You are an expert in PDDL (Planning Domain Definition Language) and formal logic. 
Your task is to convert natural language descriptions of planning domains into valid PDDL domain files.

Guidelines:
- Generate complete, syntactically correct PDDL domains
- Include all necessary sections: domain name, requirements, types, predicates, and actions
- Ensure actions have proper parameters, preconditions, and effects
- Use appropriate typing for all variables
- Follow PDDL syntax strictly
- Only output the PDDL domain definition, no explanations or additional text"""
    
    def initial_formalization(self, natural_language_description: str) -> PDDLDomain:
        """Convert natural language description to initial PDDL domain"""
        logger.info(f"[{self.name}] Starting initial formalization")
        
        prompt = f"""Convert the following natural language description into a PDDL domain:

{natural_language_description}

Generate a complete PDDL domain definition:"""
        
        pddl_text = self.llm.generate(prompt, self.system_prompt)
        return self._parse_pddl_domain(pddl_text, natural_language_description)
    
    def re_formalization(self, original_description: str, feedback_prompt: str) -> PDDLDomain:
        """Re-formalize based on investigator feedback"""
        logger.info(f"[{self.name}] Starting re-formalization with feedback")
        
        prompt = f"""Original Description: {original_description}

Issues Found and Feedback:
{feedback_prompt}

Please generate an improved PDDL domain that addresses all the identified issues while maintaining the core functionality described in the original description:"""
        
        pddl_text = self.llm.generate(prompt, self.system_prompt)
        return self._parse_pddl_domain(pddl_text, original_description)
    
    def _parse_pddl_domain(self, pddl_text: str, original_description: str) -> PDDLDomain:
        """Parse PDDL text into structured domain object"""
        # Simple regex-based parsing (could be enhanced with proper PDDL parser)
        domain_name_match = re.search(r'\(define \(domain ([^)]+)\)', pddl_text)
        domain_name = domain_name_match.group(1) if domain_name_match else "unknown"
        
        types = self._extract_section(pddl_text, "types")
        constants = self._extract_section(pddl_text, "constants")
        predicates = self._extract_section(pddl_text, "predicates")
        functions = self._extract_section(pddl_text, "functions")
        actions = self._extract_actions(pddl_text)
        
        return PDDLDomain(
            domain_name=domain_name,
            types=types,
            constants=constants,
            predicates=predicates,
            functions=functions,
            actions=actions,
            raw_text=pddl_text
        )
    
    def _extract_section(self, pddl_text: str, section_name: str) -> List[str]:
        """Extract items from a PDDL section"""
        pattern = rf'\(:{section_name}([^)]*)\)'
        match = re.search(pattern, pddl_text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return [item.strip() for item in content.split() if item.strip()]
        return []
    
    def _extract_actions(self, pddl_text: str) -> List[str]:
        """Extract action definitions"""
        actions = []
        action_pattern = r'\(:action[^)]*(?:\([^)]*\)[^)]*)*\)'
        matches = re.findall(action_pattern, pddl_text, re.DOTALL)
        return matches

class SuccessRateCritic(LLMAgent):
    """Agent that evaluates how well the PDDL domain matches the description"""
    
    def __init__(self, llm_provider: LLMInterface, threshold: float = 0.8):
        super().__init__("SuccessRateCritic", llm_provider)
        self.threshold = threshold
        self.system_prompt = """You are an expert evaluator of PDDL domains. Your task is to assess how well a generated PDDL domain matches the original natural language description.

Evaluation criteria:
- Completeness: Does the domain capture all key elements from the description?
- Correctness: Is the PDDL syntax correct and semantically valid?
- Appropriateness: Are the predicates, actions, and types suitable for the described domain?
- Consistency: Are all parts of the domain consistent with each other?

Provide a success rate between 0.0 and 1.0, detailed reasoning, and specific concerns."""
    
    def evaluate(self, natural_description: str, pddl_domain: PDDLDomain) -> SuccessRateEvaluation:
        """Evaluate the success rate of the formalization"""
        logger.info(f"[{self.name}] Evaluating domain against description")
        
        prompt = f"""Evaluate how well this PDDL domain matches the natural language description:

NATURAL LANGUAGE DESCRIPTION:
{natural_description}

GENERATED PDDL DOMAIN:
{pddl_domain.raw_text}

Please provide your evaluation in the following format:
SUCCESS_RATE: [0.0-1.0]
REASONING: [Detailed explanation of the score]
CONCERNS: [List specific issues or areas for improvement, one per line]"""
        
        response = self.llm.generate(prompt, self.system_prompt)
        return self._parse_evaluation_response(response)
    
    def _parse_evaluation_response(self, response: str) -> SuccessRateEvaluation:
        """Parse the LLM evaluation response"""
        try:
            # Extract success rate
            success_rate = 0.7  # default
            if "SUCCESS_RATE:" in response:
                rate_line = [line for line in response.split('\n') if 'SUCCESS_RATE:' in line][0]
                success_rate = float(rate_line.split('SUCCESS_RATE:')[1].strip())
            
            # Extract reasoning
            reasoning = "Evaluation completed" # default
            if "REASONING:" in response:
                reasoning_start = response.find("REASONING:") + len("REASONING:")
                reasoning_end = response.find("CONCERNS:")
                if reasoning_end == -1:
                    reasoning_end = len(response)
                reasoning = response[reasoning_start:reasoning_end].strip()
            
            # Extract concerns
            concerns = [] # default - no concerns
            if "CONCERNS:" in response:
                concerns_start = response.find("CONCERNS:") + len("CONCERNS:")
                concerns_text = response[concerns_start:].strip()
                concerns = [line.strip() for line in concerns_text.split('\n') if line.strip()]
            
            return SuccessRateEvaluation(
                success_rate=success_rate,
                reasoning=reasoning,
                passes_threshold=success_rate >= self.threshold,
                specific_concerns=concerns
            )
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            # Return default evaluation
            return SuccessRateEvaluation(
                success_rate=0.5,
                reasoning="Failed to parse evaluation response",
                passes_threshold=False,
                specific_concerns=["Evaluation parsing error"]
            )

class InvestigatorAgent(LLMAgent):
    """Base class for specialized investigators"""
    
    def __init__(self, name: str, focus_area: str, llm_provider: LLMInterface):
        super().__init__(name, llm_provider)
        self.focus_area = focus_area
        self.system_prompt = f"""You are a specialized PDDL domain investigator focusing on {focus_area}. 
Your task is to analyze PDDL domains for issues specific to your area of expertise.

Guidelines:
- Identify specific technical issues in the PDDL domain
- Provide severity scores (0.0-1.0) for each issue
- Suggest concrete improvements
- Focus only on your area of specialization
- Be thorough but concise"""
    
    def investigate(self, description: str, domain: PDDLDomain) -> InvestigationReport:
        """Investigate the domain for issues in the agent's specialty"""
        logger.info(f"[{self.name}] Investigating {self.focus_area}")
        
        prompt = f"""Analyze this PDDL domain for issues related to {self.focus_area}:

ORIGINAL DESCRIPTION:
{description}

PDDL DOMAIN:
{domain.raw_text}

Please identify issues and provide your analysis in the following format:
ISSUE: [Description of issue]
SEVERITY: [0.0-1.0 score]
SUGGESTION: [How to fix this issue]
---
[Repeat for each issue found]

If no issues are found, respond with "NO_ISSUES_FOUND"."""
        
        response = self.llm.generate(prompt, self.system_prompt)
        return self._parse_investigation_response(response)
    
    def _parse_investigation_response(self, response: str) -> InvestigationReport:
        """Parse the LLM investigation response"""
        if "NO_ISSUES_FOUND" in response.upper():
            return InvestigationReport(
                investigator_type=self.focus_area,
                issues_found=[],
                severity_scores=[],
                suggestions=[]
            )
        
        issues = []
        scores = []
        suggestions = []
        
        # Split by separator and parse each issue block
        issue_blocks = response.split('---')
        for block in issue_blocks:
            if not block.strip():
                continue
                
            issue_text = ""
            severity = 0.5
            suggestion = ""
            
            lines = block.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('ISSUE:'):
                    issue_text = line[6:].strip()
                elif line.startswith('SEVERITY:'):
                    try:
                        severity = float(line[9:].strip())
                    except ValueError:
                        severity = 0.5
                elif line.startswith('SUGGESTION:'):
                    suggestion = line[11:].strip()
            
            if issue_text:
                issues.append(issue_text)
                scores.append(severity)
                suggestions.append(suggestion)
        
        return InvestigationReport(
            investigator_type=self.focus_area,
            issues_found=issues,
            severity_scores=scores,
            suggestions=suggestions
        )

class ActionSignatureInvestigator(InvestigatorAgent):
    """Investigator specializing in action signatures"""
    
    def __init__(self, llm_provider: LLMInterface):
        super().__init__("ActionSignatureInvestigator", "Action Signatures", llm_provider)
        
    def investigate(self, description: str, domain: PDDLDomain) -> InvestigationReport:
        """Override with more specific prompt for action signatures"""
        prompt = f"""Analyze this PDDL domain specifically for ACTION SIGNATURE issues:

ORIGINAL DESCRIPTION:
{description}

PDDL DOMAIN:
{domain.raw_text}

Focus on these aspects:
- Are action parameters properly typed?
- Do parameter names make sense?
- Are all necessary parameters included?
- Is the parameter syntax correct?
- Are variable names consistent?

Format: ISSUE: / SEVERITY: / SUGGESTION: / --- (repeat)"""
        
        response = self.llm.generate(prompt, self.system_prompt)
        return self._parse_investigation_response(response)

class EffectsAndPreconditionsInvestigator(InvestigatorAgent):
    """Investigator specializing in effects and preconditions"""
    
    def __init__(self, llm_provider: LLMInterface):
        super().__init__("EffectsAndPreconditionsInvestigator", "Effects and Preconditions", llm_provider)
        
    def investigate(self, description: str, domain: PDDLDomain) -> InvestigationReport:
        """Override with more specific prompt for effects and preconditions"""
        prompt = f"""Analyze this PDDL domain specifically for EFFECTS AND PRECONDITIONS issues:

ORIGINAL DESCRIPTION:
{description}

PDDL DOMAIN:
{domain.raw_text}

Focus on these aspects:
- Are preconditions logically necessary and sufficient?
- Do effects accurately represent the action outcomes?
- Are negative effects (deletions) properly specified?
- Is the logic consistent and complete?
- Are frame axioms properly handled?

Format: ISSUE: / SEVERITY: / SUGGESTION: / --- (repeat)"""
        
        response = self.llm.generate(prompt, self.system_prompt)
        return self._parse_investigation_response(response)

class TypingInvestigator(InvestigatorAgent):
    """Investigator specializing in typing issues"""
    
    def __init__(self, llm_provider: LLMInterface):
        super().__init__("TypingInvestigator", "Typing", llm_provider)
        
    def investigate(self, description: str, domain: PDDLDomain) -> InvestigationReport:
        """Override with more specific prompt for typing"""
        prompt = f"""Analyze this PDDL domain specifically for TYPING issues:

ORIGINAL DESCRIPTION:
{description}

PDDL DOMAIN:
{domain.raw_text}

Focus on these aspects:
- Are all types properly defined?
- Is the type hierarchy logical and complete?
- Are predicates using appropriate types?
- Are action parameters consistently typed?
- Are there missing or unnecessary types?

Format: ISSUE: / SEVERITY: / SUGGESTION: / --- (repeat)"""
        
        response = self.llm.generate(prompt, self.system_prompt)
        return self._parse_investigation_response(response)

class Combinator:
    """Module that combines investigator reports into feedback prompt"""
    
    @staticmethod
    def combine_reports(reports: List[InvestigationReport], original_description: str) -> str:
        """Combine investigation reports into a feedback prompt"""
        logger.info("Combining investigation reports")

        feedback_parts = [original_description,
            "Issues identified during domain analysis:",
            ""
        ]

        for report in reports:
            if report.issues_found:
                feedback_parts.append(f"=== {report.investigator_type} Issues ===")
                for i, issue in enumerate(report.issues_found):
                    severity = report.severity_scores[i] if i < len(report.severity_scores) else 0.5
                    feedback_parts.append(f"- {issue} (Severity: {severity:.2f})")
                
                feedback_parts.append("\nSuggestions:")
                for suggestion in report.suggestions:
                    feedback_parts.append(f"- {suggestion}")
                feedback_parts.append("")

        if len(feedback_parts) == 3:  # Only original description and header
            feedback_parts.append("No issues found by any investigator.")

        feedback_parts.append("=== End of Issues ===")
        feedback_parts.append("Please use this feedback to improve the PDDL domain. Focus on addressing all identified issues and suggestions.")

        logger.debug("Finished combining reports, here's the feedback prompt:")
        logger.debug("\n".join(feedback_parts))

        return "\n".join(feedback_parts)

class PDDLGeneratorSystem:
    """Main orchestrator for the multi-agent PDDL generation system"""
    
    def __init__(self, 
                 llm_configs: Dict[str, LLMConfig] = None,
                 success_threshold: float = 0.8, 
                 max_iterations: int = DEFAULT_MAX_ITERATIONS):
        self.success_threshold = success_threshold
        self.max_iterations = max_iterations
        
        # Set up default LLM configurations if none provided
        if llm_configs is None:
            llm_configs = self._get_default_configs()
        
        # Create LLM providers
        self.llm_providers = {}
        for role, config in llm_configs.items():
            self.llm_providers[role] = LLMFactory.create_provider(config)
        
        # Initialize agents with their respective LLM providers
        self.formalizer = FormalizerAgent(
            self.llm_providers.get('formalizer', self.llm_providers['default'])
        )
        self.critic = SuccessRateCritic(
            self.llm_providers.get('critic', self.llm_providers['default']), 
            success_threshold
        )
        self.investigators = [
            ActionSignatureInvestigator(self.llm_providers.get('investigator', self.llm_providers['default'])),
            EffectsAndPreconditionsInvestigator(self.llm_providers.get('investigator', self.llm_providers['default'])),
            TypingInvestigator(self.llm_providers.get('investigator', self.llm_providers['default']))
        ]
        self.combinator = Combinator()
    
    def _get_default_configs(self) -> Dict[str, LLMConfig]:
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
    
    @classmethod
    def create_with_single_provider(cls, provider_config: LLMConfig, **kwargs) -> 'PDDLGeneratorSystem':
        """Create system using single LLM provider for all agents"""
        configs = {
            'default': provider_config,
            'formalizer': provider_config,
            'critic': provider_config,
            'investigator': provider_config
        }
        return cls(llm_configs=configs, **kwargs)
    
    @classmethod
    def create_with_mixed_providers(cls,
                                    default_config: LLMConfig,
                                    formalizer_config: LLMConfig = None,
                                    critic_config: LLMConfig = None,
                                    investigator_config: LLMConfig = None,
                                    **kwargs) -> 'PDDLGeneratorSystem':
        """Create system with different providers for different roles"""
        configs = {
            'default': default_config,
            'formalizer': formalizer_config or default_config,
            'critic': critic_config or default_config,
            'investigator': investigator_config or default_config
        }
        return cls(llm_configs=configs, **kwargs)
    
    def generate_pddl(self, natural_language_description: str) -> str:
        """Main method to generate PDDL from natural language description"""
        logger.info("Starting PDDL generation process")
        
        current_domain: PDDLDomain | None = None
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Formalization step
            if iteration == 1:
                current_domain = self.formalizer.initial_formalization(natural_language_description)
            else:
                # Get feedback from previous iteration
                reports = [inv.investigate(natural_language_description, current_domain) 
                          for inv in self.investigators]
                feedback_prompt = self.combinator.combine_reports(reports, natural_language_description)
                current_domain = self.formalizer.re_formalization(natural_language_description, feedback_prompt)
            
            # Evaluation step
            evaluation = self.critic.evaluate(natural_language_description, current_domain)
            
            logger.info(f"Success rate: {evaluation.success_rate:.2f}")
            logger.info(f"Passes threshold: {evaluation.passes_threshold}")
            
            if evaluation.passes_threshold:
                logger.info("Domain passed evaluation - returning final PDDL")
                return current_domain.raw_text
            else:
                logger.info(f"Domain failed evaluation: {evaluation.reasoning}")
        
        logger.info(f"Maximum iterations ({self.max_iterations}) reached, LLM as a judge did not pass the threshold.")
        logger.info("Returning last generated PDDL domain.")
        return current_domain.raw_text if current_domain else ""



parser = argparse.ArgumentParser(
    description="Choose an LLM configuration for your multi-agent system.",
    epilog="""
Default configurations:
  uniform : All agents use gpt-4.
  mixed   : Formalizer uses gpt-4 (low temperature), Critic uses Claude (0.4 temp), 
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
    required=True,
    help=f"Provide a path to a custom JSON config file."
)
parser.add_argument(
    '--mixed',
    action='store_true',
    help="Use mixed provider configuration ."
)


def create_system_from_config_file(config_path: str) -> PDDLGeneratorSystem:
    """Create PDDL generator system from JSON configuration file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    llm_configs = {}
    for role, config_dict in config_data.get('llm_configs', {}).items():
        llm_configs[role] = LLMFactory.create_from_dict(config_dict)

    return PDDLGeneratorSystem.create_with_mixed_providers(default_config=llm_configs.get('default', None),
                                                            formalizer_config=llm_configs.get('formalizer', None),
                                                            critic_config=llm_configs.get('critic', None),
                                                            investigator_config=llm_configs.get('investigator', None),
                                                            success_threshold=config_data.get('success_threshold', DEFAULT_SUCCESS_THRESHOLD),
                                                            max_iterations=config_data.get('max_iterations', DEFAULT_MAX_ITERATIONS))

# Example usage and configuration
if __name__ == "__main__":
    llm_system = None
    args = parser.parse_args()

    # if for some reason user tries to use both mixed and config, raise an error
    if args.mixed and args.config:
        raise ValueError("Cannot use both --mixed and --config. Choose one configuration method.")
    # if there are no arguments, uniform is default
    elif not args.config and not args.mixed:
        print("No configuration specified. Using default uniform configuration with gpt-4.")
        gpt4_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="your-openai-api-key",  # or set OPENAI_API_KEY env var
            temperature=0.7,
            max_tokens=2000
        )
        llm_system = PDDLGeneratorSystem.create_with_single_provider(
            gpt4_config,
            success_threshold=DEFAULT_SUCCESS_THRESHOLD,
            max_iterations=DEFAULT_MAX_ITERATIONS
        )
    # if mixed is specified, use mixed configuration
    elif args.mixed:
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
                api_key="your-openai-api-key",  # or set OPENAI_API_KEY env var
                temperature=0.7,
                max_tokens=2000
            )
        }
        llm_system = PDDLGeneratorSystem.create_with_mixed_providers(
            default_config=mixed_configs['default'],
            formalizer_config=mixed_configs['formalizer'],
            critic_config=mixed_configs['critic'],
            investigator_config=mixed_configs['investigator'],
            success_threshold=DEFAULT_SUCCESS_THRESHOLD,
            max_iterations=DEFAULT_MAX_ITERATIONS
        )

    elif args.config:
        # Load configuration from file
        config_path = args.config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        llm_system = create_system_from_config_file(config_path)

    # Example natural language description
    description = """
    This is a blocks world domain where we have blocks and a table. 
    Blocks can be stacked on top of each other or placed on the table. 
    The goal is to move blocks from one configuration to another.
    We need actions to pick up blocks, put them down, and stack them.
    A robot arm can only hold one block at a time.
    """
    
    # Choose which system to use (comment out others for testing)
    print("\n=== Generating PDDL with selected system ===")
    
    try:
        # Use one of the systems - change this line to test different providers
        result = llm_system.generate_pddl(description)
        print("Generated PDDL:")
        print(result)
        
    except Exception as e:
        print(f"Error generating PDDL: {e}")
        print("Make sure you have:")
        print("1. Required packages installed (pip install openai anthropic google-generativeai cohere requests)")
        print("2. API keys set in environment variables or config")
        print("3. For Ollama: local server running on http://localhost:11434")