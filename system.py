"""
Main system orchestrator for the multi-agent PDDL generation system
"""

import logging
from typing import Dict, Optional

from llm_providers import LLMInterface, LLMConfig, LLMFactory, get_default_configs
from agents import FormalizerAgent, SuccessRateCritic, ActionSignatureInvestigator, EffectsAndPreconditionsInvestigator, TypingInvestigator, Combinator
from pddl_models import PDDLDomain
from config import DEFAULT_MAX_ITERATIONS

logger = logging.getLogger(__name__)


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
            llm_configs = get_default_configs()
        
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