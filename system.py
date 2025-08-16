"""
Main system orchestrator for the multi-agent PDDL generation system
"""

import logging
from typing import Dict

from agents import FormalizerAgent, SuccessRateCritic, ActionSignatureInvestigator, EffectsAndPreconditionsInvestigator, \
    TypingInvestigator, Combinator
from config import DEFAULT_MAX_ITERATIONS
from llm_providers import LLMConfig, LLMFactory, get_default_configs
from pddl_models import PDDLDomain
from experiment_logging import get_experiment_logger

logger = logging.getLogger(__name__)


class PDDLGeneratorSystem:
    """Main orchestrator for the multi-agent PDDL generation system"""
    
    def __init__(self, 
                 llm_configs: Dict[str, LLMConfig] = None,
                 success_threshold: float = 0.8, 
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 ablation_mode: str = "full"):
        self.success_threshold = success_threshold
        self.max_iterations = max_iterations
        self.ablation_mode = ablation_mode
        
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
        logger.info(f"Starting PDDL generation process in {self.ablation_mode} mode")
        
        # Route to appropriate generation method based on ablation mode
        if self.ablation_mode == "baseline":
            return self._baseline_generation(natural_language_description)
        elif self.ablation_mode == "iterative":
            return self._iterative_only_generation(natural_language_description)
        elif self.ablation_mode == "investigators":
            return self._investigators_only_generation(natural_language_description)
        else:  # "full" - default behavior
            return self._full_generation(natural_language_description)
    
    def _full_generation(self, natural_language_description: str) -> str:
        """Original full multi-agent generation method"""
        logger.info("Starting PDDL generation process")
        
        # Get experiment logger for conversation logging
        exp_logger = get_experiment_logger()

        try:
            system_logger = exp_logger.get_agent_logger("System")
        except KeyError:
            system_logger = logging.Logger('System')

        if exp_logger:
            system_logger = exp_logger.get_agent_logger("System")
            system_logger.info("Starting PDDL generation process with multi-agent refinement")
        
        current_domain: PDDLDomain | None = None
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Log iteration start right before doing the work
            if exp_logger:
                exp_logger.log_iteration_start(iteration)
                system_logger.info(f"Beginning iteration {iteration} of {self.max_iterations}")

            # Formalization step
            if iteration == 1:
                if exp_logger:
                    system_logger.info("Requesting initial PDDL formalization from Formalizer")
                current_domain = self.formalizer.initial_formalization(natural_language_description)
            else:
                # Get feedback from previous iteration
                if exp_logger:
                    system_logger.info("Requesting investigation reports from specialist agents")
                
                reports = [inv.investigate(natural_language_description, current_domain) 
                          for inv in self.investigators]
                feedback_prompt = self.combinator.combine_reports(reports, natural_language_description)
                
                if exp_logger:
                    combinator_logger = exp_logger.get_agent_logger("Combinator")
                    combinator_logger.info("Combined investigation reports into feedback:")
                    combinator_logger.info(feedback_prompt)
                    system_logger.info("Requesting refined PDDL from Formalizer based on feedback")
                
                current_domain = self.formalizer.re_formalization(natural_language_description, feedback_prompt)
            
            # Save iteration PDDL
            if exp_logger and current_domain:
                exp_logger.save_iteration_pddl(iteration, current_domain.raw_text)
                system_logger.info(f"Saved iteration {iteration} PDDL to iteration_{iteration}.pddl")
            
            # Evaluation step
            if exp_logger:
                system_logger.info("Requesting evaluation from Critic")
            
            evaluation = self.critic.evaluate(natural_language_description, current_domain)
            
            logger.info(f"Success rate: {evaluation.success_rate:.2f}")
            logger.info(f"Passes threshold: {evaluation.passes_threshold}")
            
            if evaluation.passes_threshold:
                logger.info("Domain passed evaluation - returning final PDDL")
                if exp_logger:
                    system_logger.info(f"SUCCESS! Domain achieved {evaluation.success_rate:.2f} success rate (threshold: {self.success_threshold})")
                    system_logger.info("Experiment completed successfully")
                return current_domain.raw_text
            else:
                logger.info(f"Domain failed evaluation: {evaluation.reasoning}")
                if exp_logger:
                    system_logger.info(f"Domain did not meet threshold ({evaluation.success_rate:.2f} < {self.success_threshold})")
                    if iteration < self.max_iterations:
                        system_logger.info("Proceeding to next iteration for refinement")
        
        logger.info(f"Maximum iterations ({self.max_iterations}) reached, LLM as a judge did not pass the threshold.")
        logger.info("Returning last generated PDDL domain.")
        
        if exp_logger:
            system_logger.info(f"Maximum iterations ({self.max_iterations}) reached without achieving success threshold")
            system_logger.info("Returning best attempt from final iteration")
        
        return current_domain.raw_text if current_domain else ""

    def _baseline_generation(self, natural_language_description: str) -> str:
        """Baseline: Single-shot generation without iterations or investigators"""
        logger.info("Running baseline generation (single FormalizerAgent call)")
        
        exp_logger = get_experiment_logger()
        if exp_logger:
            system_logger = exp_logger.get_agent_logger("System")
            system_logger.info("ABLATION MODE: Baseline - Single-shot generation only")
            system_logger.info("Requesting initial PDDL formalization from Formalizer")
        
        # Single formalization call
        current_domain = self.formalizer.initial_formalization(natural_language_description)
        
        # Save as iteration 1 for consistency
        if exp_logger and current_domain:
            exp_logger.log_iteration_start(1)
            exp_logger.save_iteration_pddl(1, current_domain.raw_text)
        
        # Evaluate for metrics but don't use result for iteration
        evaluation = self.critic.evaluate(natural_language_description, current_domain)
        logger.info(f"Baseline success rate: {evaluation.success_rate:.2f}")
        
        if exp_logger:
            system_logger.info(f"Baseline generation completed with success rate: {evaluation.success_rate:.2f}")
        
        return current_domain.raw_text

    def _iterative_only_generation(self, natural_language_description: str) -> str:
        """Iterative-only: Multiple iterations with critic feedback but no investigators"""
        logger.info("Running iterative-only generation (critic feedback without investigators)")
        
        exp_logger = get_experiment_logger()
        if exp_logger:
            system_logger = exp_logger.get_agent_logger("System")
            system_logger.info("ABLATION MODE: Iterative-only - Using critic feedback without investigators")
        
        current_domain: PDDLDomain | None = None
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            if exp_logger:
                exp_logger.log_iteration_start(iteration)
                system_logger.info(f"Beginning iteration {iteration} of {self.max_iterations}")
            
            # Formalization step
            if iteration == 1:
                if exp_logger:
                    system_logger.info("Requesting initial PDDL formalization from Formalizer")
                current_domain = self.formalizer.initial_formalization(natural_language_description)
            else:
                # Use critic reasoning as feedback instead of investigators
                prev_evaluation = self.critic.evaluate(natural_language_description, current_domain)
                feedback_prompt = f"Previous iteration feedback: {prev_evaluation.reasoning}\n\nSpecific concerns to address:\n"
                for concern in prev_evaluation.specific_concerns:
                    feedback_prompt += f"- {concern}\n"
                feedback_prompt += "\nPlease generate an improved PDDL domain that addresses these issues."
                
                if exp_logger:
                    system_logger.info("Using critic evaluation feedback for refinement")
                
                current_domain = self.formalizer.re_formalization(natural_language_description, feedback_prompt)
            
            # Save iteration PDDL
            if exp_logger and current_domain:
                exp_logger.save_iteration_pddl(iteration, current_domain.raw_text)
                system_logger.info(f"Saved iteration {iteration} PDDL to iteration_{iteration}.pddl")
            
            # Evaluation step
            if exp_logger:
                system_logger.info("Requesting evaluation from Critic")
            
            evaluation = self.critic.evaluate(natural_language_description, current_domain)
            
            logger.info(f"Success rate: {evaluation.success_rate:.2f}")
            logger.info(f"Passes threshold: {evaluation.passes_threshold}")
            
            if evaluation.passes_threshold:
                logger.info("Domain passed evaluation - returning final PDDL")
                if exp_logger:
                    system_logger.info(f"SUCCESS! Domain achieved {evaluation.success_rate:.2f} success rate (threshold: {self.success_threshold})")
                return current_domain.raw_text
            else:
                logger.info(f"Domain failed evaluation: {evaluation.reasoning}")
                if exp_logger:
                    system_logger.info(f"Domain did not meet threshold ({evaluation.success_rate:.2f} < {self.success_threshold})")
                    if iteration < self.max_iterations:
                        system_logger.info("Proceeding to next iteration for refinement")
        
        logger.info(f"Maximum iterations ({self.max_iterations}) reached")
        if exp_logger:
            system_logger.info(f"Maximum iterations reached without achieving success threshold")
        
        return current_domain.raw_text if current_domain else ""

    def _investigators_only_generation(self, natural_language_description: str) -> str:
        """Investigators-only: Single iteration with investigator feedback"""
        logger.info("Running investigators-only generation (single iteration with investigator feedback)")
        
        exp_logger = get_experiment_logger()
        if exp_logger:
            system_logger = exp_logger.get_agent_logger("System")
            system_logger.info("ABLATION MODE: Investigators-only - Single iteration with specialized feedback")
        
        # Initial formalization
        if exp_logger:
            exp_logger.log_iteration_start(1)
            system_logger.info("Requesting initial PDDL formalization from Formalizer")
        
        current_domain = self.formalizer.initial_formalization(natural_language_description)
        
        # Save initial domain
        if exp_logger and current_domain:
            exp_logger.save_iteration_pddl(1, current_domain.raw_text)
            system_logger.info("Saved iteration 1 PDDL to iteration_1.pddl")
        
        # Evaluate initial domain
        initial_evaluation = self.critic.evaluate(natural_language_description, current_domain)
        logger.info(f"Initial success rate: {initial_evaluation.success_rate:.2f}")
        
        # Get investigator feedback
        if exp_logger:
            system_logger.info("Requesting investigation reports from specialist agents")
        
        reports = [inv.investigate(natural_language_description, current_domain) 
                  for inv in self.investigators]
        feedback_prompt = self.combinator.combine_reports(reports, natural_language_description)
        
        if exp_logger:
            combinator_logger = exp_logger.get_agent_logger("Combinator")
            combinator_logger.info("Combined investigation reports into feedback:")
            combinator_logger.info(feedback_prompt)
            system_logger.info("Requesting refined PDDL from Formalizer based on feedback")
        
        # Single refinement iteration
        refined_domain = self.formalizer.re_formalization(natural_language_description, feedback_prompt)
        
        # Save refined domain as iteration 2
        if exp_logger and refined_domain:
            exp_logger.log_iteration_start(2)
            exp_logger.save_iteration_pddl(2, refined_domain.raw_text)
            system_logger.info("Saved iteration 2 PDDL to iteration_2.pddl")
        
        # Final evaluation
        final_evaluation = self.critic.evaluate(natural_language_description, refined_domain)
        logger.info(f"Final success rate: {final_evaluation.success_rate:.2f}")
        
        if exp_logger:
            system_logger.info(f"Investigators-only generation completed")
            system_logger.info(f"Initial success rate: {initial_evaluation.success_rate:.2f}")
            system_logger.info(f"Final success rate: {final_evaluation.success_rate:.2f}")
            system_logger.info(f"Improvement: {final_evaluation.success_rate - initial_evaluation.success_rate:.2f}")
        
        return refined_domain.raw_text