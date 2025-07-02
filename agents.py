"""
Agent classes for PDDL generation system
"""

import logging
from abc import ABC
from typing import List

from llm_providers import LLMInterface
from pddl_models import PDDLDomain, InvestigationReport, SuccessRateEvaluation, parse_pddl_domain
from experiment_logging import get_experiment_logger

logger = logging.getLogger(__name__)


class LLMAgent(ABC):
    """Base class for all LLM agents"""
    
    def __init__(self, name: str, llm_provider: LLMInterface):
        self.name = name
        self.llm = llm_provider


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
        
        # Log to experiment conversation
        exp_logger = get_experiment_logger()
        if exp_logger:
            agent_logger = exp_logger.get_agent_logger(self.name)
            agent_logger.info("I'll convert this natural language description into an initial PDDL domain.")
            agent_logger.info(f"Working with description: '{natural_language_description.strip()}'")
        
        prompt = f"""Convert the following natural language description into a PDDL domain:

{natural_language_description}

Generate a complete PDDL domain definition:"""
        
        pddl_text = self.llm.generate(prompt, self.system_prompt)
        logger.debug(f"========================\n[{self.name}] Generated PDDL text after initial formalization:\n {pddl_text}\n========================")
        
        if exp_logger:
            agent_logger.info("Here's my initial PDDL formalization:")
            agent_logger.info(f"\n{pddl_text}")
        
        return parse_pddl_domain(pddl_text)
    
    def re_formalization(self, original_description: str, feedback_prompt: str) -> PDDLDomain:
        """Re-formalize based on investigator feedback"""
        logger.info(f"[{self.name}] Starting re-formalization with feedback")
        
        # Log to experiment conversation
        exp_logger = get_experiment_logger()
        if exp_logger:
            agent_logger = exp_logger.get_agent_logger(self.name)
            agent_logger.info("I've received feedback from the investigation team. Let me refine the PDDL domain.")
            agent_logger.info("Analyzing the issues and incorporating improvements...")
        
        prompt = f"""Original Description: {original_description}

Issues Found and Feedback:
{feedback_prompt}

Please generate an improved PDDL domain that addresses all the identified issues while maintaining the core functionality described in the original description:"""
        
        pddl_text = self.llm.generate(prompt, self.system_prompt)
        logger.debug(f"========================\n[{self.name}] Generated PDDL text after re-formalization:\n {pddl_text}\n========================")
        
        if exp_logger:
            agent_logger.info("Here's my refined PDDL domain addressing the feedback:")
            agent_logger.info(f"\n{pddl_text}")
        
        return parse_pddl_domain(pddl_text)


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
        
        # Log to experiment conversation
        exp_logger = get_experiment_logger()
        if exp_logger:
            agent_logger = exp_logger.get_agent_logger(self.name)
            agent_logger.info("I'm evaluating how well this PDDL domain matches the original description.")
            agent_logger.info("Analyzing completeness, correctness, and appropriateness...")
        
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
        evaluation = self._parse_evaluation_response(response)
        
        if exp_logger:
            agent_logger.info(f"My evaluation: SUCCESS_RATE = {evaluation.success_rate:.2f}")
            agent_logger.info(f"REASONING: {evaluation.reasoning}")
            if evaluation.specific_concerns:
                agent_logger.info("CONCERNS:")
                for concern in evaluation.specific_concerns:
                    agent_logger.info(f"  - {concern}")
            agent_logger.info(f"Domain {'PASSES' if evaluation.passes_threshold else 'FAILS'} the success threshold ({self.threshold})")
        
        return evaluation
    
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
        except Exception as err:
            logger.error(f"Failed to parse evaluation response: {err}")
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
        
        # Log to experiment conversation
        exp_logger = get_experiment_logger()
        if exp_logger:
            agent_logger = exp_logger.get_agent_logger(self.name)
            agent_logger.info(f"I'm investigating the PDDL domain for issues related to {self.focus_area}.")
        
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
        logger.debug(f"========================\n[{self.name}] Investigation Response:\n {response}\n========================")
        
        report = self._parse_investigation_response(response)
        
        if exp_logger:
            if report.issues_found:
                agent_logger.info(f"I found {len(report.issues_found)} issues in {self.focus_area}:")
                for i, issue in enumerate(report.issues_found):
                    severity = report.severity_scores[i] if i < len(report.severity_scores) else 0.5
                    suggestion = report.suggestions[i] if i < len(report.suggestions) else "No suggestion provided"
                    agent_logger.info(f"  Issue {i+1} (severity {severity:.2f}): {issue}")
                    agent_logger.info(f"    Suggestion: {suggestion}")
            else:
                agent_logger.info(f"No issues found in {self.focus_area}. The domain looks good from my perspective.")
        
        return report
    
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
        logger.debug(f"========================\n[{self.name}] Investigation Response:\n {response}\n========================")
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
        logger.debug(f"========================\n[{self.name}] Investigation Response:\n {response}\n========================")
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
        logger.debug(f"========================\n[{self.name}] Investigation Response:\n {response}\n========================")
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