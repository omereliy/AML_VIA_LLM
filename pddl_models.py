"""
PDDL domain models and parsing utilities
"""

import re
from dataclasses import dataclass
from typing import List


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


def extract_section(pddl_text: str, section_name: str) -> List[str]:
    """Extract items from a PDDL section"""
    pattern = rf'\(:{section_name}([^)]*)\)'
    match = re.search(pattern, pddl_text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return [item.strip() for item in content.split() if item.strip()]
    return []


def extract_actions(pddl_text: str) -> List[str]:
    """Extract action definitions"""
    action_pattern = r'\(:action[^)]*(?:\([^)]*\)[^)]*)*\)'
    matches = re.findall(action_pattern, pddl_text, re.DOTALL)
    return matches


def parse_pddl_domain(pddl_text: str) -> PDDLDomain:
    """Parse PDDL text into structured domain object"""
    # Simple regex-based parsing (could be enhanced with proper PDDL parser)
    domain_name_match = re.search(r'\(define \(domain ([^)]+)\)', pddl_text)
    domain_name = domain_name_match.group(1) if domain_name_match else "unknown"

    types = extract_section(pddl_text, "types")
    constants = extract_section(pddl_text, "constants")
    predicates = extract_section(pddl_text, "predicates")
    functions = extract_section(pddl_text, "functions")
    actions = extract_actions(pddl_text)

    return PDDLDomain(
        domain_name=domain_name,
        types=types,
        constants=constants,
        predicates=predicates,
        functions=functions,
        actions=actions,
        raw_text=pddl_text
    )