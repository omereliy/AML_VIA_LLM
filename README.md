# Multi-Agent LLM System for PDDL Domain Generation

A multi-agent system that uses Large Language Models to generate PDDL (Planning Domain Definition Language) domains from natural language descriptions through iterative refinement and error detection.

## System Overview

The system converts textual descriptions of planning domains into formal PDDL domain files using a multi-agent approach:

- **Input**: Natural language description of a problem domain
- **Output**: Complete PDDL domain file with types, predicates, actions, and their specifications
- **Process**: Iterative refinement through specialized LLM agents with error detection and correction

## Architecture

### Agents
- **Formalizer**: Converts textual descriptions to PDDL format and re-formalizes based on feedback
- **Success Rate Critic**: Evaluates domain quality and acts as gatekeeper for iteration threshold
- **Investigators**: Three specialized agents for error detection:
  - Action Signature Investigator
  - Effects and Preconditions Investigator  
  - Typing Investigator
- **Combinator**: Aggregates investigator feedback for refinement prompts

### LLM Provider Support
- OpenAI GPT models
- Anthropic Claude
- Google Gemini
- Cohere
- Environment-based API key management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AML_VIA_LLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables for API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Command Line Interface

Run the system with default uniform configuration (all agents use GPT-4):
```bash
python cli.py --input "your domain description"
```

Use mixed configuration (different models for different agents):
```bash
python cli.py --mixed --input "your domain description"
```

Use custom configuration file:
```bash
python cli.py --config config.json --input "your domain description"
```

### Configuration Options

- `--success-threshold`: Minimum success rate to stop iterations (default: 0.8)
- `--max-iterations`: Maximum refinement iterations (default: 10)
- `--output`: Output file path for generated PDDL

### Custom Configuration Format

Create a JSON configuration file with agent-specific LLM settings:
```json
{
  "formalizer": {
    "model": "gpt-4",
    "temperature": 0.2,
    "max_tokens": 2048
  },
  "critic": {
    "model": "claude-3-sonnet",
    "temperature": 0.4
  },
  "investigator": {
    "model": "gemini-1.5-pro",
    "temperature": 0.9
  }
}
```

## Project Structure

```
AML_VIA_LLM/
├── agents.py              # Agent implementations
├── cli.py                 # Command-line interface
├── config.py              # Configuration constants
├── llm_providers.py       # LLM provider abstractions
├── pddl_generator.py      # Core PDDL generation logic
├── pddl_models.py         # PDDL data models
├── system.py              # Main system orchestrator
├── bug_detection/         # Specialized investigator modules
├── requirements.txt       # Python dependencies
└── Misc/                  # Documentation and diagrams
```

## Requirements

All dependencies are pinned for stability. Key requirements include:
- LangChain framework (0.2.9+) for agent orchestration
- LLM provider packages (OpenAI, Anthropic, Google, Cohere)
- Pydantic for data validation
- Python-dotenv for environment management

See `requirements.txt` for complete dependency list with versions.
