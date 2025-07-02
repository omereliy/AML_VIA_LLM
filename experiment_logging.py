"""
Experiment logging configuration for PDDL generation conversations
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class ExperimentLogger:
    """Manages experiment logging with conversation-style formatting"""
    
    def __init__(self, experiment_name: str = None, output_dir: str = "experiments"):
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create individual experiment folder
        self.experiment_dir = self.base_output_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create log file path in experiment folder
        self.log_file = self.experiment_dir / "conversation.log"
        
        # Create shared file handler for all loggers to maintain chronological order
        self.shared_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        self.shared_handler.setLevel(logging.INFO)
        
        # Create clean formatter for file (no colors)
        formatter = ConversationFormatter()
        self.shared_handler.setFormatter(formatter)
        
        # Enable auto-flush for immediate writes
        self.shared_handler.stream.reconfigure(line_buffering=True)
        
        # Optional: Create console handler with colors for terminal output
        self.console_handler = None
        self.enable_console_output = False
        
        # Set up conversation logger
        self.conversation_logger = self._setup_conversation_logger()
        
        # Set up individual agent loggers
        self.agent_loggers = {}
        
    def _setup_conversation_logger(self) -> logging.Logger:
        """Set up the main conversation logger"""
        logger = logging.getLogger(f"experiment.{self.experiment_name}.conversation")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Use shared handler
        logger.addHandler(self.shared_handler)
        logger.propagate = False  # Don't propagate to root logger
        
        return logger
    
    def get_agent_logger(self, agent_name: str) -> logging.Logger:
        """Get or create a logger for a specific agent"""
        if agent_name not in self.agent_loggers:
            logger = logging.getLogger(f"experiment.{self.experiment_name}.{agent_name}")
            logger.setLevel(logging.INFO)
            
            # Remove existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Use the same shared handler for chronological ordering
            logger.addHandler(self.shared_handler)
            
            # Add console handler if enabled
            if self.console_handler:
                logger.addHandler(self.console_handler)
            
            logger.propagate = False
            
            # Wrap logger to auto-flush after each message
            original_info = logger.info
            def auto_flush_info(message):
                original_info(message)
                self.shared_handler.flush()
                if self.console_handler:
                    self.console_handler.flush()
            logger.info = auto_flush_info
            
            self.agent_loggers[agent_name] = logger
        
        return self.agent_loggers[agent_name]
    
    def save_iteration_pddl(self, iteration: int, pddl_content: str):
        """Save PDDL content for a specific iteration"""
        iteration_file = self.experiment_dir / f"iteration_{iteration}.pddl"
        with open(iteration_file, 'w', encoding='utf-8') as f:
            f.write(pddl_content)
    
    def log_experiment_start(self, description: str, config: dict):
        """Log the start of an experiment"""
        self.conversation_logger.info("=== PDDL GENERATION EXPERIMENT STARTED ===")
        self.conversation_logger.info(f"Experiment: {self.experiment_name}")
        self.conversation_logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.conversation_logger.info("")
        self.conversation_logger.info("CONFIGURATION:")
        for key, value in config.items():
            self.conversation_logger.info(f"  {key}: {value}")
        self.conversation_logger.info("")
        self.conversation_logger.info("DOMAIN DESCRIPTION:")
        self.conversation_logger.info(description)
        self.conversation_logger.info("")
        self.conversation_logger.info("=== AGENT CONVERSATION BEGINS ===")
        self.conversation_logger.info("")
    
    def log_iteration_start(self, iteration: int):
        """Log the start of a new iteration"""
        self.conversation_logger.info(f"--- ITERATION {iteration} ---")
        self.conversation_logger.info("")
    
    def log_experiment_end(self, final_result: str, success: bool):
        """Log the end of an experiment"""
        # Save final PDDL as well
        final_file = self.experiment_dir / "final_domain.pddl"
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(final_result)
        
        self.conversation_logger.info("")
        self.conversation_logger.info("=== EXPERIMENT COMPLETED ===")
        self.conversation_logger.info(f"Success: {success}")
        self.conversation_logger.info("")
        self.conversation_logger.info("FINAL PDDL DOMAIN:")
        self.conversation_logger.info(final_result)
        self.conversation_logger.info("")
        self.conversation_logger.info("=== END OF CONVERSATION ===")
    
    def enable_console_colors(self):
        """Enable colored output to console (optional)"""
        if not self.console_handler:
            import sys
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setLevel(logging.INFO)
            colored_formatter = ColoredConversationFormatter()
            self.console_handler.setFormatter(colored_formatter)
            self.enable_console_output = True
            
            # Add console handler to all existing loggers
            self.conversation_logger.addHandler(self.console_handler)
            for logger in self.agent_loggers.values():
                logger.addHandler(self.console_handler)
    
    def close(self):
        """Close all loggers and handlers"""
        # Close the shared handler once
        if self.shared_handler:
            self.shared_handler.close()
        
        # Close console handler if it exists
        if self.console_handler:
            self.console_handler.close()
        
        # Remove handlers from all loggers
        for logger in [self.conversation_logger] + list(self.agent_loggers.values()):
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)


class ConversationFormatter(logging.Formatter):
    """Custom formatter for conversation-style logging (clean, no colors)"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Extract agent name from logger name
        logger_parts = record.name.split('.')
        if len(logger_parts) >= 3 and logger_parts[-1] != 'conversation':
            agent_name = logger_parts[-1]
            # Format as clean conversation without colors
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            formatted = f"[{timestamp}] {agent_name}: {record.getMessage()}"
        else:
            # For system messages, just return the message
            formatted = record.getMessage()
        
        return formatted


class ColoredConversationFormatter(logging.Formatter):
    """Custom formatter for conversation-style logging with colors (terminal only)"""
    
    # ANSI color codes for different agents
    COLORS = {
        'System': '\033[96m',        # Cyan
        'Formalizer': '\033[92m',    # Green  
        'SuccessRateCritic': '\033[93m',  # Yellow
        'ActionSignatureInvestigator': '\033[94m',  # Blue
        'EffectsAndPreconditionsInvestigator': '\033[95m',  # Magenta
        'TypingInvestigator': '\033[91m',  # Red
        'Combinator': '\033[97m',    # White
    }
    RESET = '\033[0m'  # Reset color
    
    def format(self, record: logging.LogRecord) -> str:
        # Extract agent name from logger name
        logger_parts = record.name.split('.')
        if len(logger_parts) >= 3 and logger_parts[-1] != 'conversation':
            agent_name = logger_parts[-1]
            # Get color for this agent
            color = self.COLORS.get(agent_name, '\033[90m')  # Default to gray
            # Format as conversation with color
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            formatted = f"{color}[{timestamp}] {agent_name}: {record.getMessage()}{self.RESET}"
        else:
            # For system messages, just return the message (no color)
            formatted = record.getMessage()
        
        return formatted


# Global experiment logger instance
_current_experiment_logger: Optional[ExperimentLogger] = None


def setup_experiment_logging(experiment_name: str = None, output_dir: str = "experiments") -> ExperimentLogger:
    """Set up experiment logging and return the logger instance"""
    global _current_experiment_logger
    
    if _current_experiment_logger:
        _current_experiment_logger.close()
    
    _current_experiment_logger = ExperimentLogger(experiment_name, output_dir)
    return _current_experiment_logger


def get_experiment_logger() -> Optional[ExperimentLogger]:
    """Get the current experiment logger"""
    return _current_experiment_logger


def cleanup_experiment_logging():
    """Clean up experiment logging"""
    global _current_experiment_logger
    if _current_experiment_logger:
        _current_experiment_logger.close()
        _current_experiment_logger = None