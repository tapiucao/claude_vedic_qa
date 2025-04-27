"""
Logging utilities for Vedic Knowledge AI.
Configures and provides logging functionality throughout the system.
"""
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, List

from ..config import LOG_LEVEL

def setup_logger(
    name: str = "vedic_knowledge_ai",
    log_file: str = "vedic_knowledge_ai.log",
    level: str = LOG_LEVEL,
    console: bool = True,
    max_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """Set up a logger with file and optional console output."""
    # Parse log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(name)

class LogCapture:
    """Context manager to capture logs for analysis."""
    
    def __init__(self, logger_name: str = None, level: int = logging.INFO):
        """Initialize the log capture."""
        self.logger_name = logger_name
        self.level = level
        self.captured_logs = []
        self.handler = None
    
    def __enter__(self):
        """Set up log capturing when entering context."""
        self.handler = CaptureHandler(self.captured_logs)
        self.handler.setLevel(self.level)
        
        # Add handler to the specified logger or root logger
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()
        
        logger.addHandler(self.handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context."""
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()
        
        if self.handler in logger.handlers:
            logger.removeHandler(self.handler)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get the captured logs."""
        return self.captured_logs

class CaptureHandler(logging.Handler):
    """Custom handler to capture logs in a list."""
    
    def __init__(self, captured_logs: List[Dict[str, Any]]):
        """Initialize with a list to store logs."""
        super().__init__()
        self.captured_logs = captured_logs
    
    def emit(self, record):
        """Process a log record by adding it to the captured list."""
        log_entry = {
            'timestamp': record.created,
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'pathname': record.pathname,
            'lineno': record.lineno,
            'exc_info': record.exc_info
        }
        self.captured_logs.append(log_entry)