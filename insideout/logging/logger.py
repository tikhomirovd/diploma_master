"""
Logging configuration with timestamp-based filenames.

Provides structured logging for experiments with automatic file naming.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    log_dir: Path,
    experiment_name: str,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Setup logger with both file and console handlers.
    
    Creates a log file with timestamp: {experiment_name}_{timestamp}.log
    
    Args:
        log_dir: Directory to store log files
        experiment_name: Name of the experiment (used in filename)
        level: Overall logger level
        console_level: Level for console output
        file_level: Level for file output
    
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Get or create logger
    logger = logging.getLogger("insideout")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler with detailed logging
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(file_level)
    
    # Console handler with less verbose output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    
    # Detailed formatter for file
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Simpler formatter for console
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    
    fh.setFormatter(file_formatter)
    ch.setFormatter(console_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


def get_logger() -> logging.Logger:
    """
    Get the existing insideout logger.
    
    Returns:
        Logger instance
    """
    return logging.getLogger("insideout")


class LoggingContext:
    """
    Context manager for temporarily adding context to log messages.
    
    Example:
        with LoggingContext(logger, "ERC Pipeline"):
            logger.info("Starting processing")  # Will include context prefix
    """
    
    def __init__(self, logger: logging.Logger, context: str):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            record.msg = f"[{self.context}] {record.msg}"
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_experiment_config(logger: logging.Logger, config: dict) -> None:
    """
    Log experiment configuration in a structured format.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("Experiment Configuration:")
    logger.info("-" * 80)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = "") -> None:
    """
    Log metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names to values
        prefix: Optional prefix for metric names
    """
    logger.info("=" * 80)
    logger.info(f"{prefix}Metrics:" if prefix else "Metrics:")
    logger.info("-" * 80)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")
    logger.info("=" * 80)
