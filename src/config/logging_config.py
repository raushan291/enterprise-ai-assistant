import logging
import os
from datetime import datetime


def get_logger(
    name: str, log_dir: str = "logs", level: int = logging.INFO
) -> logging.Logger:
    """
    Create and configure a logger that logs to both console and a file.

    Args:
        name (str): name of the logger (usually __name__ or module name)
        log_dir (str): directory where logs will be stored
        level (int): logging level (default: INFO)
    """
    os.makedirs(log_dir, exist_ok=True)

    # Log filename with timestamp
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers (important in notebooks or reruns)
    if logger.handlers:
        return logger

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Silence noisy external libraries
    # Presidio warnings (multi-language recognizer messages)
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

    return logger
