import os
import random
import time
import tempfile
import logging
from typing import Any, Dict
import numpy as np
import torch
import yaml

# Ensure matplotlib uses a writable temp cache directory to avoid Windows permission warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))


def setup_logger(name: str = "blogs_classification", level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """
    Enforces strict reproducibility across Python, NumPy, PyTorch (CPU and GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detects and returns torch.device('cuda' or 'cpu')."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads configuration parameters from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


class Timer:
    """High-precision context manager and utility for elapsed wall-clock time measurement."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed

