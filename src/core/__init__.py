from .logger import logger
from .json_handler import log_to_json, setup_json

setup_json()

__all__ = [
    "logger",
    "log_to_json",
]
