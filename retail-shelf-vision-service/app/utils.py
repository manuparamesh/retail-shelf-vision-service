import json
import logging
from logging.handlers import RotatingFileHandler

import yaml

from app.config import CONFIG_PATH, LOG_DIR


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logger(name: str = "retail_shelf_vision") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=1_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_json(path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)