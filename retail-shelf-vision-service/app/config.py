from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "configs" / "training_config.yaml"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "shelf_classifier.pth"
METADATA_PATH = MODEL_DIR / "metadata.json"

DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "raw" / "train"
VAL_DIR = DATA_DIR / "raw" / "val"
TEST_DIR = DATA_DIR / "raw" / "test"

LOG_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"