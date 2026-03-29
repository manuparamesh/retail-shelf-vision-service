import json

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models
import torch.nn as nn

from app.config import METADATA_PATH, MODEL_PATH
from app.utils import load_config
from pipelines.preprocess import get_eval_transforms


class ShelfVisionPredictor:
    def __init__(self):
        config = load_config()

        self.class_names = config["classes"]
        self.image_size = config["model"]["image_size"]

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.class_names))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transforms = get_eval_transforms(self.image_size)

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_label = self.class_names[predicted_idx.item()]

        return {
            "predicted_label": predicted_label,
            "confidence": round(float(confidence.item()), 4),
            "model_name": self.metadata.get("model_name", "unknown_model"),
            "model_version": self.metadata.get("model_version", "unknown_version"),
        }