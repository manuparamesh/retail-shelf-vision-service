import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models

from app.config import METADATA_PATH, MODEL_DIR, MODEL_PATH, TEST_DIR, TRAIN_DIR, VAL_DIR
from app.utils import load_config, save_json, setup_logger
from pipelines.preprocess import get_eval_transforms, get_train_transforms


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return loss, accuracy


def main():
    logger = setup_logger("training")
    config = load_config()

    image_size = config["model"]["image_size"]
    batch_size = config["model"]["batch_size"]
    num_epochs = config["model"]["num_epochs"]
    learning_rate = config["model"]["learning_rate"]
    class_names = config["classes"]

    logger.info("Loading datasets")
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transforms(image_size),
    )
    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=get_eval_transforms(image_size),
    )
    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_eval_transforms(image_size),
    )

    logger.info("Detected classes from folders: %s", train_dataset.classes)

    if sorted(train_dataset.classes) != sorted(class_names):
        raise ValueError(
            f"Class mismatch. Config classes={class_names}, dataset classes={train_dataset.classes}"
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Initializing pretrained ResNet18")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        train_accuracy = correct / total if total > 0 else 0.0

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %s/%s | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch + 1,
            num_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info("Saved best model to %s", MODEL_PATH)

    logger.info("Loading best model for final test evaluation")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    metadata = {
        "model_name": config["model"]["name"],
        "model_version": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "trained_at_utc": datetime.utcnow().isoformat(),
        "image_size": image_size,
        "classes": train_dataset.classes,
        "metrics": {
            "best_val_accuracy": best_val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
        "device": str(device),
    }
    save_json(METADATA_PATH, metadata)

    logger.info("Training complete")
    logger.info("Test accuracy: %.4f", test_accuracy)
    logger.info("Metadata saved to %s", METADATA_PATH)


if __name__ == "__main__":
    main()