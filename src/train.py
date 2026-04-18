import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.eurosat_opencv import EuroSATOpenCV
from src.data.transforms import TrainTransform, ValTransform
from src.models.vit_lora import ViTLoRAClassifier
from src.utils.io import ensure_dir, load_yaml
from src.utils.metrics import classification_metrics
from src.utils.plots import save_confusion_matrix, save_training_curves
from src.utils.scipy_utils import compute_class_weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    all_preds = []

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    metrics = classification_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["y_true"] = all_labels.numpy()
    metrics["y_pred"] = all_preds.numpy()
    return metrics


def append_history_csv(csv_path: str, epoch: int, train_loss: float, val_metrics: dict) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])
        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['accuracy']:.6f}",
                f"{val_metrics['macro_f1']:.6f}",
            ]
        )


def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])

    device_name = cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    selected_classes = cfg["data"].get("selected_classes", [])
    selected_classes = selected_classes if selected_classes else None

    train_ds = EuroSATOpenCV(
        root_dir=cfg["data"]["root_dir"],
        split="train",
        image_size=cfg["data"]["image_size"],
        transform=TrainTransform(),
        selected_classes=selected_classes,
    )
    val_ds = EuroSATOpenCV(
        root_dir=cfg["data"]["root_dir"],
        split="test",
        image_size=cfg["data"]["image_size"],
        transform=ValTransform(),
        selected_classes=selected_classes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=False,
    )

    model = ViTLoRAClassifier(
        num_classes=len(train_ds.class_names),
        in_channels=7,
        rank=cfg["model"]["lora_rank"],
        alpha=cfg["model"]["lora_alpha"],
        dropout=cfg["model"]["lora_dropout"],
    ).to(device)

    trainable, total = model.trainable_parameter_counts()
    print(f"Classes: {train_ds.class_names}")
    print(f"Trainable params: {trainable:,} / {total:,}")

    train_labels = [
        train_ds.target_remap[train_ds.dataset.targets[i]]
        for i in train_ds.samples
    ]
    class_weights = compute_class_weights(train_labels, len(train_ds.class_names)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    save_dir = cfg["train"]["save_dir"]
    ensure_dir(save_dir)
    print(f"[debug] save_dir = {os.path.abspath(save_dir)}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }
    csv_path = os.path.join(save_dir, "train_log.csv")

    best_f1 = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))

        append_history_csv(csv_path, epoch + 1, train_loss, val_metrics)

        print(
            f"[debug] history lengths: "
            f"train_loss={len(history['train_loss'])}, "
            f"val_loss={len(history['val_loss'])}, "
            f"val_accuracy={len(history['val_accuracy'])}, "
            f"val_macro_f1={len(history['val_macro_f1'])}"
        )

        save_training_curves(history, save_dir)
        save_confusion_matrix(
            y_true=val_metrics["y_true"],
            y_pred=val_metrics["y_pred"],
            output_dir=save_dir,
            class_names=train_ds.class_names,
            normalize=False,
        )
        save_confusion_matrix(
            y_true=val_metrics["y_true"],
            y_pred=val_metrics["y_pred"],
            output_dir=save_dir,
            class_names=train_ds.class_names,
            normalize=True,
        )
        print(f"[debug] finished writing plots for epoch {epoch + 1}")

        print(
            f"Epoch {epoch + 1}/{cfg['train']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_names": train_ds.class_names,
            "image_size": cfg["data"]["image_size"],
            "lora_rank": cfg["model"]["lora_rank"],
            "lora_alpha": cfg["model"]["lora_alpha"],
            "lora_dropout": cfg["model"]["lora_dropout"],
            "in_channels": 7,
            "history": history,
        }

        last_path = os.path.join(save_dir, "last_model.pt")
        torch.save(checkpoint, last_path)

        current_f1 = float(val_metrics["macro_f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)