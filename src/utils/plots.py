import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_training_curves(history: Dict[str, List[float]], output_dir: str) -> None:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not history["train_loss"]:
        raise ValueError("history['train_loss'] is empty")

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = output_path / "loss_curve.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_accuracy"], label="val_accuracy")
    plt.plot(epochs, history["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    metrics_path = output_path / "metrics_curve.png"
    plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close()

    if not loss_path.exists():
        raise RuntimeError(f"Failed to create {loss_path}")
    if not metrics_path.exists():
        raise RuntimeError(f"Failed to create {metrics_path}")

    print(f"[plots] wrote: {loss_path}")
    print(f"[plots] wrote: {metrics_path}")


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_dir: str,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
) -> None:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    num_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    display_matrix = cm.astype(np.float32)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display_matrix = display_matrix / row_sums

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    else:
        class_names = list(class_names)
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names length ({len(class_names)}) does not match num_classes ({num_classes})"
            )

    plt.figure(figsize=(8, 6))
    plt.imshow(display_matrix, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = display_matrix.max() / 2.0 if display_matrix.size > 0 else 0.0
    value_format = ".2f" if normalize else "d"

    for i in range(num_classes):
        for j in range(num_classes):
            value = display_matrix[i, j] if normalize else cm[i, j]
            plt.text(
                j,
                i,
                format(value, value_format),
                ha="center",
                va="center",
                color="white" if display_matrix[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    filename = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    cm_path = output_path / filename
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    if not cm_path.exists():
        raise RuntimeError(f"Failed to create {cm_path}")

    print(f"[plots] wrote: {cm_path}")