from typing import Dict

import numpy as np
import torch


@torch.no_grad()
def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    accuracy = float((preds == labels).float().mean().item())

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
    f1_scores = []

    for class_idx in range(num_classes):
        tp = np.logical_and(preds_np == class_idx, labels_np == class_idx).sum()
        fp = np.logical_and(preds_np == class_idx, labels_np != class_idx).sum()
        fn = np.logical_and(preds_np != class_idx, labels_np == class_idx).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(float(f1))

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }