from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

from src.data.feature_engineering import build_multichannel_tensor_input
from src.models.vit_lora import ViTLoRAClassifier
from src.utils.io import ensure_dir
from src.utils.visualization import (
    draw_regions,
    normalize_heatmap,
    overlay_heatmap,
    smooth_and_threshold_heatmap,
)


def preprocess(image_rgb: np.ndarray, image_size: int):
    original = image_rgb.copy()
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    features = build_multichannel_tensor_input(resized)
    tensor = np.transpose(features, (2, 0, 1)).astype(np.float32)
    tensor = torch.from_numpy(tensor).unsqueeze(0)

    return original, tensor


def register_vit_hook(model, storage: dict):
    target_module = model.backbone.encoder.layers[-1].ln_1

    def forward_hook(module, inputs, output):
        storage["features"] = output
        output.retain_grad()

    return target_module.register_forward_hook(forward_hook)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = ViTLoRAClassifier(
        num_classes=len(class_names),
        in_channels=checkpoint.get("in_channels", 7),
        rank=checkpoint["lora_rank"],
        alpha=checkpoint["lora_alpha"],
        dropout=checkpoint["lora_dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    ensure_dir(args.output_dir)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}. Expected something like data/2750"
        )

    dataset = ImageFolder(root=str(dataset_root))

    feature_store = {}
    hook_handle = register_vit_hook(model, feature_store)

    max_examples = min(args.num_examples, len(dataset))

    for idx in range(max_examples):
        pil_image, _ = dataset[idx]
        image_rgb = np.array(pil_image)

        original_rgb, tensor = preprocess(image_rgb, image_size)
        tensor = tensor.to(device)
        tensor.requires_grad_(True)

        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_name = class_names[pred_idx]
        confidence = float(probs[0, pred_idx].item())

        model.zero_grad(set_to_none=True)
        score = logits[0, pred_idx]
        score.backward()

        features = feature_store["features"]
        grads = features.grad

        token_features = features[:, 1:, :]
        token_grads = grads[:, 1:, :]

        weights = token_grads.mean(dim=1, keepdim=True)
        cam = (token_features * weights).sum(dim=-1)
        cam = cam.reshape(1, 14, 14)
        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()

        cam = cv2.resize(
            cam,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        cam = normalize_heatmap(cam)

        mask = smooth_and_threshold_heatmap(
            cam,
            sigma=args.heatmap_sigma,
            threshold=args.heatmap_threshold,
        )

        overlay = overlay_heatmap(original_rgb, cam)
        overlay = draw_regions(overlay, mask)

        cv2.imwrite(
            str(Path(args.output_dir) / f"sample_{idx:03d}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(Path(args.output_dir) / f"sample_{idx:03d}_mask.png"),
            (mask * 255).astype(np.uint8),
        )

        with open(Path(args.output_dir) / f"sample_{idx:03d}_prediction.txt", "w", encoding="utf-8") as f:
            f.write(f"class: {pred_name}\n")
            f.write(f"confidence: {confidence:.4f}\n")

    hook_handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="data/2750")
    parser.add_argument("--num_examples", type=int, default=16)
    parser.add_argument("--heatmap_sigma", type=float, default=3.0)
    parser.add_argument("--heatmap_threshold", type=float, default=0.45)
    args = parser.parse_args()
    main(args)