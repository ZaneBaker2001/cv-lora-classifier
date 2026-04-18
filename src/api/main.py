from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.data.feature_engineering import build_multichannel_tensor_input
from src.models.vit_lora import ViTLoRAClassifier
from src.utils.visualization import (
    draw_regions,
    normalize_heatmap,
    overlay_heatmap,
    smooth_and_threshold_heatmap,
)


CHECKPOINT_PATH = Path("outputs/best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


app = FastAPI(
    title="EuroSAT LoRA Vision Classifier API",
    version="1.0.0",
    description="FastAPI service for EuroSAT land-use classification with LoRA-fine-tuned ViT and OpenCV/SciPy feature engineering.",
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


class ModelService:
    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. Train the model first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        self.class_names = checkpoint["class_names"]
        self.image_size = int(checkpoint["image_size"])
        self.in_channels = int(checkpoint.get("in_channels", 7))
        self.heatmap_sigma = 3.0
        self.heatmap_threshold = 0.45

        self.model = ViTLoRAClassifier(
            num_classes=len(self.class_names),
            in_channels=self.in_channels,
            rank=int(checkpoint["lora_rank"]),
            alpha=int(checkpoint["lora_alpha"]),
            dropout=float(checkpoint["lora_dropout"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

        self.feature_store: dict[str, Any] = {}
        self._hook = self._register_vit_hook()

    def _register_vit_hook(self):
        target_module = self.model.backbone.encoder.layers[-1].ln_1

        def forward_hook(module, inputs, output):
            self.feature_store["features"] = output
            output.retain_grad()

        return target_module.register_forward_hook(forward_hook)

    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Unable to decode uploaded image.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_rgb = image_rgb.copy()

        resized = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        features = build_multichannel_tensor_input(resized)
        tensor = np.transpose(features, (2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)

        return original_rgb, tensor

    def _generate_cam(self, tensor: torch.Tensor, pred_idx: int, output_size: tuple[int, int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(tensor)
        score = logits[0, pred_idx]
        score.backward()

        features = self.feature_store.get("features")
        if features is None:
            raise RuntimeError("Transformer features were not captured by the forward hook.")

        grads = features.grad
        if grads is None:
            raise RuntimeError("Gradients for transformer features were not captured.")

        token_features = features[:, 1:, :]
        token_grads = grads[:, 1:, :]

        weights = token_grads.mean(dim=1, keepdim=True)
        cam = (token_features * weights).sum(dim=-1)
        cam = cam.reshape(1, 14, 14)
        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()

        cam = cv2.resize(cam, output_size, interpolation=cv2.INTER_CUBIC)
        cam = normalize_heatmap(cam)
        return cam

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        original_rgb, tensor = self.preprocess(image_bytes)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_name = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        heatmap = self._generate_cam(
            tensor=tensor,
            pred_idx=pred_idx,
            output_size=(original_rgb.shape[1], original_rgb.shape[0]),
        )

        binary_mask = smooth_and_threshold_heatmap(
            heatmap,
            sigma=self.heatmap_sigma,
            threshold=self.heatmap_threshold,
        )
        heatmap_overlay = overlay_heatmap(original_rgb, heatmap)
        region_overlay = draw_regions(heatmap_overlay, binary_mask)

        overlay_bgr = cv2.cvtColor(region_overlay, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", overlay_bgr)
        if not success:
            raise RuntimeError("Failed to encode overlay image.")

        overlay_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        class_probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(self.class_names, probs)
        }

        return PredictionResponse(
            predicted_class=pred_name,
            confidence=confidence,
            class_probabilities=class_probabilities,
            overlay_png_base64=overlay_base64,
        )


service: ModelService | None = None


@app.on_event("startup")
def load_model() -> None:
    global service
    service = ModelService(CHECKPOINT_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "EuroSAT LoRA Vision Classifier API is running."}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if service is None:
        raise HTTPException(status_code=500, detail="Model service is not initialized.")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return service.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc