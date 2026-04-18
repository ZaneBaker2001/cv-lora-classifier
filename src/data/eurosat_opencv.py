from __future__ import annotations

import os
import ssl
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import certifi
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.data.feature_engineering import build_multichannel_tensor_input


EUROSAT_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"


def _configure_ssl() -> None:
    ca_file = certifi.where()
    os.environ["SSL_CERT_FILE"] = ca_file
    os.environ["REQUESTS_CA_BUNDLE"] = ca_file
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=ca_file)


def _ensure_eurosat_downloaded(root_dir: str) -> Path:
    _configure_ssl()

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    archive_name = Path(urlparse(EUROSAT_URL).path).name
    archive_path = root / archive_name
    extract_dir = root / "2750"

    if extract_dir.exists():
        return extract_dir

    if not archive_path.exists():
        print(f"Downloading EuroSAT to {archive_path} ...")
        urlretrieve(EUROSAT_URL, archive_path)

    print(f"Extracting {archive_path} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(root)

    if not extract_dir.exists():
        raise RuntimeError(
            f"EuroSAT was downloaded, but expected extracted folder {extract_dir} was not found."
        )

    return extract_dir


class EuroSATOpenCV(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: int = 224,
        transform=None,
        selected_classes: Optional[List[str]] = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        dataset_dir = _ensure_eurosat_downloaded(root_dir)
        self.dataset = ImageFolder(root=str(dataset_dir))

        self.image_size = image_size
        self.transform = transform

        self.all_class_names = list(self.dataset.classes)
        self.class_to_original_idx = {name: i for i, name in enumerate(self.all_class_names)}

        if selected_classes:
            invalid = [c for c in selected_classes if c not in self.class_to_original_idx]
            if invalid:
                raise ValueError(f"Invalid EuroSAT classes: {invalid}")
            self.class_names = list(selected_classes)
            allowed = {self.class_to_original_idx[c] for c in self.class_names}
        else:
            self.class_names = self.all_class_names
            allowed = set(range(len(self.all_class_names)))

        all_indices = [i for i, (_, target) in enumerate(self.dataset.samples) if target in allowed]

        rng = np.random.default_rng(42)
        all_indices = np.array(all_indices)
        rng.shuffle(all_indices)

        split_idx = int(0.8 * len(all_indices))
        if split == "train":
            self.samples = all_indices[:split_idx].tolist()
        else:
            self.samples = all_indices[split_idx:].tolist()

        self.target_remap = {
            self.class_to_original_idx[name]: new_idx for new_idx, name in enumerate(self.class_names)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        real_idx = self.samples[idx]
        pil_image, original_label = self.dataset[real_idx]

        image_rgb = np.array(pil_image)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            image_rgb = self.transform(image_rgb)

        image = build_multichannel_tensor_input(image_rgb)
        image = np.transpose(image, (2, 0, 1))

        label = self.target_remap[original_label]

        return {
            "image": torch.from_numpy(image),
            "label": torch.tensor(label, dtype=torch.long),
        }