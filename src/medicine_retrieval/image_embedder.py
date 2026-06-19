"""CLIP image embedding generation for medicine package images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipImageEmbedder:
    """Thin wrapper around Hugging Face CLIP image features."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.model.eval()

    def embed_paths(self, image_paths: list[Path], batch_size: int = 8) -> np.ndarray:
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                images = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                features = self.model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                vectors.append(features.cpu().numpy().astype(np.float32))

        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)
