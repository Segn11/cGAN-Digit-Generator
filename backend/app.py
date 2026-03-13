from __future__ import annotations

import base64
import io
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from generator import (
    LATENT_DIM,
    NUM_CLASSES,
    enable_batch_stat_fallback_if_needed,
    load_generator,
)


app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATOR = load_generator(device=DEVICE)
USE_BATCH_STAT_FALLBACK = enable_batch_stat_fallback_if_needed(GENERATOR)
FALLBACK_BATCH_SIZE = 8


@app.get("/")
def index() -> Any:
    return jsonify(
        {
            "name": "GAN Web App API",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "generate": "/generate",
            },
            "usage": "Send a POST request to /generate with a digit value from 0 to 9.",
        }
    )


def build_noise_tensor(raw_noise: Any) -> torch.Tensor:
    if raw_noise is None:
        return torch.randn(1, LATENT_DIM, device=DEVICE)

    if isinstance(raw_noise, (str, bytes)) or not isinstance(raw_noise, Sequence):
        raise ValueError("Noise must be a sequence of numeric values.")

    if len(raw_noise) != LATENT_DIM:
        raise ValueError(f"Noise must contain exactly {LATENT_DIM} values.")

    try:
        values = [float(value) for value in raw_noise]
    except (TypeError, ValueError) as exc:
        raise ValueError("Noise values must be numeric.") from exc

    return torch.tensor([values], dtype=torch.float32, device=DEVICE)


def tensor_to_data_url(image_tensor: torch.Tensor) -> str:
    grayscale = image_tensor.detach().cpu().squeeze(0).squeeze(0)
    normalized = grayscale.add(1).div(2).clamp(0, 1)
    image_array = (normalized.numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/generate")
def generate() -> Any:
    payload = request.get_json(silent=True) or {}
    raw_digit = payload.get("digit")

    if raw_digit is None:
        return jsonify({"error": "A digit label between 0 and 9 is required."}), 400

    try:
        digit = int(raw_digit)
    except (TypeError, ValueError):
        return jsonify({"error": "Digit label must be an integer between 0 and 9."}), 400

    if not 0 <= digit < NUM_CLASSES:
        return jsonify({"error": "Digit label must be between 0 and 9."}), 400

    try:
        noise = build_noise_tensor(payload.get("noise"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if USE_BATCH_STAT_FALLBACK:
        extra_noise_count = max(FALLBACK_BATCH_SIZE, 2) - noise.size(0)
        if extra_noise_count > 0:
            extra_noise = torch.randn(extra_noise_count, LATENT_DIM, device=DEVICE)
            noise_batch = torch.cat((noise, extra_noise), dim=0)
        else:
            noise_batch = noise

        labels = torch.full(
            (noise_batch.size(0),), digit, dtype=torch.long, device=DEVICE
        )
    else:
        noise_batch = noise
        labels = torch.tensor([digit], dtype=torch.long, device=DEVICE)

    with torch.inference_mode():
        generated_image = GENERATOR(noise_batch, labels)[:1]

    return jsonify(
        {
            "digit": digit,
            "image": tensor_to_data_url(generated_image),
            "noise": noise.detach().cpu().squeeze(0).tolist(),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)