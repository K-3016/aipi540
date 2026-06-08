from __future__ import annotations

import base64
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from .inference import PredictionService

app = FastAPI(
    title="Brain Tumor Image Classifier",
    description="Research-only four-class brain MRI classification API.",
    version="0.2.0",
)


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    artifact_dir = Path(os.getenv("BRAIN_TUMOR_ARTIFACT_DIR", "models"))
    return PredictionService(artifact_dir)


@app.get("/health")
def health() -> dict:
    try:
        service = get_service()
    except (FileNotFoundError, RuntimeError, KeyError) as error:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {error}") from error
    return {"status": "ok", "model": service.metadata["deployed_model"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    image = await _read_image(file)
    return get_service().predict(image)


@app.post("/explain")
async def explain(file: UploadFile = File(...)) -> dict:
    image = await _read_image(file)
    prediction, overlay = get_service().explain(image)
    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    return {
        **prediction,
        "grad_cam_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "explanation_note": (
            "Grad-CAM shows image regions that most influenced this prediction. "
            "It does not prove that the model used clinically valid evidence."
        ),
    }


async def _read_image(file: UploadFile) -> Image.Image:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image.")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    try:
        image = Image.open(BytesIO(content))
        image.verify()
        image = Image.open(BytesIO(content)).convert("L")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Invalid image file.") from error
    return image


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Brain Tumor Image Classifier</title>
        <style>
          body { font-family: system-ui; max-width: 680px; margin: 4rem auto; padding: 0 1rem; }
          form { display: grid; gap: 1rem; padding: 1.5rem; border: 1px solid #bbb; border-radius: 12px; }
          button { width: fit-content; padding: .6rem 1rem; }
          pre { white-space: pre-wrap; background: #f4f4f4; padding: 1rem; border-radius: 8px; }
          .warning { color: #8a2d17; }
        </style>
      </head>
      <body>
        <h1>Brain Tumor Image Classifier</h1>
        <p>Classes: glioma, meningioma, pituitary tumor, and normal.</p>
        <p class="warning"><strong>Research use only.</strong> Not for diagnosis or treatment.</p>
        <form id="form">
          <label>MRI image <input name="file" type="file" accept="image/*" required></label>
          <button type="submit">Analyze image</button>
        </form>
        <pre id="result">Upload an image to begin.</pre>
        <img id="heatmap" alt="Grad-CAM explanation" style="display:none; max-width:100%;">
        <script>
          document.querySelector("#form").addEventListener("submit", async (event) => {
            event.preventDefault();
            const result = document.querySelector("#result");
            result.textContent = "Analyzing...";
            const response = await fetch("/explain", { method: "POST", body: new FormData(event.target) });
            const payload = await response.json();
            const heatmap = document.querySelector("#heatmap");
            if (payload.grad_cam_png_base64) {
              heatmap.src = `data:image/png;base64,${payload.grad_cam_png_base64}`;
              heatmap.style.display = "block";
              delete payload.grad_cam_png_base64;
            }
            result.textContent = JSON.stringify(payload, null, 2);
          });
        </script>
      </body>
    </html>
    """
