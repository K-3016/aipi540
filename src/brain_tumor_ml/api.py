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
        <title>NeuroVision MRI Classifier</title>
        <style>
          :root { color-scheme: dark; }
          * { box-sizing: border-box; }
          body {
            margin: 0; min-height: 100vh; font-family: Inter, ui-sans-serif, system-ui;
            color: #eaf2ff; background:
              radial-gradient(circle at 10% 10%, #15325d 0, transparent 36%),
              radial-gradient(circle at 90% 20%, #173f45 0, transparent 32%), #07111f;
          }
          main { width: min(1080px, calc(100% - 32px)); margin: 0 auto; padding: 48px 0 72px; }
          header { margin-bottom: 28px; }
          .eyebrow { color: #6ee7d8; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
          h1 { margin: 8px 0; font-size: clamp(2rem, 5vw, 3.5rem); }
          .subtitle { color: #a9b9d0; max-width: 720px; line-height: 1.6; }
          .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
          .card {
            background: rgba(13, 27, 46, .88); border: 1px solid #263e5d;
            border-radius: 20px; padding: 24px; box-shadow: 0 18px 50px rgba(0,0,0,.25);
          }
          form { display: grid; gap: 18px; }
          .dropzone {
            display: grid; place-items: center; min-height: 220px; padding: 24px;
            border: 1px dashed #4e7198; border-radius: 16px; text-align: center; cursor: pointer;
          }
          input { max-width: 100%; }
          button {
            border: 0; border-radius: 12px; padding: 13px 18px; font-weight: 750;
            color: #04101d; background: linear-gradient(135deg, #6ee7d8, #7db5ff); cursor: pointer;
          }
          button:disabled { opacity: .55; cursor: wait; }
          .preview-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
          .preview-grid img { width: 100%; aspect-ratio: 1; object-fit: contain; background: #030914; border-radius: 12px; }
          .label { color: #8ea5c2; font-size: .82rem; margin: 8px 0 5px; }
          .prediction { font-size: 1.65rem; font-weight: 800; text-transform: capitalize; }
          .confidence-track { height: 10px; border-radius: 20px; overflow: hidden; background: #26364a; }
          .confidence-fill { height: 100%; width: 0; background: #6ee7d8; transition: width .4s ease; }
          .probability { display: grid; grid-template-columns: 100px 1fr 48px; gap: 10px; align-items: center; margin: 10px 0; }
          .bar { height: 7px; background: #26364a; border-radius: 10px; overflow: hidden; }
          .bar span { display: block; height: 100%; background: #7db5ff; }
          .warning { margin-top: 22px; padding: 14px; color: #ffd7a3; background: #382713; border-radius: 12px; }
          .hidden { display: none; }
          @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
        </style>
      </head>
      <body>
        <main>
          <header>
            <div class="eyebrow">Explainable computer vision</div>
            <h1>NeuroVision MRI Classifier</h1>
            <p class="subtitle">Upload a brain MRI to estimate one of four classes and inspect
              the model's Grad-CAM attention map.</p>
          </header>
          <div class="grid">
            <section class="card">
              <form id="form">
                <label class="dropzone">
                  <strong>Select a brain MRI image</strong>
                  <span class="subtitle">PNG, JPEG, BMP, or TIFF up to 10 MB</span>
                  <input id="file" name="file" type="file" accept="image/*" required>
                </label>
                <button id="submit" type="submit">Analyze MRI</button>
              </form>
              <p class="warning"><strong>Not for clinical diagnosis.</strong> This educational
                model must not replace a qualified healthcare professional.</p>
            </section>
            <section class="card">
              <div id="empty" class="subtitle">Results will appear here after an image is analyzed.</div>
              <div id="results" class="hidden">
                <div class="preview-grid">
                  <div><div class="label">Uploaded MRI</div><img id="preview" alt="Uploaded MRI"></div>
                  <div><div class="label">Grad-CAM heatmap</div><img id="heatmap" alt="Grad-CAM explanation"></div>
                </div>
                <div class="label">Prediction</div>
                <div id="prediction" class="prediction"></div>
                <div class="label">Confidence: <span id="confidence"></span></div>
                <div class="confidence-track"><div id="confidence-fill" class="confidence-fill"></div></div>
                <div class="label">Class probabilities</div>
                <div id="probabilities"></div>
              </div>
            </section>
          </div>
        </main>
        <script>
          const fileInput = document.querySelector("#file");
          fileInput.addEventListener("change", () => {
            if (fileInput.files[0]) {
              document.querySelector("#preview").src = URL.createObjectURL(fileInput.files[0]);
            }
          });
          document.querySelector("#form").addEventListener("submit", async (event) => {
            event.preventDefault();
            const button = document.querySelector("#submit");
            button.disabled = true;
            button.textContent = "Analyzing...";
            const response = await fetch("/explain", { method: "POST", body: new FormData(event.target) });
            const payload = await response.json();
            button.disabled = false;
            button.textContent = "Analyze MRI";
            if (!response.ok) {
              alert(payload.detail || "Unable to analyze this image.");
              return;
            }
            document.querySelector("#empty").classList.add("hidden");
            document.querySelector("#results").classList.remove("hidden");
            document.querySelector("#heatmap").src = `data:image/png;base64,${payload.grad_cam_png_base64}`;
            document.querySelector("#prediction").textContent = payload.predicted_class;
            const confidence = Math.round(payload.confidence * 100);
            document.querySelector("#confidence").textContent = `${confidence}%`;
            document.querySelector("#confidence-fill").style.width = `${confidence}%`;
            document.querySelector("#probabilities").innerHTML = Object.entries(payload.probabilities)
              .map(([name, value]) => {
                const percent = Math.round(value * 100);
                return `<div class="probability"><span>${name}</span><div class="bar"><span style="width:${percent}%"></span></div><span>${percent}%</span></div>`;
              }).join("");
          });
        </script>
      </body>
    </html>
    """
