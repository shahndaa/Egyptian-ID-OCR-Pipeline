"""
app.py
------
FastAPI deployment for the Egyptian National ID OCR pipeline.

Endpoints:
    POST /extract   -> Upload an ID card image, get back structured JSON
                        with name, address, and national_id (+ validation
                        and decoded metadata).
    GET  /health     -> Simple health check.
    GET  /            -> Minimal HTML upload form (for quick manual testing).

Run locally:
    uvicorn app:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI,
or http://127.0.0.1:8000/ for a simple upload form.
"""

import io
import time
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from preprocess import preprocess_id_card_array
from ocr_pipeline import IDCardOCR
from postprocess import postprocess_fields

app = FastAPI(
    title="Egyptian National ID OCR API",
    description="Extracts Name, Address, and National ID number from ID card images.",
    version="1.0.0",
)

# Load OCR engine once at startup (model/lang data loading is the slow part)
ocr_engine = IDCardOCR(lang="ara+eng", min_confidence=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head><title>Egyptian ID OCR</title></head>
        <body style="font-family: sans-serif; max-width: 500px; margin: 60px auto;">
            <h2>Egyptian National ID OCR</h2>
            <p>Upload a JPG/PNG image of an ID card:</p>
            <form action="/extract" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <br><br>
                <button type="submit">Extract</button>
            </form>
        </body>
    </html>
    """


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Accepts an image file (JPG/PNG), runs the full pipeline
    (preprocess -> OCR -> postprocess) and returns structured JSON.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPG/PNG).")

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    start = time.time()

    try:
        pre = preprocess_id_card_array(image)
        raw_fields = ocr_engine.extract_id_fields(pre["warped"])
        result = postprocess_fields(raw_fields)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return JSONResponse({
        "success": True,
        "data": result,
        "inference_time_ms": elapsed_ms,
    })
