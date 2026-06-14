"""
ocr_pipeline.py
---------------
Text Detection & Recognition for the Egyptian National ID pipeline.

Engine: Tesseract OCR (via pytesseract) with the Arabic ('ara') + English
('eng') trained language data.

Why Tesseract here:
    - Works fully offline once `tesseract-ocr` + `tesseract-ocr-ara` are
      installed (apt-get install tesseract-ocr tesseract-ocr-ara)
    - `image_to_data()` gives word-level bounding boxes -> Detection
    - The recognized `text` field for each box -> Recognition
    - No GPU / large model download required (good for quick deployment)

Swapping engines:
    The same `IDCardOCR` interface (`detect_and_recognize`, `extract_id_fields`)
    could be backed by EasyOCR (CRAFT + CRNN, better accuracy on noisy/curved
    text) or PaddleOCR (PP-OCRv4, strong multilingual support) by replacing
    the body of `detect_and_recognize()`. Both alternatives are noted in the
    README along with trade-offs.

NOTE on Eastern vs Western Arabic numerals:
    Egyptian IDs print Western Arabic numerals (0-9). OCR can occasionally
    emit Eastern Arabic-Indic digits (0-9 -> Arabic-Indic 0-9) for
    Arabic-locale text. Both are normalized in postprocess.py.
"""

import pytesseract
from pytesseract import Output
import numpy as np


class IDCardOCR:
    def __init__(self, lang="ara+eng", min_confidence=30):
        """
        lang: Tesseract language string. 'ara+eng' lets the engine
              recognize Arabic script alongside Latin digits/labels
              that appear on the card (e.g. 'Name', 'National ID').
        min_confidence: discard detections below this confidence (0-100).
        """
        self.lang = lang
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------ #
    # Detection + Recognition
    # ------------------------------------------------------------------ #
    def detect_and_recognize(self, image: np.ndarray):
        """
        Runs word-level detection (bounding boxes) + recognition (text)
        using Tesseract's image_to_data.

        Returns a list of dicts:
            [{"bbox": (x, y, w, h), "text": str, "confidence": float,
              "line_num": int, "block_num": int}, ...]
        """
        data = pytesseract.image_to_data(
            image, lang=self.lang, output_type=Output.DICT,
            config="--psm 6"  # Assume a single uniform block of text
        )

        results = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            if not text or conf < self.min_confidence:
                continue

            results.append({
                "bbox": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                "text": text,
                "confidence": conf,
                "line_num": data["line_num"][i],
                "block_num": data["block_num"][i],
            })
        return results

    # ------------------------------------------------------------------ #
    # Field extraction
    # ------------------------------------------------------------------ #
    def _group_by_line(self, word_results):
        """Groups word-level results into full text lines, preserving order."""
        lines = {}
        for r in word_results:
            key = (r["block_num"], r["line_num"])
            lines.setdefault(key, []).append(r)

        line_texts = []
        for key, words in lines.items():
            # Sort words left-to-right within the line
            words_sorted = sorted(words, key=lambda w: w["bbox"][0])
            text = " ".join(w["text"] for w in words_sorted)
            avg_conf = sum(w["confidence"] for w in words_sorted) / len(words_sorted)
            y = min(w["bbox"][1] for w in words_sorted)
            line_texts.append({"text": text, "confidence": avg_conf, "y": y})

        # Sort lines top-to-bottom
        line_texts.sort(key=lambda l: l["y"])
        return line_texts

    def extract_id_fields(self, image: np.ndarray, photo_region_ratio: float = 0.28):
        """
        Higher-level helper: crops out the photo placeholder (left portion
        of the card, ~28% width by default on Egyptian IDs), segments the
        remaining text region into horizontal bands using a row-projection
        profile (more robust than Tesseract's internal line grouping for
        mixed Arabic/English bidirectional text), runs OCR per band, then
        maps bands to Name / Address / National ID using keyword + digit
        heuristics.

        photo_region_ratio: fraction of card width occupied by the photo
                             box on the left side (tune per template).
        """
        import cv2

        h, w = image.shape[:2]
        text_region = image[:, int(w * photo_region_ratio):]

        # Upscale 2x — improves small-text recognition accuracy
        upscaled = cv2.resize(text_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = upscaled if upscaled.ndim == 2 else cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Row projection: count dark (text) pixels per row
        row_sums = binary.sum(axis=1)
        threshold = row_sums.max() * 0.04

        bands = []
        in_band = False
        start = 0
        for y, val in enumerate(row_sums):
            if val > threshold and not in_band:
                in_band, start = True, y
            elif val <= threshold and in_band:
                in_band = False
                if y - start > 10:  # ignore tiny noise bands
                    bands.append((start, y))
        if in_band:
            bands.append((start, len(row_sums)))

        # Merge bands that are very close together (same visual line)
        merged = []
        for b in bands:
            if merged and b[0] - merged[-1][1] < 8:
                merged[-1] = (merged[-1][0], b[1])
            else:
                merged.append(b)

        lines = []
        ih, iw = gray.shape
        pad = 6
        for (y1, y2) in merged:
            y1p, y2p = max(0, y1 - pad), min(ih, y2 + pad)
            crop = gray[y1p:y2p, 0:iw]
            text = pytesseract.image_to_string(
                crop, lang=self.lang, config="--psm 7"
            ).strip()
            if text:
                lines.append({"text": text, "y": y1, "confidence": None})

        national_id = None
        name = None
        address = None
        eastern_digits = "٠١٢٣٤٥٦٧٨٩"

        for i, line in enumerate(lines):
            text = line["text"]
            digits_only = "".join(ch for ch in text if ch.isdigit() or ch in eastern_digits)

            if len(digits_only) >= 10 and national_id is None:
                national_id = digits_only
                continue

            if ("Name" in text or "اسم" in text) and name is None:
                if i + 1 < len(lines):
                    name = lines[i + 1]["text"]
                continue

            if ("ddress" in text or "نوان" in text) and address is None:
                if i + 1 < len(lines):
                    address = lines[i + 1]["text"]
                continue

        # Positional fallback: Egyptian ID layout is consistently
        # [Name label, Name value, Address label, Address value,
        #  National ID label, National ID value]. If keyword matching
        # above failed (common with low-quality OCR on label text),
        # fall back to position-based extraction.
        if (name is None or address is None or national_id is None) and len(lines) >= 6:
            if name is None:
                name = lines[1]["text"]
            if address is None:
                address = lines[3]["text"]
            if national_id is None:
                digits_only = "".join(
                    ch for ch in lines[5]["text"]
                    if ch.isdigit() or ch in eastern_digits
                )
                if digits_only:
                    national_id = digits_only

        return {
            "name": name,
            "address": address,
            "national_id": national_id,
            "lines": lines,
            "raw_results": self.detect_and_recognize(image),
        }


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from preprocess import preprocess_id_card

    results = preprocess_id_card("../images/synthetic_id_raw.png")

    ocr = IDCardOCR(lang="ara+eng", min_confidence=20)

    print("Running OCR on warped (color) image...\n")
    fields = ocr.extract_id_fields(results["warped"])

    print("--- Extracted Fields ---")
    print(f"Name        : {fields['name']}")
    print(f"Address     : {fields['address']}")
    print(f"National ID : {fields['national_id']}")

    print("\n--- All Detected Lines ---")
    for l in fields["lines"]:
        print(f"  {l['text']}")
