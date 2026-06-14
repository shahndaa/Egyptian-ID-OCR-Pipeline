"""
generate_synthetic_id.py
-------------------------
Generates a SYNTHETIC (fake) Egyptian National ID card image for testing
the OCR pipeline. This avoids using any real personal data while still
allowing us to test:
    - Perspective distortion / tilted capture
    - Arabic text rendering (name, address)
    - 14-digit National ID number
    - Background security-pattern noise

Output: images/synthetic_id_raw.png (tilted, noisy)
        images/synthetic_id_flat.png (flat reference, for comparison)
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

FONT_PATH = "fonts/NotoKufiArabic.ttf"
OUT_DIR = "images"
os.makedirs(OUT_DIR, exist_ok=True)

CARD_W, CARD_H = 1000, 630  # standard ID-card-ish aspect ratio


def build_flat_card():
    """Builds a flat, front-facing synthetic ID card with Arabic text."""
    card = Image.new("RGB", (CARD_W, CARD_H), (235, 240, 245))
    draw = ImageDraw.Draw(card)

    # Background "security pattern" — light repeating diagonal lines
    for i in range(-CARD_H, CARD_W, 18):
        draw.line([(i, 0), (i + CARD_H, CARD_H)], fill=(210, 218, 228), width=2)

    # Header bar
    draw.rectangle([0, 0, CARD_W, 90], fill=(20, 60, 110))
    title_font = ImageFont.truetype(FONT_PATH, 40)
    draw.text((30, 20), "جمهورية مصر العربية", font=title_font, fill=(255, 255, 255))
    draw.text((CARD_W - 320, 20), "ARAB REPUBLIC OF EGYPT", font=ImageFont.truetype(FONT_PATH, 24), fill=(255, 255, 255))

    # Photo placeholder
    draw.rectangle([40, 130, 260, 410], outline=(80, 80, 80), width=3, fill=(200, 200, 200))
    ph_font = ImageFont.truetype(FONT_PATH, 22)
    draw.text((90, 250), "PHOTO", font=ph_font, fill=(120, 120, 120))

    # Labels + values (synthetic/fake data only)
    label_font = ImageFont.truetype(FONT_PATH, 26)
    value_font = ImageFont.truetype(FONT_PATH, 32)

    fields = [
        ("الاسم / Name", "احمد محمد علي حسن"),
        ("العنوان / Address", "15 شارع التحرير القاهرة"),
        ("الرقم القومي / National ID", "29901011234567"),
    ]

    y = 150
    for label, value in fields:
        draw.text((300, y), label, font=label_font, fill=(60, 60, 60))
        draw.text((300, y + 35), value, font=value_font, fill=(10, 10, 10))
        y += 100

    return card


def warp_for_realism(flat_img_path, out_path):
    """
    Takes a flat card image and applies a perspective warp + slight
    rotation + noise to simulate a real phone-camera capture at an angle.
    """
    img = cv2.imread(flat_img_path)
    h, w = img.shape[:2]

    # Place the flat card onto a larger "table/background" canvas
    canvas_w, canvas_h = int(w * 1.4), int(h * 1.5)
    canvas = np.full((canvas_h, canvas_w, 3), (90, 100, 95), dtype=np.uint8)  # desk-like background

    # Define source corners (flat card) and destination corners (skewed placement)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    offset_x, offset_y = (canvas_w - w) // 2, (canvas_h - h) // 2
    dst_pts = np.float32([
        [offset_x + 40,  offset_y + 60],   # top-left pushed right/down
        [offset_x + w - 10, offset_y + 10],  # top-right
        [offset_x + w - 60, offset_y + h - 20],  # bottom-right
        [offset_x + 20,  offset_y + h - 70],   # bottom-left
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (canvas_w, canvas_h), borderValue=(90, 100, 95))

    # Add mild gaussian noise to simulate low-quality camera
    noise = np.random.normal(0, 8, warped.shape).astype(np.int16)
    noisy = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Slight blur to simulate camera focus issues
    noisy = cv2.GaussianBlur(noisy, (3, 3), 0)

    cv2.imwrite(out_path, noisy)
    return dst_pts, (canvas_w, canvas_h)


if __name__ == "__main__":
    flat = build_flat_card()
    flat_path = os.path.join(OUT_DIR, "synthetic_id_flat.png")
    flat.save(flat_path)
    print(f"Saved flat reference card -> {flat_path}")

    raw_path = os.path.join(OUT_DIR, "synthetic_id_raw.png")
    corners, canvas_size = warp_for_realism(flat_path, raw_path)
    print(f"Saved warped/noisy 'camera capture' -> {raw_path}")
    print(f"Card corners in raw image (for reference): \n{corners}")
