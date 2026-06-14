"""
preprocess.py
-------------
Pre-processing pipeline for Egyptian National ID card images using OpenCV.

Steps implemented:
    1. Card detection (largest 4-corner contour)
    2. Perspective Transform -> top-down, flat view
    3. Orientation correction (ensure landscape, text horizontal)
    4. Denoising (Gaussian Blur)
    5. Binarization (Adaptive Thresholding) for OCR-ready output

Each step can also be run independently for debugging / visualization.
"""

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Card detection + Perspective Transform
# --------------------------------------------------------------------------- #
def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]      # bottom-right has largest sum
    rect[1] = pts[np.argmin(diff)]   # top-right has smallest (x-y)
    rect[3] = pts[np.argmax(diff)]   # bottom-left has largest (x-y)
    return rect


def find_card_contour(image):
    """
    Finds the most likely 4-point contour representing the ID card.
    Falls back to the image bounding box if no clean quadrilateral is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Dilate to close small gaps in edges
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = image.shape[:2]
    img_area = h * w

    for c in contours[:10]:
        area = cv2.contourArea(c)
        if area < 0.15 * img_area:  # too small to be the card
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    # Fallback: use min-area rect of largest contour, or whole image
    if contours:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        return box.astype("float32")

    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")


def perspective_transform(image, pts=None, out_size=(1000, 630)):
    """
    Detects the card (if pts not provided) and warps it to a flat,
    top-down rectangular view of size `out_size` (width, height).
    """
    if pts is None:
        pts = find_card_contour(image)

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    target_w, target_h = out_size
    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_w, target_h))
    return warped


# --------------------------------------------------------------------------- #
# 2. Orientation correction
# --------------------------------------------------------------------------- #
def correct_orientation(image):
    """
    Ensures the card is in landscape orientation (wider than tall).
    Egyptian National IDs are landscape, so rotate 90 deg if portrait.
    """
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


# --------------------------------------------------------------------------- #
# 3. Denoising + Binarization
# --------------------------------------------------------------------------- #
def denoise_and_binarize(image, method="adaptive"):
    """
    Converts to grayscale, removes noise, and binarizes the image
    so text stands out from the background security pattern.

    method:
        "adaptive" -> Adaptive Gaussian Thresholding (handles uneven lighting)
        "otsu"     -> Global Otsu Thresholding (faster, uniform lighting)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise: stronger blur + median filter removes high-frequency
    # security-pattern noise while preserving thicker text strokes
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    denoised = cv2.medianBlur(denoised, 5)

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=41,
            C=20
        )
        # Morphological opening removes remaining thin diagonal-line noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:  # otsu
        _, binary = cv2.threshold(
            denoised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return gray, denoised, binary


# --------------------------------------------------------------------------- #
# 4. Full pipeline
# --------------------------------------------------------------------------- #
def preprocess_id_card(image_path, debug_dir=None):
    """
    Runs the full pre-processing pipeline on an ID card image file.

    Returns a dict with intermediate results:
        - original
        - warped   (after perspective transform + orientation correction)
        - gray
        - denoised
        - binary   (final OCR-ready image)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = preprocess_id_card_array(image)

    if debug_dir:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "1_original.png"), image)
        cv2.imwrite(os.path.join(debug_dir, "2_warped.png"), results["warped"])
        cv2.imwrite(os.path.join(debug_dir, "3_gray.png"), results["gray"])
        cv2.imwrite(os.path.join(debug_dir, "4_binary.png"), results["binary"])

    return results


def preprocess_id_card_array(image: np.ndarray) -> dict:
    """
    In-memory version of `preprocess_id_card` — takes a BGR numpy array
    (e.g. from cv2.imdecode on an uploaded file) and runs the same
    pipeline: perspective transform -> orientation correction ->
    denoise & binarize.
    """
    warped = perspective_transform(image)
    warped = correct_orientation(warped)
    gray, denoised, binary = denoise_and_binarize(warped, method="adaptive")

    return {
        "original": image,
        "warped": warped,
        "gray": gray,
        "denoised": denoised,
        "binary": binary,
    }


if __name__ == "__main__":
    results = preprocess_id_card(
        "images/synthetic_id_raw.png",
        debug_dir="images/preprocessing_steps"
    )
    print("Preprocessing complete. Steps saved to images/preprocessing_steps/")
    for k, v in results.items():
        print(f"  {k}: shape={v.shape}")
