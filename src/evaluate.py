"""
evaluate.py
-----------
Computes Character Error Rate (CER) and Word Error Rate (WER) for the
OCR pipeline on a small test set with known ground truth.

CER = (Substitutions + Deletions + Insertions) / Total characters in reference
WER = same formula, but computed at the word level.

Both are computed via Levenshtein (edit) distance.
"""

import sys
sys.path.append(".")
from preprocess import preprocess_id_card
from ocr_pipeline import IDCardOCR
from postprocess import postprocess_fields


def levenshtein(a, b):
    """Classic dynamic-programming edit distance between two sequences."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


# --------------------------------------------------------------------------- #
# Test set: (image_path, {field: ground_truth})
# --------------------------------------------------------------------------- #
TEST_SET = [
    {
        "image": "../images/synthetic_id_raw.png",
        "ground_truth": {
            "national_id": "29901011234567",
            "address": "15 شارع التحرير القاهرة",
            "name": "احمد محمد علي حسن",
        },
    },
    # Add more (image_path, ground_truth) pairs here to expand the test set.
]


def run_evaluation():
    ocr = IDCardOCR(lang="ara+eng", min_confidence=20)

    field_cers = {"name": [], "address": [], "national_id": []}
    field_wers = {"name": [], "address": [], "national_id": []}

    for sample in TEST_SET:
        pre = preprocess_id_card(sample["image"])
        raw = ocr.extract_id_fields(pre["warped"])
        result = postprocess_fields(raw)

        predictions = {
            "name": result["name"]["cleaned"],
            "address": result["address"]["cleaned"],
            "national_id": result["national_id"]["cleaned"] or "",
        }

        print(f"\nImage: {sample['image']}")
        for field, gt in sample["ground_truth"].items():
            pred = predictions[field]
            c = cer(gt, pred)
            w = wer(gt, pred)
            field_cers[field].append(c)
            field_wers[field].append(w)
            print(f"  {field}:")
            print(f"    GT  : {gt}")
            print(f"    Pred: {pred}")
            print(f"    CER : {c:.2%}   WER: {w:.2%}")

    print("\n=== Average across test set ===")
    for field in field_cers:
        if field_cers[field]:
            avg_cer = sum(field_cers[field]) / len(field_cers[field])
            avg_wer = sum(field_wers[field]) / len(field_wers[field])
            print(f"  {field:12s} -> CER: {avg_cer:.2%}   WER: {avg_wer:.2%}")


if __name__ == "__main__":
    run_evaluation()
