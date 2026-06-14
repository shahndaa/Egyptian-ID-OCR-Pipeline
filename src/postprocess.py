"""
postprocess.py
---------------
Post-processing & logic validation layer for the Egyptian National ID
OCR pipeline. Cleans up "noisy" raw OCR output into a structured,
validated result.

Implements:
    1. Eastern <-> Western Arabic numeral normalization
    2. National ID Regex validation (exactly 14 digits)
    3. Text cleaning (strip special chars / background-pattern artifacts)
    4. Arabic-only check for the Name field
    5. National ID structure decoding (century, birth date, governorate, sex)
"""

import re

# Eastern Arabic-Indic digits -> Western Arabic digits
_EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Governorate codes used in the Egyptian National ID (digits 8-9)
GOVERNORATE_CODES = {
    "01": "Cairo", "02": "Alexandria", "03": "Port Said", "04": "Suez",
    "11": "Damietta", "12": "Dakahlia", "13": "Sharqia", "14": "Qalyubia",
    "15": "Kafr El Sheikh", "16": "Gharbia", "17": "Monufia", "18": "Beheira",
    "19": "Ismailia", "21": "Giza", "22": "Beni Suef", "23": "Fayoum",
    "24": "Minya", "25": "Assiut", "26": "Sohag", "27": "Qena",
    "28": "Aswan", "29": "Luxor", "31": "Red Sea", "32": "New Valley",
    "33": "Matrouh", "34": "North Sinai", "35": "South Sinai",
    "88": "Born outside Egypt",
}

NATIONAL_ID_REGEX = re.compile(r"^\d{14}$")


# --------------------------------------------------------------------------- #
# 1. Numeral normalization
# --------------------------------------------------------------------------- #
def normalize_digits(text: str) -> str:
    """Converts Eastern Arabic-Indic digits (٠-٩) to Western digits (0-9)."""
    return text.translate(_EASTERN_TO_WESTERN)


# --------------------------------------------------------------------------- #
# 2. Text cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """
    Removes OCR noise: stray punctuation, repeated symbols picked up from
    the card's background security pattern, and excess whitespace.
    """
    if text is None:
        return ""

    text = normalize_digits(text)

    # Remove characters that are neither Arabic letters, Latin letters,
    # digits, spaces, nor common separators (/ - .)
    text = re.sub(r"[^\u0600-\u06FF a-zA-Z0-9/\-\.]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_arabic_only(text: str) -> str:
    """Returns only the Arabic-script characters and spaces from `text`."""
    if text is None:
        return ""
    arabic = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    return re.sub(r"\s+", " ", arabic).strip()


# --------------------------------------------------------------------------- #
# 3. National ID validation
# --------------------------------------------------------------------------- #
def validate_national_id(raw_id: str):
    """
    Cleans and validates a National ID string.

    Returns:
        (is_valid: bool, cleaned_id: str or None, details: dict)
    """
    if raw_id is None:
        return False, None, {"error": "No ID detected"}

    cleaned = normalize_digits(raw_id)
    cleaned = re.sub(r"[^\d]", "", cleaned)  # keep digits only

    if not NATIONAL_ID_REGEX.match(cleaned):
        return False, cleaned, {
            "error": f"Expected 14 digits, got {len(cleaned)}",
            "raw_digits": cleaned,
        }

    details = decode_national_id(cleaned)
    return True, cleaned, details


def decode_national_id(national_id: str) -> dict:
    """
    Decodes the structure of a valid 14-digit Egyptian National ID:

        Position:  1   2-3  4-5  6-7  8-9   10-13   14
                  cent. yy   mm   dd   gov.  serial  sex_check

        - digit 1     : century (2 -> 1900s, 3 -> 2000s)
        - digits 2-7  : date of birth (YYMMDD)
        - digits 8-9  : governorate code (birth/registration)
        - digits 10-13: unique serial number
        - digit 14    : check digit; odd = male, even = female
    """
    century_digit = national_id[0]
    century = 1900 if century_digit == "2" else 2000 if century_digit == "3" else None

    yy, mm, dd = national_id[1:3], national_id[3:5], national_id[5:7]
    governorate_code = national_id[7:9]
    serial = national_id[9:13]
    check_digit = int(national_id[13])

    birth_year = f"{century + int(yy)}" if century else "Unknown"
    sex = "Male" if check_digit % 2 == 1 else "Female"

    return {
        "birth_date": f"{birth_year}-{mm}-{dd}" if century else None,
        "governorate_code": governorate_code,
        "governorate_name": GOVERNORATE_CODES.get(governorate_code, "Unknown"),
        "serial": serial,
        "sex": sex,
    }


# --------------------------------------------------------------------------- #
# 4. Full post-processing pass
# --------------------------------------------------------------------------- #
def postprocess_fields(raw_fields: dict) -> dict:
    """
    Takes the raw dict returned by IDCardOCR.extract_id_fields() and
    returns a cleaned, validated, structured result.
    """
    name_raw = raw_fields.get("name")
    address_raw = raw_fields.get("address")
    id_raw = raw_fields.get("national_id")

    name_clean = clean_text(name_raw)
    name_arabic_only = extract_arabic_only(name_raw or "")

    address_clean = clean_text(address_raw)

    is_valid_id, id_clean, id_details = validate_national_id(id_raw)

    return {
        "name": {
            "raw": name_raw,
            "cleaned": name_clean,
            "arabic_only": name_arabic_only,
            "is_arabic_valid": bool(name_arabic_only) and name_arabic_only == name_clean.replace("/", "").strip(),
        },
        "address": {
            "raw": address_raw,
            "cleaned": address_clean,
        },
        "national_id": {
            "raw": id_raw,
            "cleaned": id_clean,
            "is_valid": is_valid_id,
            "details": id_details,
        },
    }


if __name__ == "__main__":
    # Quick self-test with sample / edge-case inputs
    samples = {
        "national_id": "٢٩٩٠١٠١١٢٣٤٥٦٧",  # Eastern digits
        "name": "احمد محمد علي حسن ## //",
        "address": "15 شارع التحرير، القاهرة !!",
    }

    print("--- Raw samples ---")
    for k, v in samples.items():
        print(f"  {k}: {v}")

    result = postprocess_fields(samples)

    print("\n--- Post-processed result ---")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
