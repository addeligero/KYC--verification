import os
import re
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image
from passporteye import read_mrz

from app.utils.image import preprocess_for_ocr

# Optional: allow overriding tesseract path via env var on Windows
TESS_CMD = os.getenv("TESSERACT_CMD")
if TESS_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESS_CMD


@dataclass
class ParsedFields:
    full_name: Optional[str] = None
    dob: Optional[str] = None
    document_number: Optional[str] = None
    nationality: Optional[str] = None
    expiry_date: Optional[str] = None
    address: Optional[str] = None


DOB_PATTERNS = [
    r"(\d{4}[-/\.]\d{2}[-/\.]\d{2})",    # YYYY-MM-DD
    r"(\d{2}[-/\.]\d{2}[-/\.]\d{4})",    # DD-MM-YYYY
]
DOCNUM_PATTERNS = [
    r"\b([A-Z]{1,2}\d{6,9})\b",
    r"\b(\d{8,10})\b",
]


def _normalize_dob(s: str) -> Optional[str]:
    s = s.strip()
    m = re.match(r"(\d{4})[-/\.](\d{2})[-/\.](\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r"(\d{2})[-/\.](\d{2})[-/\.](\d{4})", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def extract_text_fields(id_front_bgr: np.ndarray, id_back_bgr: Optional[np.ndarray]) -> tuple[ParsedFields, float]:
    # Simple OCR pipeline using Tesseract
    fields = ParsedFields()
    confs = []

    for img in [id_front_bgr, id_back_bgr]:
        if img is None:
            continue
        pre = preprocess_for_ocr(img)
        pil = Image.fromarray(pre)
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, lang="eng")
        text = " ".join([w for w in data["text"] if w and w.strip()])
        conf_vals = [int(c) for c in data.get("conf", []) if isinstance(c, str) and c.isdigit()]
        if conf_vals:
            confs.append(max(0, min(100, sum(conf_vals) / len(conf_vals))) / 100.0)

        # Name: look for lines with uppercase words
        candidates = re.findall(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b", text)
        if candidates and not fields.full_name:
            fields.full_name = candidates[0].title()

        # DOB
        for pat in DOB_PATTERNS:
            m = re.search(pat, text)
            if m and not fields.dob:
                fields.dob = _normalize_dob(m.group(1))
                break

        # Document number
        for pat in DOCNUM_PATTERNS:
            m = re.search(pat, text)
            if m and not fields.document_number:
                fields.document_number = m.group(1)
                break

        # Nationality
        nat = re.search(r"\b(Nationality|Citizenship)\s*[:\-]?\s*([A-Z]{3,})\b", text, re.IGNORECASE)
        if nat and not fields.nationality:
            fields.nationality = nat.group(2).upper()

        # Expiry
        exp = re.search(r"(Expiry|Expires|Valid\s*Until)\s*[:\-]?\s*([0-9./-]{8,10})", text, re.IGNORECASE)
        if exp and not fields.expiry_date:
            fields.expiry_date = _normalize_dob(exp.group(2)) or exp.group(2)

        # Address (very rough)
        addr = re.search(r"(Address|Residence)\s*[:\-]?\s*(.+)$", text, re.IGNORECASE)
        if addr and not fields.address:
            fields.address = addr.group(2).strip()

    ocr_conf = max(confs) if confs else 0.4
    return fields, ocr_conf


def try_mrz_parse(id_front_bgr: np.ndarray) -> ParsedFields:
    """
    MRZ parsing with Windows-safe temp file handling and graceful failure.
    Set KYC_DISABLE_MRZ=1 to skip MRZ entirely (e.g., if deps conflict with NumPy 2.x).
    """
    if os.getenv("KYC_DISABLE_MRZ", "").strip().lower() in ("1", "true", "yes", "y"):
        return ParsedFields()

    fields = ParsedFields()

    # Prefer system temp dir; allow override via KYC_TMPDIR if needed
    tmp_dir = os.getenv("KYC_TMPDIR", None)
    fd, path = tempfile.mkstemp(suffix=".jpg", dir=tmp_dir)
    os.close(fd)  # Important on Windows: release the handle before cv2.imwrite

    try:
        ok = cv2.imwrite(path, id_front_bgr)
        if not ok:
            return fields

        try:
            mrz = read_mrz(path)
        except Exception as e:
            # Log and fall back to OCR-only without crashing the API
            import sys, traceback
            print("[MRZ] read_mrz failed:", e, file=sys.stderr)
            traceback.print_exc()
            return fields

        if mrz is None:
            return fields

        mrz_data = mrz.to_dict()
        surname = mrz_data.get("surname")
        given = mrz_data.get("names")
        if surname or given:
            fields.full_name = " ".join([given or "", surname or ""]).strip().title()

        dob = mrz_data.get("date_of_birth")
        if dob and len(dob) == 6:
            yy, mm, dd = dob[:2], dob[2:4], dob[4:]
            century = "19" if int(yy) > 30 else "20"
            fields.dob = f"{century}{yy}-{mm}-{dd}"

        fields.document_number = mrz_data.get("number") or mrz_data.get("document_number")
        fields.nationality = (mrz_data.get("nationality") or "").upper() or None

        exp = mrz_data.get("expiration_date")
        if exp and len(exp) == 6:
            yy, mm, dd = exp[:2], exp[2:4], exp[4:]
            century = "19" if int(yy) > 30 else "20"
            fields.expiry_date = f"{century}{yy}-{mm}-{dd}"

        return fields
    finally:
        try:
            os.remove(path)
        except OSError:
            pass