import io
import os
import re
import tempfile
from typing import Optional, Tuple, List

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import KycResult, ExtractedFields, Scores, SanctionsMatch
from app.config import Settings, get_settings
from app.utils.image import read_upload_as_bgr, to_pil
from app.services.ocr import extract_text_fields, try_mrz_parse
from app.services.face import FaceService, cosine_similarity
from app.services.liveness import liveness_heuristics
from app.services.sanctions import query_opensanctions

settings: Settings = get_settings()
app = FastAPI(title="KYC MVP", version="0.1.0")

# Static demo form
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# Global model singletons (lazy init)
_face_service: Optional[FaceService] = None


def face_service() -> FaceService:
    global _face_service
    if _face_service is None:
        _face_service = FaceService()
    return _face_service


@app.get("/", response_class=HTMLResponse)
def root():
    return '<html><head><meta http-equiv="refresh" content="0; url=/web/index.html"></head><body></body></html>'


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Log server-side
    import traceback, sys
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"[ERROR] {request.method} {request.url}\n{tb}", file=sys.stderr)
    # Return JSON so the frontend can display it
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )


@app.post("/api/kyc/verify", response_model=KycResult)
async def kyc_verify(
    id_front: UploadFile = File(...),
    selfie: UploadFile = File(...),
    id_back: Optional[UploadFile] = File(None),
    full_name: Optional[str] = Form(None),
    dob: Optional[str] = Form(None),  # YYYY-MM-DD preferred
):
    # Read images
    try:
        id_front_img = await read_upload_as_bgr(id_front)
        selfie_img = await read_upload_as_bgr(selfie)
        id_back_img = await read_upload_as_bgr(id_back) if id_back else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")

    # 1) OCR + MRZ
    mrz_fields = try_mrz_parse(id_front_img)
    ocr_fields, ocr_conf = extract_text_fields(id_front_img, id_back_img)

    extracted = ExtractedFields(
        full_name=mrz_fields.full_name or ocr_fields.full_name or full_name,
        dob=mrz_fields.dob or ocr_fields.dob or dob,
        document_number=mrz_fields.document_number or ocr_fields.document_number,
        nationality=mrz_fields.nationality or ocr_fields.nationality,
        expiry_date=mrz_fields.expiry_date or ocr_fields.expiry_date,
        address=ocr_fields.address,
        source="mrz" if mrz_fields.document_number else "ocr",
    )

    # 2) Face extraction and verification
    fs = face_service()

    id_face_embed, id_face_bbox = fs.detect_and_embed(id_front_img)
    if id_face_embed is None:
        raise HTTPException(status_code=422, detail="Could not detect a face on the ID image.")

    selfie_embed, selfie_bbox = fs.detect_and_embed(selfie_img)
    if selfie_embed is None:
        raise HTTPException(status_code=422, detail="Could not detect a face on the selfie image.")

    face_match_score = float(cosine_similarity(id_face_embed, selfie_embed))  # 0..1 approx

    # 3) Basic liveness heuristic on selfie
    live_score = liveness_heuristics(selfie_img)

    # 4) Sanctions/PEP screening (OpenSanctions)
    sanctions_matches: List[SanctionsMatch] = []
    sanctions_score = 0.0
    sanctions_flag = False
    try:
        if extracted.full_name:
            sanctions_matches = query_opensanctions(
                name=extracted.full_name,
                birth_date=extracted.dob,
                api_key=settings.OPEN_SANCTIONS_API_KEY,
                top_k=5,
            )
            if len(sanctions_matches) > 0:
                top = sanctions_matches[0]
                sanctions_score = top.score or 0.0
                sanctions_flag = sanctions_score >= settings.SANCTIONS_FLAG_THRESHOLD
    except Exception:
        sanctions_matches = []
        sanctions_score = 0.0
        sanctions_flag = False

    # 5) Risk aggregation (toy scoring)
    face_component = max(0.0, min(1.0, (face_match_score - settings.FACE_PASS_THRESHOLD + 0.2) / 0.2))
    live_component = max(0.0, min(1.0, live_score))
    ocr_component = max(0.0, min(1.0, ocr_conf))
    sanctions_component = max(0.0, min(1.0, 1.0 - sanctions_score))

    overall = (
        0.55 * face_component
        + 0.20 * live_component
        + 0.15 * ocr_component
        + 0.10 * sanctions_component
    )

    passed = (
        face_match_score >= settings.FACE_PASS_THRESHOLD
        and live_score >= settings.LIVENESS_PASS_THRESHOLD
        and not sanctions_flag
    )

    return KycResult(
        extracted=extracted,
        scores=Scores(
            face_match=face_match_score,
            liveness=live_score,
            ocr_confidence=ocr_conf,
            sanctions_match=sanctions_score,
            overall=overall,
        ),
        sanctions_matches=sanctions_matches,
        passed=bool(passed),
        reason="sanctions_flag" if sanctions_flag else ("low_face_match" if face_match_score < settings.FACE_PASS_THRESHOLD else ("low_liveness" if live_score < settings.LIVENESS_PASS_THRESHOLD else "ok")),
    )