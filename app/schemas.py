from typing import Optional, List
from pydantic import BaseModel


class ExtractedFields(BaseModel):
    full_name: Optional[str] = None
    dob: Optional[str] = None  # YYYY-MM-DD where possible
    document_number: Optional[str] = None
    nationality: Optional[str] = None
    expiry_date: Optional[str] = None
    address: Optional[str] = None
    source: str = "ocr"  # "mrz" or "ocr"


class Scores(BaseModel):
    face_match: float
    liveness: float
    ocr_confidence: float
    sanctions_match: float
    overall: float


class SanctionsMatch(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    country: Optional[str] = None
    dataset: Optional[str] = None
    schema: Optional[str] = None
    score: Optional[float] = None
    link: Optional[str] = None


class KycResult(BaseModel):
    extracted: ExtractedFields
    scores: Scores
    sanctions_matches: List[SanctionsMatch]
    passed: bool
    reason: str