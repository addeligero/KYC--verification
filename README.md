# KYC MVP (free, non-compliance-grade)

A minimal KYC pipeline using free/open-source components:

- FastAPI REST endpoint: upload ID front (and optional back) + selfie
- OCR: Tesseract (pytesseract)
- MRZ parsing: PassportEye + python-mrz
- Face verification: OpenCV YuNet (detection) + SFace (embeddings) — lightweight CPU-only
- Basic liveness heuristic: sharpness/saturation/FFT — not spoof-resistant
- Sanctions/PEP screening: OpenSanctions API

This is not compliance-grade. Use at your own risk.

## Quick start

### Docker

```bash
docker build -t kyc-mvp .
docker run --rm -p 8000:8000 kyc-mvp
```

Open http://localhost:8000/web to use the simple form.

### Local (Python 3.11)

Install Tesseract on your OS (Debian/Ubuntu: `sudo apt-get install tesseract-ocr`, macOS: `brew install tesseract`), then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/web

## Models

On first run, the service will download two ONNX models from the OpenCV Zoo into `./models/`:

- YuNet (face detector)
- SFace (face embedding)

The downloader now understands Git LFS pointers and will automatically retry through GitHub's `media.githubusercontent.com` endpoint, which fixes the "downloaded file too small" failure that can happen in headless environments. You can still provide your own mirror via environment variables:

- `YUNET_URL` / `SFACE_URL` – override the download URL(s)
- `YUNET_FILE` / `SFACE_FILE` – override the expected filenames under `models/`

If outbound network access is completely blocked, fetch the files manually and place them in `models/`:

- face_detection_yunet_2023mar.onnx from OpenCV Zoo
- face_recognition_sface_2021dec.onnx from OpenCV Zoo

## Configuration

Environment variables:

- FACE_PASS_THRESHOLD: default `0.35` (cosine similarity)
- LIVENESS_PASS_THRESHOLD: default `0.35`
- OPEN_SANCTIONS_API_KEY: optional
- SANCTIONS_TOPK: default `5`
- SANCTIONS_FLAG_THRESHOLD: default `0.85`

## Notes and limitations

- OpenCV YuNet + SFace are lightweight and easy to install but not as strong as production-grade SDKs.
- Liveness is a simple heuristic and not robust against spoofing.
- Handle PII carefully: secure storage, encryption at rest, strict access, and logs.

## Next improvements (optional)

- Replace liveness with a real anti-spoof model (e.g., Silent-Face-Anti-Spoofing)
- Add selfie video capture + active challenge
- Use PaddleOCR for stronger OCR and field detection
- Add address normalization (libpostal)
- Persist artifacts (Postgres + S3/MinIO) and audit logs
