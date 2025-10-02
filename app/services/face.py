from typing import Optional, Tuple
import os
import cv2
import numpy as np

from app.utils.models import ensure_model

# Correct raw URLs on the main branch (use raw.githubusercontent.com)
DEFAULT_YUNET_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",  # fallback if repo uses master
]
DEFAULT_SFACE_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]

# Allow overrides via env (e.g., private mirror or local file server)
ENV_YUNET_URL = os.getenv("YUNET_URL", "").strip()
ENV_SFACE_URL = os.getenv("SFACE_URL", "").strip()

# File names expected on disk
YUNET_FILE = os.getenv("YUNET_FILE", "face_detection_yunet_2023mar.onnx")
SFACE_FILE = os.getenv("SFACE_FILE", "face_recognition_sface_2021dec.onnx")


class FaceService:
    """
    Lightweight face detection + embedding using OpenCV YuNet + SFace.
    CPU-only, no InsightFace or ONNX Runtime needed.
    """
    def __init__(self, det_score_threshold: float = 0.6):
        yunet_urls = [ENV_YUNET_URL] if ENV_YUNET_URL else DEFAULT_YUNET_URLS
        sface_urls = [ENV_SFACE_URL] if ENV_SFACE_URL else DEFAULT_SFACE_URLS

        self.yunet_path = ensure_model(YUNET_FILE, yunet_urls)
        self.sface_path = ensure_model(SFACE_FILE, sface_urls)

        # Initialize models
        self.detector = cv2.FaceDetectorYN_create(
            self.yunet_path, "", (320, 320), det_score_threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF_create(self.sface_path, "")

    def detect_and_embed(self, img_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
        h, w = img_bgr.shape[:2]
        # YuNet requires setting the input size to the image size before detection
        self.detector.setInputSize((w, h))
        retval, faces = self.detector.detect(img_bgr)

        if faces is None or len(faces) == 0:
            return None, None

        # faces: Nx15, [x, y, w, h, ...]; choose the largest by area
        def _area(face_row):
            x, y, fw, fh = face_row[:4]
            return float(fw * fh)

        faces_sorted = sorted(faces, key=_area, reverse=True)
        face = faces_sorted[0]

        # Align and extract feature
        aligned = self.recognizer.alignCrop(img_bgr, face)
        feat = self.recognizer.feature(aligned)  # returns a vector-like ndarray
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feat))
        if norm < 1e-9:
            return None, None
        emb = feat / (norm + 1e-9)

        # Build bbox: [x, y, w, h] -> (x1, y1, x2, y2)
        x, y, fw, fh = face[:4]
        bbox = (int(x), int(y), int(x + fw), int(y + fh))
        return emb, bbox


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)