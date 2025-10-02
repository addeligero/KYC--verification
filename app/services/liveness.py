import cv2
import numpy as np


def liveness_heuristics(img_bgr: np.ndarray) -> float:
    """
    Very basic, non-robust heuristics that correlate loosely with live captures:
    - Sharpness (variance of Laplacian)
    - Color presence (saturation)
    - Moiré/screen pattern penalty (FFT high-frequency bias)
    Returns a 0..1 score. This is NOT spoof-resistant; use proper anti-spoofing in production.
    """
    # Sharpness
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp = min(1.0, fm / 150.0)  # normalize

    # Saturation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    sat_mean = sat.mean() / 255.0

    # FFT-based moire penalty
    # High HF energy relative to total could indicate screen recapture
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    central = magnitude[cy - 15:cy + 15, cx - 15:cx + 15].sum() + 1e-6
    total = magnitude.sum() + 1e-6
    hf_ratio = 1.0 - (central / total)
    moire_penalty = min(1.0, hf_ratio * 1.5)

    # Combine
    score = (0.6 * sharp) + (0.3 * sat_mean) - (0.3 * moire_penalty)
    return float(max(0.0, min(1.0, score)))