import cv2
import numpy as np

# =========================
# CONFIG (puedes versionar luego)
# =========================
TARGET_SIZE = (224, 224)
CONTRAST_MODE = "GLOBAL"
GLOBAL_CLAHE_CLIP = 2.0
GLOBAL_CLAHE_TILE = (8, 8)
DENOISE_MODE = "NLM"

# =========================
# UTILIDADES
# =========================
def to_gray(img):
    if img is None:
        raise ValueError("Imagen None")
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def minmax_0_255(img):
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return img.astype(np.uint8)
    out = (img - mn) * (255.0 / (mx - mn))
    return out.astype(np.uint8)

def apply_clahe(img, clip=2.0, tile=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img)

def denoise(img, mode):
    if mode == "GAUSSIAN":
        return cv2.GaussianBlur(img, (5, 5), 0)
    if mode == "MEDIAN":
        return cv2.medianBlur(img, 3)
    if mode == "NLM":
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    return img

# =========================
# PIPELINE PRINCIPAL
# =========================
def preprocess_image(img):
    """
    img: np.ndarray BGR o gris
    return: imagen procesada (uint8)
    """
    gray = to_gray(img)
    gray = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.uint8)

    gray = denoise(gray, DENOISE_MODE)

    if CONTRAST_MODE == "GLOBAL":
        gray = minmax_0_255(gray)
        gray = apply_clahe(gray, GLOBAL_CLAHE_CLIP, GLOBAL_CLAHE_TILE)

    return gray
