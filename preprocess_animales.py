import cv2
import numpy as np

# =========================
# CONFIG
# =========================
TARGET_SIZE = (224, 224)
CONTRAST_MODE = "GLOBAL"
GLOBAL_CLAHE_CLIP = 2.0
GLOBAL_CLAHE_TILE = (8, 8)
DENOISE_MODE = "NLM"

# =========================
# UTILIDADES
# =========================
def analyze_stats(gray: np.ndarray) -> dict:
    g = gray.astype(np.uint8)
    mn, mx = int(g.min()), int(g.max())
    return {
        "min": mn,
        "max": mx,
        "range": mx - mn,
        "mean": float(np.mean(g)),
        "std": float(np.std(g))
    }

def denoise_bgr(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.upper()
    if mode == "GAUSSIAN":
        return cv2.GaussianBlur(img_bgr, (5, 5), 0)
    if mode == "MEDIAN":
        return cv2.medianBlur(img_bgr, 3)
    if mode == "NLM":
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, 8, 8, 7, 21)
    return img_bgr

def apply_clahe_luminance(img_bgr: np.ndarray, clip, tile) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def apply_contrast_bgr(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    stats = analyze_stats(gray)

    if CONTRAST_MODE.upper() == "GLOBAL":
        out = apply_clahe_luminance(
            img_bgr,
            GLOBAL_CLAHE_CLIP,
            GLOBAL_CLAHE_TILE
        )
        tech = f"LAB_CLAHE({GLOBAL_CLAHE_CLIP},{GLOBAL_CLAHE_TILE})"
        return out, tech, stats

    return img_bgr, "NO_CONTRAST", stats

# =========================
# SEGMENTACIÓN
# =========================
def grabcut_rect(img_bgr, rect_scale=0.1, iters=5):
    h, w = img_bgr.shape[:2]
    x, y = int(rect_scale*w), int(rect_scale*h)
    rw, rh = int((1-2*rect_scale)*w), int((1-2*rect_scale)*h)
    rect = (x, y, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgModel, fgModel, iters, cv2.GC_INIT_WITH_RECT)

    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0
    ).astype("uint8")

def refine_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    return mask

def keep_largest_component(mask, min_area_ratio=0.02):
    h, w = mask.shape
    min_area = int(min_area_ratio * h * w)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )

    best_id, best_area = -1, 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_id, best_area = i, area

    if best_id == -1:
        return mask

    out = np.zeros_like(mask)
    out[labels == best_id] = 255
    return out

def remove_background(img_bgr: np.ndarray, cls_es: str):
    if cls_es == "elefante":
        mask = grabcut_rect(img_bgr, 0.03, 10)
        mask = refine_mask(mask)
        mask = keep_largest_component(mask)
    else:
        mask = grabcut_rect(img_bgr, 0.1, 5)
        mask = refine_mask(mask)

    fg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return fg, mask

# =========================
# PIPELINE COMPACTO FINAL
# =========================
def preprocess_image_compact(img_bgr: np.ndarray, cls_es: str):
    """
    Pipeline resumido.
    Retorna únicamente:
    - mask: máscara binaria (uint8, 0/255)
    """
    if img_bgr is None:
        raise ValueError("Imagen None")

    # Resize
    img = cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Denoise
    img = denoise_bgr(img, DENOISE_MODE)

    # Contraste
    proc_bgr, _, _ = apply_contrast_bgr(img)

    # Segmentación
    _, mask = remove_background(proc_bgr, cls_es)

    return mask

