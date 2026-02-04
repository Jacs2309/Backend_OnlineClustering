import cv2
import numpy as np
import onnxruntime as ort
import mahotas
from skimage.feature import hog
import threading
import time

# =========================
# CONFIGURACIÓN ONNX (CNN)
# =========================
# Cargamos la sesión de ONNX una sola vez globalmente con precarga
ORT_SESSION = None
ORT_LOADED = False
ORT_LOCK = threading.Lock()

def preload_onnx_session():
    """Precarga el modelo ONNX en background thread"""
    global ORT_SESSION, ORT_LOADED
    if ORT_LOADED:
        return
    
    with ORT_LOCK:
        if ORT_LOADED:  # Double-check locking
            return
        try:
            print("[FEATURES] Precargando modelo ONNX MobileNetV2...")
            start = time.time()
            ORT_SESSION = ort.InferenceSession("mobilenet_v2.onnx", providers=['CPUExecutionProvider'])
            ORT_LOADED = True
            print(f"[FEATURES] Modelo cargado en {time.time() - start:.2f}s")
        except Exception as e:
            print(f"[FEATURES] Error precargando ONNX: {e}")
            ORT_LOADED = False

def get_ort_session():
    global ORT_SESSION, ORT_LOADED
    if not ORT_LOADED:
        preload_onnx_session()
    return ORT_SESSION

def extract_cnn(img):
    session = get_ort_session()  # Solo carga el modelo si se llama a esta función
    if session is None:
        return np.zeros(1280, dtype=np.float32)

    # 1. Ajuste de canales: convertir a RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        # OpenCV carga BGR por defecto, convertir a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Redimensionar y Preprocesar (Escala MobileNet: [-1, 1])
    img_resized = cv2.resize(img, (224, 224)).astype(np.float32)
    img_resized = (img_resized / 127.5) - 1.0

    # 3. Formato de batch
    x = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # 4. Adaptar layout si el modelo espera NCHW (canales primero)
    try:
        input_meta = session.get_inputs()[0]
        input_shape = input_meta.shape
        input_name = input_meta.name
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
            # Si la segunda dimensión es 3, el modelo espera (N, C, H, W)
            if input_shape[1] == 3:
                x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
    except Exception:
        # Fallback: obtener nombre de input si algo falla
        input_name = session.get_inputs()[0].name

    # 5. Inferencia
    outputs = session.run(None, {input_name: x})
    return np.asarray(outputs[0]).flatten().astype(np.float32)

# =========================
# MOMENTOS DE ZERNIKE (Mahotas)
# =========================
def extract_zernike(img, radius=21, degree=8):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Umbralización simple para resaltar la forma
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Mahotas calcula los momentos basándose en el radio y grado especificado
    # Retorna un vector de características de forma eficiente
    features = mahotas.features.zernike_moments(thresh, radius, degree)
    return features.astype(np.float32)

# =========================
# MOMENTOS DE HU
# =========================
def extract_hu_moments(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    # Escala logarítmica para evitar valores extremadamente pequeños
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)

# =========================
# SIFT (Vector promedio)
# =========================
def extract_sift(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        return np.zeros(128, dtype=np.float32)

    return descriptors.mean(axis=0).astype(np.float32)

# =========================
# HOG
# =========================
def extract_hog(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (128, 128))
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    return features.astype(np.float32)

# =========================
# SELECTOR GENERAL
# =========================
def extract_features(img, mode="hu"):
    mode = mode.lower()
    if mode == "hu": return extract_hu_moments(img)
    if mode == "zernike": return extract_zernike(img)
    if mode == "sift": return extract_sift(img)
    if mode == "hog": return extract_hog(img)
    if mode == "cnn": return extract_cnn(img)
    raise ValueError(f"Extractor desconocido: {mode}")