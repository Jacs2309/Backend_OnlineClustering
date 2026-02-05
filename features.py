import cv2
import numpy as np
import onnxruntime as ort
import mahotas
from skimage.feature import hog
import threading
import time

# ============================================================
# CONFIGURACIÓN GLOBAL ONNX (CNN - MobileNetV2)
# ============================================================
# Se utiliza una sesión ONNX global para evitar cargar el modelo
# repetidamente, reduciendo el tiempo de inferencia total.
ORT_SESSION = None
ORT_LOADED = False
ORT_LOCK = threading.Lock()

def preload_onnx_session():
    """
    Precarga el modelo ONNX en un hilo seguro (thread-safe).

    La función implementa un mecanismo de doble verificación
    (double-check locking) para evitar cargas duplicadas del modelo
    cuando múltiples hilos intentan acceder simultáneamente.

    El modelo utilizado es MobileNetV2 en formato ONNX.
    """
    global ORT_SESSION, ORT_LOADED
    if ORT_LOADED:
        return
    
    with ORT_LOCK:
        if ORT_LOADED:
            return
        try:
            print("[FEATURES] Precargando modelo ONNX MobileNetV2...")
            start = time.time()
            ORT_SESSION = ort.InferenceSession(
                "mobilenet_v2.onnx",
                providers=['CPUExecutionProvider']
            )
            ORT_LOADED = True
            print(f"[FEATURES] Modelo cargado en {time.time() - start:.2f}s")
        except Exception as e:
            print(f"[FEATURES] Error precargando ONNX: {e}")
            ORT_LOADED = False

def get_ort_session():
    """
    Retorna la sesión ONNX activa.

    Si el modelo aún no ha sido cargado, se inicializa automáticamente.
    """
    global ORT_SESSION, ORT_LOADED
    if not ORT_LOADED:
        preload_onnx_session()
    return ORT_SESSION

def extract_cnn(img):
    """
    Extrae características profundas usando MobileNetV2 (CNN).

    Parámetros
    ----------
    img : ndarray
        Imagen de entrada en formato OpenCV (BGR, RGB o escala de grises).

    Retorna
    -------
    ndarray
        Vector de características CNN (1280 dimensiones).
    """
    session = get_ort_session()
    if session is None:
        return np.zeros(1280, dtype=np.float32)

    # Conversión a RGB según el número de canales
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionamiento y normalización según MobileNetV2
    img_resized = cv2.resize(img, (224, 224)).astype(np.float32)
    img_resized = (img_resized / 127.5) - 1.0

    # Formato batch
    x = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # Ajuste automático de layout (NHWC → NCHW si es necesario)
    try:
        input_meta = session.get_inputs()[0]
        input_shape = input_meta.shape
        input_name = input_meta.name
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
            if input_shape[1] == 3:
                x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
    except Exception:
        input_name = session.get_inputs()[0].name

    # Inferencia
    outputs = session.run(None, {input_name: x})
    return np.asarray(outputs[0]).flatten().astype(np.float32)

# ============================================================
# MOMENTOS DE ZERNIKE (Mahotas)
# ============================================================
def extract_zernike(img, radius=21, degree=8):
    """
    Extrae momentos de Zernike para describir la forma de la imagen.

    Los momentos de Zernike son invariantes a rotación y escala,
    lo que los hace adecuados para reconocimiento de patrones.

    Parámetros
    ----------
    img : ndarray
        Imagen de entrada.
    radius : int
        Radio del círculo usado para el cálculo.
    degree : int
        Grado máximo de los polinomios de Zernike.

    Retorna
    -------
    ndarray
        Vector de características de forma.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    features = mahotas.features.zernike_moments(thresh, radius, degree)
    return features.astype(np.float32)

# ============================================================
# MOMENTOS DE HU
# ============================================================
def extract_hu_moments(img):
    """
    Extrae los siete momentos invariantes de Hu.

    Estos momentos son invariantes a rotación, escala y traslación,
    y se usan comúnmente en análisis de forma.

    Retorna
    -------
    ndarray
        Vector de 7 características normalizadas logarítmicamente.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)

# ============================================================
# SIFT (PROMEDIO DE DESCRIPTORES)
# ============================================================
def extract_sift(img):
    """
    Extrae características SIFT y retorna el promedio de descriptores.

    Este enfoque reduce la dimensionalidad y permite representar
    la imagen con un solo vector de 128 dimensiones.

    Retorna
    -------
    ndarray
        Vector SIFT promedio.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        return np.zeros(128, dtype=np.float32)

    return descriptors.mean(axis=0).astype(np.float32)

# ============================================================
# HOG (HISTOGRAM OF ORIENTED GRADIENTS)
# ============================================================
def extract_hog(img):
    """
    Extrae características HOG para capturar gradientes locales
    y estructura espacial de la imagen.

    Retorna
    -------
    ndarray
        Vector HOG normalizado.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (128, 128))
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features.astype(np.float32)

# ============================================================
# SELECTOR GENERAL DE CARACTERÍSTICAS
# ============================================================
def extract_features(img, mode="hu"):
    """
    Selector general de extracción de características.

    Parámetros
    ----------
    img : ndarray
        Imagen de entrada.
    mode : str
        Tipo de descriptor: 'hu', 'zernike', 'sift', 'hog', 'cnn'.

    Retorna
    -------
    ndarray
        Vector de características correspondiente al método seleccionado.
    """
    mode = mode.lower()
    if mode == "hu": return extract_hu_moments(img)
    if mode == "zernike": return extract_zernike(img)
    if mode == "sift": return extract_sift(img)
    if mode == "hog": return extract_hog(img)
    if mode == "cnn": return extract_cnn(img)
    raise ValueError(f"Extractor desconocido: {mode}")
