import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
#import mahotas

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CNN EMBEDDINGS (MobileNetV2)
# =========================
# Cargamos el modelo globalmente (incluyendo pesos de ImageNet)
# include_top=False elimina las capas densas de clasificación final
CNN_MODEL = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_cnn(img):
    # 1. Asegurar que la imagen esté en formato RGB (3 canales)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # 2. Redimensionar al tamaño que espera la red (224x224)
    img_resized = cv2.resize(img, (224, 224))
    
    # 3. Preparar para el modelo (añadir dimensión de batch y preprocesar)
    x = np.expand_dims(img_resized, axis=0)
    x = preprocess_input(x)
    
    # 4. Extraer embedding (vector de 1280 elementos)
    features = CNN_MODEL.predict(x, verbose=0)
    
    return features.flatten().astype(np.float32)

# =========================
# MOMENTOS DE HU
# =========================
def extract_hu_moments(img):
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


# =========================
# SIFT (vector fijo)
# =========================
def extract_sift(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=128)
    keypoints, descriptors = sift.detectAndCompute(img, None)

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

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    return features.astype(np.float32)

# =========================
# SELECTOR GENERAL
# =========================
def extract_features(img, mode="hu"):
    """
    mode:  | 'hu' |  | 'sift' | 'hog'
    """
    mode = mode.lower()

    if mode == "hu":
        return extract_hu_moments(img)

    if mode == "sift":
        return extract_sift(img)

    if mode == "hog":
        return extract_hog(img)
    
    if mode == "cnn":
        return extract_cnn(img)

    raise ValueError(f"Extractor desconocido: {mode}")
