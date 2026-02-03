from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

from preprocess import preprocess_image
from features import extract_features
from clustering import OnlineKMeansSizeConstrained
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,silhouette_score
from scipy.spatial.distance import cdist


app = Flask(__name__)
CORS(app, resources={ r".*": {"origins": "*"}})
# ======================================================
# Configuración de extractores
# ======================================================
EXTRACTORS = ["hu", "sift","zernike" ,"hog", "cnn"]
#EXTRACTORS = ["cnn"]
K = 4
MAX_CLUSTER_SIZES = [300, 306, 405, 300]
seed = 42

# ======================================================
# Modelos de clustering (stateful)
# ======================================================
clusterings = {
    # dim fija conocida
    "hu": OnlineKMeansSizeConstrained(k=K, dim=7, max_sizes=MAX_CLUSTER_SIZES,init_buffer_size=5 * K),
    "sift": OnlineKMeansSizeConstrained(k=K, dim=128, max_sizes=MAX_CLUSTER_SIZES,init_buffer_size=5 * K),

    # dim dinámica → se inicializa luego
    "hog": None,
    "zernike": None,
    "cnn": None
}

# ======================================================
# Utils
# ======================================================
def read_image(file):
    img_bytes = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

def dunn_index(X, labels):
    clusters = np.unique(labels)
    if len(clusters) < 2: return 0
    
    # Mínima distancia entre clusters (Inter-cluster)
    inter_cluster = np.inf
    for i in clusters:
        for j in clusters:
            if i < j:
                dist = cdist(X[labels == i], X[labels == j]).min()
                inter_cluster = min(inter_cluster, dist)

    # Máximo diámetro intra-cluster
    intra_cluster = 0
    for c in clusters:
        points = X[labels == c]
        if len(points) > 1:
            intra_cluster = max(intra_cluster, cdist(points, points).max())
    
    return inter_cluster / intra_cluster if intra_cluster > 0 else 0

# ======================================================
# Endpoint principal
# ======================================================
@app.route("/process", methods=["POST"])
def process_batch():
    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400

    # Recibir listas de archivos y etiquetas
    files = request.files.getlist("images")
    labels = request.form.getlist("labels")
    
    batch_results = []

    for i, file in enumerate(files):
        img_raw = read_image(file)
        img_proc = None
        current_img_results = {"filename": file.filename}
        true_label = labels[i] if i < len(labels) else None

        for mode in EXTRACTORS:
            # Preprocesamiento inteligente
            if mode in ["hu", "geom", "zernike"]:
                if img_proc is None:
                    img_proc = preprocess_image(img_raw)
                img_to_use = img_proc
            else:
                img_to_use = img_raw

            # 1. Extracción
            features = extract_features(img_to_use, mode=mode)
            dim = features.shape[0]

            # 2. Lazy Init
            if clusterings[mode] is None:
                clusterings[mode] = OnlineKMeansSizeConstrained(
                    k=K, dim=dim, max_sizes=MAX_CLUSTER_SIZES,
                    init_buffer_size=10 * K, random_state=seed
                )

            # 3. Fit
            cluster_id = clusterings[mode].partial_fit(features, true_label=true_label)

            # Formatear ID
            if isinstance(cluster_id, list):
                final_id = int(cluster_id[-1]) if len(cluster_id) > 0 else -1
            elif cluster_id == "pending": final_id = -1
            elif cluster_id is None: final_id = -2
            else: final_id = int(cluster_id)

            current_img_results[mode] = {
                "cluster": final_id,
                "cluster_sizes": clusterings[mode].cluster_sizes.tolist()
            }
        
        batch_results.append(current_img_results)

    return jsonify({
        "status": "ok",
        "results": batch_results # Ahora es una lista de resultados
    })
# ======================================================
# Endpoint Metricas
# ======================================================
@app.route("/metrics", methods=["GET"])
def get_metrics():
    evaluation = {}
    
    for mode in EXTRACTORS:
        model = clusterings[mode]
        
        # Solo evaluamos si el modelo ya procesó datos más allá del buffer
        if model and len(model.labels_) > 0:
            X = np.array(model.features_list)
            y_true = np.array(model.assigned_true_labels)
            y_pred = np.array(model.labels_)
            
            # Métricas Externas
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)
            ami = adjusted_mutual_info_score(y_true, y_pred)
            # Métricas Internas
            sil = silhouette_score(X, y_pred) if len(np.unique(y_pred)) > 1 else 0
            dunn = dunn_index(X, y_pred)

            distribution = model.cluster_sizes.tolist()
            
            
            evaluation[mode] = {
                "ari": round(float(ari), 4),
                "nmi": round(float(nmi), 4),
                "ami": round(float(ami), 4),
                "silhouette": round(float(sil), 4),
                "dunn": round(float(dunn), 4),
                "samples": int(len(y_pred)),
                "distribution": distribution
            }
        else:
            evaluation[mode] = {"error": "Sin datos suficientes"}
            
    return jsonify(evaluation)

# ======================================================
# Endpoint Reset
# ======================================================
@app.route("/reset", methods=["POST"])
def reset_backend():
    global clusterings, K, MAX_CLUSTER_SIZES, seed
    
    # Obtener parámetros del body si existen
    data = request.get_json()
    if data:
        K = int(data.get("k", K))
        new_max = data.get("max_size")
        if new_max is not None:
            if isinstance(new_max, list):
                # Aseguramos que todos sean int
                MAX_CLUSTER_SIZES = [int(x) for x in new_max]
            else:
                # Si llega un solo número, lo repetimos K veces
                MAX_CLUSTER_SIZES = [int(new_max)] * K
        seed = int(data.get("seed", seed))

    # Volvemos a inicializar el diccionario de modelos
    clusterings = {
        "hu": OnlineKMeansSizeConstrained(k=K, dim=7, max_sizes=MAX_CLUSTER_SIZES, init_buffer_size=5 * K, random_state=seed),
        "sift": OnlineKMeansSizeConstrained(k=K, dim=128, max_sizes=MAX_CLUSTER_SIZES, init_buffer_size=5 * K, random_state=seed),
        "hog": None,
        "zernike": None,
        "cnn": None
    }
    return jsonify({"status": "success", "message": f"Backend reiniciado: K={K}, Tamaños={MAX_CLUSTER_SIZES}, Seed={seed}"})
# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    app.run(port=5001, debug=True)
