    import gc
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import numpy as np
    import cv2

    from preprocess import preprocess_image
    from features import extract_features
    from clustering import OnlineKMeansSizeConstrained
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,silhouette_score
    from scipy.spatial.distance import cdist, pdist

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


    clusterings = {ext: None for ext in EXTRACTORS}

    # ======================================================
    # Utils
    # ======================================================
    def read_image(file):
        try:
            file.seek(0)
            img_bytes = np.frombuffer(file.read(), np.uint8)
            if img_bytes.size == 0: return None
            return cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        except:
            return None

    def dunn_index(X, labels, centroids):
        clusters = np.unique(labels)
        if len(clusters) < 2: return 0
        
        # 1. Inter-cluster: Distancia mínima entre centroides
        inter_dist = pdist(centroids).min() if len(centroids) > 1 else 0
        
        # 2. Intra-cluster: Diámetro máximo (punto más alejado de su centroide)
        max_diameter = 0
        for i, c_idx in enumerate(clusters):
            points = X[labels == c_idx]
            if len(points) > 0:
                # Distancia de cada punto a su centroide
                dists = cdist(points, centroids[i:i+1])
                max_diameter = max(max_diameter, dists.max() * 2) # Estimación del diámetro
                
        return float(inter_dist / max_diameter) if max_diameter > 0 else 0

    # ======================================================
    # Endpoint principal
    # ======================================================
    @app.route("/process", methods=["POST"])
    def process_batch():
        if "images" not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        data = request.get_json()
        # Obtener el modo del frontend (por defecto 'hu')
        mode = request.form.get("mode", "hu").lower()
        # Recibir listas de archivos y etiquetas
        files = request.files.getlist("images")
        labels = request.form.getlist("labels")
        
        batch_results = []

        if clusterings.get(mode) is None:
            # Probamos una extracción rápida para conocer la dimensión
            test_img = np.zeros((224,224,3), dtype=np.uint8)
            test_feat = extract_features(test_img, mode=mode)
            clusterings[mode] = OnlineKMeansSizeConstrained(
                k=K, dim=test_feat.shape[0], max_sizes=tamaños_máximos,
                init_buffer_size=5 * K, random_state=seed
            )

        for i, file in enumerate(files):
            img_raw = read_image(file)
            if img_raw is None: continue

            if mode in ["hu", "zernike"]:
                img_gray = preprocess_image(img_raw)
                features = extract_features(img_gray, mode=mode)
            else:
                features = extract_features(img_raw, mode=mode)

            cluster_id = clusterings[mode].partial_fit(features, true_label=labels[i] if i < len(labels) else None)

            # Formatear respuesta
            final_id = -1
            if isinstance(cluster_id, (int, np.integer)): final_id = int(cluster_id)
            elif isinstance(cluster_id, list) and len(cluster_id) > 0: final_id = int(cluster_id[-1])

            batch_results.append({
                "filename": file.filename,
                mode: {
                    "cluster": final_id,
                    "cluster_sizes": clusterings[mode].cluster_sizes.tolist()
                }
            })
            
            # Limpieza agresiva de RAM
            del img_raw, features
        
        gc.collect()
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
            # Evitar procesar si hay muy pocos datos (mínimo 2 clusters con datos)
            if model and len(model.labels_) > K and len(np.unique(model.labels_)) > 1:
                X = np.array(model.features_list)
                y_true = np.array(model.assigned_true_labels)
                y_pred = np.array(model.labels_)
                
                # Muestreo si X es demasiado grande (Evita OOM en Render)
                # Si hay más de 500 puntos, calculamos silueta con una muestra
                if len(X) > 500:
                    indices = np.random.choice(len(X), 500, replace=False)
                    X_sub, y_sub = X[indices], y_pred[indices]
                    sil = silhouette_score(X_sub, y_sub)
                else:
                    sil = silhouette_score(X, y_pred)

                evaluation[mode] = {
                    "ari": round(float(adjusted_rand_score(y_true, y_pred)), 4),
                    "nmi": round(float(normalized_mutual_info_score(y_true, y_pred)), 4),
                    "silhouette": round(float(sil), 4),
                    "dunn": round(dunn_index(X, y_pred, model.centroids), 4),
                    "samples": int(len(y_pred)),
                    "distribution": model.cluster_sizes.tolist()
                }
            else:
                evaluation[mode] = {"error": "Esperando más datos..."}
                
        return jsonify(evaluation)

    # ======================================================
    # Endpoint Reset
    # ======================================================
    @app.route("/reset", methods=["POST"])
    def reset_backend():
        global clusterings, k, MAX_CLUSTER_SIZES, seed
        data = request.get_json()
        
        # Extraemos y actualizamos la configuración global
        K = int(data.get("k", 4))
        seed = int(data.get("seed", 42))
        
        # Recibimos la lista de tamaños dinámicos
        incoming_sizes = data.get("max_size")
        if incoming_sizes and len(incoming_sizes) == K:
            MAX_CLUSTER_SIZES = [int(s) for s in incoming_sizes]
        
        # Limpiamos los modelos para forzar la reinicialización con los nuevos tamaños
        clusterings = {ext: None for ext in EXTRACTORS}
        gc.collect()
        
        return jsonify({
            "status": "success", 
            "message": f"Reiniciado"
        })
# ======================================================
# Main
# ======================================================
    if __name__ == "__main__":
        # En local puerto 5001, en Render/HF se suele usar variable de entorno PORT
        import os
        port = int(os.environ.get("PORT", 5001))
        app.run(host="0.0.0.0", port=port)
    