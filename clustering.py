import numpy as np
from sklearn.cluster import KMeans

class OnlineKMeansSizeConstrained:
    def __init__(self, k, dim, max_sizes, init_buffer_size=None, eta0=0.3, random_state=42):
        self.k = k
        self.dim = dim
        ms = np.atleast_1d(max_sizes)
        if len(ms) == 1:
            self.max_sizes = np.full(k, ms[0], dtype=int)
        elif len(ms) != k:
            # Fallback: si la lista no coincide, tomamos el primer elemento y lo repetimos
            self.max_sizes = np.full(k, ms[0], dtype=int)
        else:
            self.max_sizes = ms.astype(int)
        
        
        self.eta0 = eta0
        self.random_state = random_state
        self.init_buffer_size = init_buffer_size or (5 * k)
        # Buffer para inicialización
        self.init_buffer = []
        self.init_indices = []
        self.init_labels = []      # Buffer temporal de etiquetas reales
        #Estado del clustering
        self.centroids = None
        self.cluster_sizes = np.zeros(k, dtype=int)
        self.initialized = False
        #Metricas de evaluación
        self.cluster_members_ = {i: [] for i in range(k)} #guarda que puntos van a cada cluster
        self.features_list = [] # guarda todos las asginaciones de clusters
        self.total_assigned = 0
        self.assigned_true_labels = [] # Etiquetas reales de puntos ya asignados a clusters
        self.labels_ = []

    def _assign_point(self, x, idx, true_label):
        available_clusters = np.where(self.cluster_sizes < self.max_sizes)[0]
        if len(available_clusters) == 0:
            return None
        #calcular distancias a centroides disponibles
        distances = np.linalg.norm(self.centroids[available_clusters] - x, axis=1)
        target_cluster = available_clusters[np.argmin(distances)]
        #actualizar centroide y tamaño
        self.cluster_sizes[target_cluster] += 1
        eta = self.eta0 / (1 + self.cluster_sizes[target_cluster])
        self.centroids[target_cluster] += eta * (x - self.centroids[target_cluster])

        #actualizar miembros del cluster
        #self.cluster_members_[target_cluster].append(idx)
        clean_idx = int(idx) if (idx is not None and idx != "temp") else int(self.total_assigned)
        self.cluster_members_[target_cluster].append(clean_idx)
        self.total_assigned += 1

        # Sincronización de métricas externas
        self.features_list.append(x)
        self.labels_.append(int(target_cluster))
        self.assigned_true_labels.append(true_label)

        return target_cluster

    def partial_fit(self, x, idx=None, true_label=None):
        if not self.initialized:
            self.init_buffer.append(x)
            self.init_indices.append(idx if idx is not None else "temp")
            self.init_labels.append(true_label)

            if len(self.init_buffer) >= self.init_buffer_size:
                # 1. Inicializar centroides
                X_buf = np.array(self.init_buffer)
                km = KMeans(n_clusters=self.k, init="k-means++", n_init=1, random_state=self.random_state)
                km.fit(X_buf)
                self.centroids = km.cluster_centers_
                self.initialized = True

                # 2. Asignar retroactivamente los puntos del buffer
                buffer_assignments = []
                for b_x, b_idx, b_label in zip(self.init_buffer, self.init_indices, self.init_labels):
                    buffer_assignments.append(self._assign_point(b_x, b_idx, b_label))

                self.init_buffer = [] # Limpiar
                self.init_labels = []
                return buffer_assignments # Retorna lista de IDs

            return "pending" # Marcador temporal

        return self._assign_point(x, idx, true_label)