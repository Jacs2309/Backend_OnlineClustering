import numpy as np
from sklearn.cluster import KMeans


class OnlineKMeansSizeConstrained:
    """
    Implementación de K-Means online (incremental) con restricción
    de tamaño máximo por cluster.

    - Los puntos se procesan uno a uno (streaming).
    - Cada cluster tiene un número máximo de puntos permitidos.
    - La inicialización de centroides se hace usando un buffer inicial
      y KMeans clásico.
    """

    def __init__(self, k, dim, max_sizes, init_buffer_size=None, eta0=0.3, random_state=42):
        """
        Parámetros
        ----------
        k : int
            Número de clusters.
        dim : int
            Dimensionalidad de los datos.
        max_sizes : int o array-like
            Tamaño máximo permitido por cluster.
            - Si es un escalar, se aplica el mismo límite a todos los clusters.
            - Si es una lista, debe tener longitud k.
        init_buffer_size : int, opcional
            Número de puntos usados para inicializar los centroides.
            Por defecto es 5 * k.
        eta0 : float
            Tasa de aprendizaje base para la actualización online
            de los centroides.
        random_state : int
            Semilla para reproducibilidad.
        """

        self.k = k
        self.dim = dim

        # Normalización del parámetro max_sizes
        ms = np.atleast_1d(max_sizes)
        if len(ms) == 1:
            self.max_sizes = np.full(k, ms[0], dtype=int)
        elif len(ms) != k:
            # Fallback: si la longitud no coincide, se repite el primer valor
            self.max_sizes = np.full(k, ms[0], dtype=int)
        else:
            self.max_sizes = ms.astype(int)

        self.eta0 = eta0
        self.random_state = random_state
        self.init_buffer_size = init_buffer_size or (5 * k)

        # Buffers para la fase de inicialización
        self.init_buffer = []       # Puntos iniciales
        self.init_indices = []      # Índices de los puntos
        self.init_labels = []       # Etiquetas reales (ground truth)

        # Estado del clustering
        self.centroids = None
        self.cluster_sizes = np.zeros(k, dtype=int)
        self.initialized = False

        # Métricas y tracking
        self.cluster_members_ = {i: [] for i in range(k)}  # Índices por cluster
        self.features_list = []        # Todos los puntos asignados
        self.labels_ = []              # Cluster asignado a cada punto
        self.assigned_true_labels = [] # Etiquetas reales de los puntos
        self.total_assigned = 0

    def _assign_point(self, x, idx, true_label):
        """
        Asigna un punto a un cluster disponible (que no haya alcanzado
        su tamaño máximo) y actualiza el centroide de forma incremental.

        Parámetros
        ----------
        x : np.ndarray
            Vector de características del punto.
        idx : int o None
            Identificador del punto.
        true_label : int o None
            Etiqueta real del punto (para métricas externas).

        Retorna
        -------
        int o None
            ID del cluster asignado o None si no hay clusters disponibles.
        """

        # Clusters que aún tienen capacidad
        available_clusters = np.where(self.cluster_sizes < self.max_sizes)[0]
        if len(available_clusters) == 0:
            return None

        # Distancia del punto a los centroides disponibles
        distances = np.linalg.norm(
            self.centroids[available_clusters] - x, axis=1
        )

        # Cluster más cercano
        target_cluster = available_clusters[np.argmin(distances)]

        # Actualización del tamaño del cluster
        self.cluster_sizes[target_cluster] += 1

        # Tasa de aprendizaje decreciente
        eta = self.eta0 / (1 + self.cluster_sizes[target_cluster])

        # Actualización online del centroide
        self.centroids[target_cluster] += eta * (x - self.centroids[target_cluster])

        # Registro del punto en el cluster
        clean_idx = (
            int(idx) if (idx is not None and idx != "temp")
            else int(self.total_assigned)
        )
        self.cluster_members_[target_cluster].append(clean_idx)
        self.total_assigned += 1

        # Sincronización de métricas externas
        self.features_list.append(x)
        self.labels_.append(int(target_cluster))
        self.assigned_true_labels.append(true_label)

        return target_cluster

    def partial_fit(self, x, idx=None, true_label=None):
        """
        Procesa un único punto de manera incremental.

        - Si el modelo no está inicializado, el punto se guarda en un buffer.
        - Una vez lleno el buffer, se inicializan los centroides con KMeans.
        - Luego, cada punto se asigna online respetando las restricciones
          de tamaño de los clusters.

        Parámetros
        ----------
        x : np.ndarray
            Vector de características del punto.
        idx : int, opcional
            Identificador del punto.
        true_label : int, opcional
            Etiqueta real del punto.

        Retorna
        -------
        int, list o str
            - ID del cluster asignado si el modelo ya está inicializado.
            - Lista de asignaciones si se acaba de inicializar.
            - "pending" si aún se está llenando el buffer inicial.
        """

        # Fase de inicialización
        if not self.initialized:
            self.init_buffer.append(x)
            self.init_indices.append(idx if idx is not None else "temp")
            self.init_labels.append(true_label)

            # Cuando el buffer está lleno
            if len(self.init_buffer) >= self.init_buffer_size:
                # 1. Inicialización de centroides con KMeans clásico
                X_buf = np.array(self.init_buffer)
                km = KMeans(
                    n_clusters=self.k,
                    init="k-means++",
                    n_init=1,
                    random_state=self.random_state
                )
                km.fit(X_buf)
                self.centroids = km.cluster_centers_
                self.initialized = True

                # 2. Asignación retroactiva de los puntos del buffer
                buffer_assignments = []
                for b_x, b_idx, b_label in zip(
                    self.init_buffer, self.init_indices, self.init_labels
                ):
                    buffer_assignments.append(
                        self._assign_point(b_x, b_idx, b_label)
                    )

                # Limpieza de buffers
                self.init_buffer = []
                self.init_labels = []

                return buffer_assignments

            return "pending"

        # Fase online normal
        return self._assign_point(x, idx, true_label)
