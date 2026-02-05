# Backend – Online Clustering de Imágenes

Este repositorio contiene el **backend del sistema de clustering online de imágenes** desarrollado para el **Proyecto Integrador de Visión por Computador**.

El backend es responsable de **todo el procesamiento computacional**, incluyendo:
- preprocesamiento de imágenes,
- extracción de características,
- generación de embeddings,
- ejecución del algoritmo de clustering online con restricciones de tamaño,
- y cálculo de métricas de evaluación.

El sistema expone estos servicios mediante una **API REST**, la cual es consumida por el frontend web.

---

## INTEGRANTES
 - Kevin Vallejo
 - Freddy Viracocha
 - Julián Cañas
 - John Serrano

## 🧠 Funcionalidades principales

El backend permite:

- Recibir imágenes desde el frontend
- Preprocesar imágenes (normalización, redimensionamiento, mejora de contraste, etc.)
- Extraer características y/o embeddings
- Ejecutar clustering online con restricciones de tamaño
- Calcular métricas de validación internas
- Mantener estado del clustering durante la sesión
- Exponer resultados vía endpoints HTTP

---

## 🛠️ Tecnologías utilizadas

- **Python 3**
- **Flask** – API REST
- **NumPy / SciPy** – operaciones numéricas
- **OpenCV** – procesamiento de imágenes
- **Scikit-learn** – métricas y clustering auxiliar
- **ONNX Runtime** – inferencia de modelos (embeddings)
- **Docker** (opcional) – despliegue

---

## 📁 Estructura general del proyecto

```text
Backend_OnlineClustering/
├── app.py
├── requirements.txt
├── preprocess/
│   └── preprocess_image.py
├── features/
│   ├── extract_features.py
│   └── onnx_model.py
├── clustering/
│   └── online_kmeans_size_constrained.py
├── utils/
│   └── metrics.py
└── README.md
