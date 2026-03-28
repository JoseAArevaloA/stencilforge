"""
Modulo de cuantizacion tonal/color con K-Means.

Dos modos:
  - Escala de grises (Banksy): K-Means sobre luminancia [0,255]
  - Color real: K-Means sobre espacio LAB (perceptualmente uniforme)

Pipeline escala de grises:
    Imagen gris -> (N_pixeles, 1) normalizado -> K-Means -> N mascaras binarias

Pipeline color (LAB):
    Imagen BGR -> LAB -> (N_pixeles, 3) normalizado -> K-Means
    -> N mascaras binarias + color representativo por capa (hex)

LAB es superior a RGB para clustering porque la distancia euclidiana
en LAB corresponde a diferencias percibidas por el ojo humano.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_layers(
    gray_image: np.ndarray,
    max_k: int = 6,
    sample_size: int = 50000,
) -> dict:
    """
    Evalua diferentes valores de K para encontrar el numero optimo de capas.

    Usa Elbow Method (inercia) y Silhouette Score, las mismas metricas
    de evaluacion de clustering del curso de ML.

    Args:
        gray_image: Imagen en escala de grises (H, W), dtype uint8.
        max_k: Maximo numero de clusters a evaluar.
        sample_size: Numero de pixeles a muestrear (para velocidad).

    Returns:
        dict con:
            - 'inertias': Lista de inercias por cada K
            - 'silhouette_scores': Lista de silhouette scores por cada K
            - 'suggested_k': K sugerido basado en silhouette score maximo
            - 'k_range': Rango de K evaluados [2, max_k]
    """
    # Aplanar imagen a vector de pixeles
    pixels = gray_image.flatten().astype(np.float32).reshape(-1, 1)

    # Muestrear para velocidad (imagenes grandes tienen millones de pixeles)
    if len(pixels) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pixels), size=sample_size, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels

    # Pixeles de imagen ya estan en [0, 255], no necesitan escalado.
    # StandardScaler perjudica K-Means en este caso porque comprime
    # la distribucion y hace que clusters distintos colapsen.
    pixels_norm = pixels_sample / 255.0  # Normalizar a [0, 1] sin centrar

    inertias = []
    sil_scores = []
    k_range = list(range(2, max_k + 1))

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        labels = kmeans.fit_predict(pixels_norm)

        inertias.append(float(kmeans.inertia_))

        # Silhouette score: mide que tan bien separados estan los clusters
        # Rango [-1, 1], mas alto = mejor separacion
        sil = silhouette_score(pixels_norm, labels, sample_size=min(10000, len(pixels_norm)))
        sil_scores.append(float(sil))

    # Sugerir K con mejor silhouette score
    best_idx = np.argmax(sil_scores)
    suggested_k = k_range[best_idx]

    return {
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "suggested_k": suggested_k,
        "k_range": k_range,
    }


def quantize_tonal(
    gray_image: np.ndarray,
    n_layers: int = 3,
    sample_size: int = 100000,
) -> dict:
    """
    Cuantiza la imagen en N capas tonales usando K-Means.

    El proceso:
    1. Aplanar pixeles y escalar
    2. Aplicar K-Means con k = n_layers
    3. Ordenar clusters de oscuro a claro
    4. Generar mascara binaria por cada capa

    Args:
        gray_image: Imagen en escala de grises (H, W), dtype uint8.
        n_layers: Numero de capas tonales (2-6).
        sample_size: Pixeles para entrenar K-Means (el modelo se aplica a todos).

    Returns:
        dict con:
            - 'layers': Lista de N mascaras binarias (H, W), cada una uint8 0/255
            - 'labels_map': Mapa de labels (H, W) con el cluster asignado por pixel
            - 'centers': Centros de los clusters (valores tonales)
            - 'quantized_image': Imagen cuantizada (cada pixel = centro de su cluster)
            - 'inertia': Inercia del modelo
            - 'layer_percentages': Porcentaje de pixeles en cada capa
    """
    h, w = gray_image.shape
    pixels = gray_image.flatten().astype(np.float32).reshape(-1, 1)

    # Entrenar K-Means en muestra
    if len(pixels) > sample_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pixels), size=sample_size, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels

    pixels_norm_sample = pixels_sample / 255.0
    pixels_norm_all = pixels / 255.0

    kmeans = KMeans(
        n_clusters=n_layers,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    kmeans.fit(pixels_norm_sample)

    # Predecir labels para TODOS los pixeles
    labels = kmeans.predict(pixels_norm_all)

    # Centros en escala original [0, 255]
    centers_original = (kmeans.cluster_centers_.flatten() * 255.0)

    # Ordenar clusters de oscuro a claro
    sort_order = np.argsort(centers_original)
    # Re-mapear labels segun el nuevo orden
    label_remap = np.zeros(n_layers, dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        label_remap[old_idx] = new_idx

    labels_sorted = label_remap[labels]
    centers_sorted = centers_original[sort_order]

    # Reshape a imagen
    labels_map = labels_sorted.reshape(h, w)

    # Generar imagen cuantizada
    quantized = centers_sorted[labels_sorted].reshape(h, w).astype(np.uint8)

    # Generar mascaras binarias por capa
    # Capa 0 = mas oscura (sombras), Capa N-1 = mas clara (luces)
    layers = []
    layer_percentages = []
    for i in range(n_layers):
        mask = (labels_map == i).astype(np.uint8) * 255
        layers.append(mask)
        pct = float(np.sum(labels_map == i)) / (h * w) * 100
        layer_percentages.append(round(pct, 1))

    return {
        "layers": layers,
        "labels_map": labels_map,
        "centers": centers_sorted.tolist(),
        "quantized_image": quantized,
        "inertia": float(kmeans.inertia_),
        "layer_percentages": layer_percentages,
        "layer_colors_hex": [None] * n_layers,  # Grayscale: sin color
    }


def quantize_color(
    image_bgr: np.ndarray,
    n_layers: int = 4,
    sample_size: int = 100000,
) -> dict:
    """
    Cuantiza la imagen en N capas de color usando K-Means en espacio LAB.

    LAB (L*a*b*) es un espacio de color perceptualmente uniforme donde
    la distancia euclidiana corresponde a la diferencia percibida por el
    ojo humano. Produce agrupaciones de color mas naturales que RGB.

    Args:
        image_bgr: Imagen BGR (H, W, 3), dtype uint8.
        n_layers: Numero de colores/capas.
        sample_size: Pixeles para entrenar K-Means.

    Returns:
        dict con:
            - 'layers': N mascaras binarias (H, W), uint8 0/255
            - 'labels_map': Mapa de labels (H, W)
            - 'layer_colors_hex': Color representativo por capa '#RRGGBB'
            - 'layer_colors_bgr': Color representativo por capa como tuple (B,G,R)
            - 'quantized_image': Imagen cuantizada en color BGR
            - 'layer_percentages': Porcentaje de pixeles por capa
            - 'centers': Centros LAB (referencia)
    """
    h, w = image_bgr.shape[:2]

    # Convertir BGR -> LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3) / 255.0  # Normalizar a [0,1]

    # Muestrear para velocidad
    if len(pixels) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), size=sample_size, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    kmeans = KMeans(
        n_clusters=n_layers,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    kmeans.fit(pixels_sample)

    labels = kmeans.predict(pixels)

    # Centros en escala LAB original [0,255]
    centers_lab = (kmeans.cluster_centers_ * 255.0).astype(np.uint8)

    # Ordenar clusters por luminosidad (canal L, indice 0 en OpenCV LAB)
    sort_order = np.argsort(centers_lab[:, 0])
    label_remap = np.zeros(n_layers, dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        label_remap[old_idx] = new_idx

    labels_sorted = label_remap[labels]
    centers_lab_sorted = centers_lab[sort_order]
    labels_map = labels_sorted.reshape(h, w)

    # Convertir centros LAB -> BGR y calcular hex
    layer_colors_bgr = []
    layer_colors_hex = []
    for lab_center in centers_lab_sorted:
        lab_px = lab_center.reshape(1, 1, 3)
        bgr_px = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0, 0]
        b, g, r = int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])
        layer_colors_bgr.append((b, g, r))
        layer_colors_hex.append(f"#{r:02x}{g:02x}{b:02x}")

    # Imagen cuantizada en color
    quantized = np.zeros((h, w, 3), dtype=np.uint8)
    for i, bgr in enumerate(layer_colors_bgr):
        quantized[labels_map == i] = bgr

    # Mascaras binarias
    layers = []
    layer_percentages = []
    for i in range(n_layers):
        mask = (labels_map == i).astype(np.uint8) * 255
        layers.append(mask)
        pct = float(np.sum(labels_map == i)) / (h * w) * 100
        layer_percentages.append(round(pct, 1))

    return {
        "layers": layers,
        "labels_map": labels_map,
        "layer_colors_hex": layer_colors_hex,
        "layer_colors_bgr": layer_colors_bgr,
        "quantized_image": quantized,
        "layer_percentages": layer_percentages,
        "centers": centers_lab_sorted.tolist(),
        "inertia": float(kmeans.inertia_),
    }
