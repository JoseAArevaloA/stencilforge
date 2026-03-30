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

Modo SLIC (opcional):
    Pre-agrupacion espacial con SLIC Superpixels antes de K-Means.
    Cada superpixel agrupa pixeles similares en color Y posicion,
    produciendo capas con bordes mas limpios y coherentes.
    K-Means opera sobre ~300 superpixeles en lugar de millones de pixeles.
    Nota: SLIC pierde detalles finos en retratos (ojos, labios). Usar
    Bilateral Filter para ese caso de uso.

Modo Bilateral Filter (opcional, recomendado para retratos):
    Filtro bilateral de OpenCV aplicado antes de K-Means.
    Suaviza ruido y texturas suaves (piel) mientras preserva bordes
    nitidos (ojos, labios, siluetas). Reduce islas pequenas sin perder
    detalles importantes.
    Presets de intensidad: light / medium / strong.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Bilateral Filter helper
# ---------------------------------------------------------------------------

# Parametros por preset: (d, sigmaColor, sigmaSpace)
_BILATERAL_PRESETS = {
    "light":  (7,  30,  30),
    "medium": (9,  75,  75),
    "strong": (11, 150, 150),
}

def _apply_bilateral(image: np.ndarray, strength: str = "medium") -> np.ndarray:
    """
    Aplica filtro bilateral a imagen en escala de grises o BGR.

    El filtro bilateral es no-lineal: promedia pixeles cercanos pesando
    tanto la distancia espacial (sigmaSpace) como la similitud de color
    (sigmaColor). Resultado: ruido y textura fina se suavizan, bordes
    nitidos se preservan.

    Parametros por preset:
        light:  d=7,  sigmaColor=30,  sigmaSpace=30   (suavizado leve)
        medium: d=9,  sigmaColor=75,  sigmaSpace=75   (retratos, general)
        strong: d=11, sigmaColor=150, sigmaSpace=150  (simplificacion maxima)

    Args:
        image: Imagen uint8, (H,W) gris o (H,W,3) BGR.
        strength: Preset de intensidad ('light', 'medium', 'strong').

    Returns:
        Imagen filtrada, mismo shape y dtype que la entrada.
    """
    d, sc, ss = _BILATERAL_PRESETS.get(strength, _BILATERAL_PRESETS["medium"])
    return cv2.bilateralFilter(image, d, sc, ss)


# ---------------------------------------------------------------------------
# SLIC helpers
# ---------------------------------------------------------------------------

def _slic_segments_gray(
    gray_image: np.ndarray,
    n_segments: int = 300,
    compactness: float = 10.0,
) -> tuple:
    """
    Genera superpixeles SLIC sobre imagen en escala de grises.

    SLIC (Simple Linear Iterative Clustering) agrupa pixeles por similitud
    de color Y proximidad espacial, produciendo regiones compactas y conexas.
    Esto corrige la debilidad de K-Means puro, que ignora posicion espacial.

    Args:
        gray_image: Imagen (H, W) uint8.
        n_segments: Numero aproximado de superpixeles a generar.
        compactness: Peso del factor espacial vs. color (mayor = mas compacto).

    Returns:
        (segment_map, features):
            - segment_map: (H, W) int, ID de superpixel por pixel.
            - features: (n_sp, 1) float32, media normalizada de gris por superpixel.
    """
    from skimage.segmentation import slic
    from skimage.color import gray2rgb

    # SLIC necesita imagen RGB o LAB; convertimos gris a RGB (R=G=B)
    rgb = gray2rgb(gray_image)
    segment_map = slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        start_label=0,
        channel_axis=-1,
    )

    n_sp = int(segment_map.max()) + 1
    features = np.zeros((n_sp, 1), dtype=np.float32)
    gray_f = gray_image.astype(np.float32) / 255.0
    for sp_id in range(n_sp):
        mask = segment_map == sp_id
        features[sp_id, 0] = gray_f[mask].mean()

    return segment_map, features


def _slic_segments_lab(
    image_bgr: np.ndarray,
    n_segments: int = 300,
    compactness: float = 10.0,
) -> tuple:
    """
    Genera superpixeles SLIC sobre imagen BGR usando espacio LAB internamente.

    Args:
        image_bgr: Imagen (H, W, 3) uint8.
        n_segments: Numero aproximado de superpixeles.
        compactness: Peso del factor espacial vs. color.

    Returns:
        (segment_map, features):
            - segment_map: (H, W) int.
            - features: (n_sp, 3) float32, media LAB normalizada por superpixel.
    """
    from skimage.segmentation import slic

    # SLIC con convert2lab=True opera en LAB internamente (requiere RGB input)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    segment_map = slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        start_label=0,
        convert2lab=True,
        channel_axis=-1,
    )

    # Features: media LAB por superpixel (misma representacion que quantize_color)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    n_sp = int(segment_map.max()) + 1
    features = np.zeros((n_sp, 3), dtype=np.float32)
    for sp_id in range(n_sp):
        mask = segment_map == sp_id
        features[sp_id] = lab[mask].mean(axis=0)

    return segment_map, features


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
    use_slic: bool = False,
    slic_segments: int = 300,
    bilateral_filter: bool = False,
    bilateral_strength: str = "medium",
) -> dict:
    """
    Cuantiza la imagen en N capas tonales usando K-Means.

    El proceso:
    1. (Opcional) Filtro bilateral — suaviza ruido preservando bordes
    2. (Opcional) Pre-agrupar pixeles en superpixeles SLIC
    3. Aplicar K-Means con k = n_layers
    4. Ordenar clusters de oscuro a claro
    5. Generar mascara binaria por cada capa

    Args:
        gray_image: Imagen en escala de grises (H, W), dtype uint8.
        n_layers: Numero de capas tonales (2-6).
        sample_size: Pixeles para entrenar K-Means cuando use_slic=False.
        use_slic: Si True, pre-agrupa con SLIC antes de K-Means.
        slic_segments: Numero aproximado de superpixeles SLIC (~200-500).
        bilateral_filter: Si True, aplica filtro bilateral antes de K-Means.
                          Recomendado para retratos: suaviza piel preservando ojos/labios.
        bilateral_strength: Preset de intensidad ('light', 'medium', 'strong').

    Returns:
        dict con:
            - 'layers': Lista de N mascaras binarias (H, W), cada una uint8 0/255
            - 'labels_map': Mapa de labels (H, W) con el cluster asignado por pixel
            - 'centers': Centros de los clusters (valores tonales)
            - 'quantized_image': Imagen cuantizada (cada pixel = centro de su cluster)
            - 'inertia': Inercia del modelo
            - 'layer_percentages': Porcentaje de pixeles en cada capa
            - 'slic_used': bool indicando si se uso SLIC
    """
    # Pre-procesado opcional: filtro bilateral
    if bilateral_filter:
        gray_image = _apply_bilateral(gray_image, bilateral_strength)

    h, w = gray_image.shape
    kmeans = KMeans(
        n_clusters=n_layers,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )

    if use_slic:
        # Modo SLIC: K-Means sobre features de superpixeles (~300 puntos en lugar de miles)
        segment_map, sp_features = _slic_segments_gray(gray_image, n_segments=slic_segments)
        kmeans.fit(sp_features)
        # Cada superpixel recibe un label; cada pixel hereda el label de su superpixel
        sp_labels = kmeans.labels_
        labels = sp_labels[segment_map]  # (H, W) — vectorizado sin bucle
        centers_original = kmeans.cluster_centers_.flatten() * 255.0
    else:
        # Modo clasico: K-Means sobre pixeles individuales
        pixels = gray_image.flatten().astype(np.float32).reshape(-1, 1)
        if len(pixels) > sample_size:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pixels), size=sample_size, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
        kmeans.fit(pixels_sample / 255.0)
        labels = kmeans.predict(pixels.flatten().reshape(-1, 1) / 255.0).reshape(h, w)
        centers_original = kmeans.cluster_centers_.flatten() * 255.0

    # Ordenar clusters de oscuro a claro
    sort_order = np.argsort(centers_original)
    label_remap = np.zeros(n_layers, dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        label_remap[old_idx] = new_idx

    labels_map = label_remap[labels]  # (H, W)
    centers_sorted = centers_original[sort_order]

    quantized = centers_sorted[labels_map].astype(np.uint8)

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
        "layer_colors_hex": [None] * n_layers,
        "slic_used": use_slic,
        "bilateral_used": bilateral_filter,
    }


def quantize_color(
    image_bgr: np.ndarray,
    n_layers: int = 4,
    sample_size: int = 100000,
    use_slic: bool = False,
    slic_segments: int = 300,
    bilateral_filter: bool = False,
    bilateral_strength: str = "medium",
) -> dict:
    """
    Cuantiza la imagen en N capas de color usando K-Means en espacio LAB.

    LAB (L*a*b*) es un espacio de color perceptualmente uniforme donde
    la distancia euclidiana corresponde a la diferencia percibida por el
    ojo humano. Produce agrupaciones de color mas naturales que RGB.

    Args:
        image_bgr: Imagen BGR (H, W, 3), dtype uint8.
        n_layers: Numero de colores/capas.
        sample_size: Pixeles para entrenar K-Means cuando use_slic=False.
        use_slic: Si True, pre-agrupa con SLIC antes de K-Means.
        slic_segments: Numero aproximado de superpixeles SLIC.
        bilateral_filter: Si True, aplica filtro bilateral antes de K-Means.
        bilateral_strength: Preset de intensidad ('light', 'medium', 'strong').

    Returns:
        dict con:
            - 'layers': N mascaras binarias (H, W), uint8 0/255
            - 'labels_map': Mapa de labels (H, W)
            - 'layer_colors_hex': Color representativo por capa '#RRGGBB'
            - 'layer_colors_bgr': Color representativo por capa como tuple (B,G,R)
            - 'quantized_image': Imagen cuantizada en color BGR
            - 'layer_percentages': Porcentaje de pixeles por capa
            - 'centers': Centros LAB (referencia)
            - 'slic_used': bool indicando si se uso SLIC
    """
    # Pre-procesado opcional: filtro bilateral
    if bilateral_filter:
        image_bgr = _apply_bilateral(image_bgr, bilateral_strength)

    h, w = image_bgr.shape[:2]

    kmeans = KMeans(
        n_clusters=n_layers,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )

    if use_slic:
        # Modo SLIC: K-Means sobre features LAB de superpixeles
        segment_map, sp_features = _slic_segments_lab(image_bgr, n_segments=slic_segments)
        kmeans.fit(sp_features)
        sp_labels = kmeans.labels_
        labels_map = sp_labels[segment_map]  # (H, W)
        centers_lab = (kmeans.cluster_centers_ * 255.0).astype(np.uint8)
    else:
        # Modo clasico: K-Means sobre pixeles individuales
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab.reshape(-1, 3) / 255.0
        if len(pixels) > sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pixels), size=sample_size, replace=False)
            pixels_sample = pixels[idx]
        else:
            pixels_sample = pixels
        kmeans.fit(pixels_sample)
        labels = kmeans.predict(pixels)
        centers_lab = (kmeans.cluster_centers_ * 255.0).astype(np.uint8)
        labels_map = labels.reshape(h, w)

    # Ordenar clusters por luminosidad (canal L, indice 0 en OpenCV LAB)
    sort_order = np.argsort(centers_lab[:, 0])
    label_remap = np.zeros(n_layers, dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        label_remap[old_idx] = new_idx

    centers_lab_sorted = centers_lab[sort_order]
    labels_map = label_remap[labels_map]  # (H, W)

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
        "slic_used": use_slic,
        "bilateral_used": bilateral_filter,
    }
