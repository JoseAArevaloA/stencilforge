"""
Modulo de deteccion de bordes con Canny adaptativo.

El algoritmo Canny es un detector de bordes multi-etapa:
1. Suavizado Gaussiano (reduce ruido)
2. Gradiente de intensidad (Sobel en X e Y)
3. Supresion de no-maximos (adelgaza bordes)
4. Umbral doble con histeresis (conecta bordes debiles a fuertes)

La version adaptativa calcula los umbrales automaticamente basandose
en la distribucion de intensidades de la imagen (mediana + sigma).

Pipeline:
    Imagen RGB -> Escala de grises -> Blur Gaussiano -> Canny adaptativo -> Mapa de bordes
"""

import cv2
import numpy as np


def auto_canny(
    gray: np.ndarray,
    sigma: float = 0.33,
) -> np.ndarray:
    """
    Canny con umbrales automaticos basados en la mediana de la imagen.

    La mediana es robusta a outliers. Los umbrales se calculan como:
        lower = max(0, (1 - sigma) * mediana)
        upper = min(255, (1 + sigma) * mediana)

    Un sigma mas alto = menos bordes detectados (mas selectivo).
    Un sigma mas bajo = mas bordes detectados (mas sensible).

    Args:
        gray: Imagen en escala de grises (H, W), dtype uint8.
        sigma: Factor de desviacion para calcular umbrales. Default 0.33.

    Returns:
        Mapa de bordes binario (H, W), 0 o 255.
    """
    median_val = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    return cv2.Canny(gray, lower, upper)


def detect_edges(
    image: np.ndarray,
    blur_kernel: int = 5,
    sigma: float = 0.33,
    detail_level: str = "medium",
    dilate_iterations: int = 1,
) -> dict:
    """
    Pipeline completo de deteccion de bordes.

    Args:
        image: Imagen BGR o RGB (H, W, 3), dtype uint8.
        blur_kernel: Tamano del kernel Gaussiano (debe ser impar).
        sigma: Factor para auto-Canny. Mas alto = menos bordes.
        detail_level: 'low', 'medium', 'high'. Ajusta sigma y blur.
        dilate_iterations: Iteraciones de dilatacion para engrosar bordes.

    Returns:
        dict con:
            - 'edges': Mapa de bordes binario (H, W)
            - 'edges_dilated': Bordes engrosados
            - 'gray': Imagen en escala de grises usada
            - 'params': Parametros efectivos usados
    """
    # Ajustar parametros segun nivel de detalle
    detail_presets = {
        "low": {"sigma": 0.50, "blur_kernel": 7},
        "medium": {"sigma": 0.33, "blur_kernel": 5},
        "high": {"sigma": 0.20, "blur_kernel": 3},
    }

    if detail_level in detail_presets:
        preset = detail_presets[detail_level]
        sigma = preset["sigma"]
        blur_kernel = preset["blur_kernel"]

    # Asegurar kernel impar
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Suavizado Gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Canny adaptativo
    edges = auto_canny(blurred, sigma=sigma)

    # Dilatar bordes para hacerlos mas visibles en el stencil
    edges_dilated = edges.copy()
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    return {
        "edges": edges,
        "edges_dilated": edges_dilated,
        "gray": gray,
        "params": {
            "sigma": sigma,
            "blur_kernel": blur_kernel,
            "detail_level": detail_level,
            "dilate_iterations": dilate_iterations,
        },
    }


def combine_edges_with_tonal(
    edges: np.ndarray,
    tonal_layer: np.ndarray,
    edge_weight: float = 0.5,
) -> np.ndarray:
    """
    Combina un mapa de bordes con una capa tonal para enriquecer el stencil.

    Los bordes aportan definicion y los tonos aportan volumen/forma.

    Args:
        edges: Mapa de bordes binario (H, W), 0 o 255.
        tonal_layer: Capa tonal binaria (H, W), 0 o 255.
        edge_weight: Peso de los bordes en la combinacion (0.0 a 1.0).

    Returns:
        Imagen combinada binaria (H, W), 0 o 255.
    """
    # Normalizar a float [0, 1]
    edges_f = edges.astype(np.float32) / 255.0
    tonal_f = tonal_layer.astype(np.float32) / 255.0

    # Combinacion ponderada
    combined = edge_weight * edges_f + (1.0 - edge_weight) * tonal_f

    # Binarizar resultado
    result = (combined > 0.5).astype(np.uint8) * 255
    return result
