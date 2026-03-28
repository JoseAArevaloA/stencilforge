"""
Modulo de generacion de stencil: islas, puentes y limpieza morfologica.

Un stencil fisico requiere que todas las piezas negras (material) esten
conectadas entre si. Las "islas" son regiones negras completamente rodeadas
de blanco que se caerian al cortar. Los "puentes" son tiras delgadas de
material que conectan las islas al cuerpo principal.

Pipeline por capa:
    Mascara binaria -> Limpieza morfologica -> Deteccion de islas (CCL)
    -> Clasificacion (principal vs isla) -> Generacion de puentes -> Stencil final

Algoritmos clave:
- Connected Component Labeling (CCL): cv2.connectedComponentsWithStats
- Morfologia: erosion, dilatacion, apertura, cierre (OpenCV)
- Puentes: raycast desde borde de isla hacia region anclada mas cercana
"""

import cv2
import numpy as np
from typing import Optional


def morphological_clean(
    binary: np.ndarray,
    kernel_size: int = 3,
    close_iterations: int = 2,
    open_iterations: int = 1,
) -> np.ndarray:
    """
    Limpieza morfologica para eliminar ruido y suavizar bordes.

    Operaciones:
    1. Cierre (dilatar + erosionar): rellena huecos pequenos en regiones negras
    2. Apertura (erosionar + dilatar): elimina puntos blancos aislados

    Args:
        binary: Imagen binaria (H, W), 0=negro(material), 255=blanco(corte).
        kernel_size: Tamano del kernel morfologico.
        close_iterations: Iteraciones de cierre.
        open_iterations: Iteraciones de apertura.

    Returns:
        Imagen binaria limpia (H, W).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Cierre: rellena huecos pequenos
    cleaned = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations
    )

    # Apertura: elimina ruido
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iterations
    )

    return cleaned


def detect_islands(
    stencil: np.ndarray,
    min_island_size: int = 100,
) -> dict:
    """
    Detecta islas (regiones negras desconectadas) usando Connected Component Labeling.

    En un stencil:
    - Negro (0) = material que se queda
    - Blanco (255) = area de corte

    Las islas son componentes conectados de pixeles negros que NO tocan el borde
    de la imagen y NO son el componente mas grande (cuerpo principal).

    Args:
        stencil: Imagen binaria (H, W), 0=material, 255=corte.
        min_island_size: Tamano minimo en pixeles para considerar una isla.
                        Islas mas pequenas se eliminan (se vuelven blancas).

    Returns:
        dict con:
            - 'labels': Mapa de labels (H, W), cada componente tiene un ID unico
            - 'num_components': Numero total de componentes
            - 'main_body_label': Label del cuerpo principal
            - 'island_labels': Lista de labels que son islas
            - 'small_labels': Labels de componentes muy pequenos (eliminados)
            - 'anchored_labels': Labels de componentes que tocan el borde
            - 'stats': Estadisticas por componente (area, bounding box, centroide)
    """
    h, w = stencil.shape

    # Invertir: CCL trabaja sobre pixeles blancos, nuestro material es negro
    material = cv2.bitwise_not(stencil)

    # Connected Component Labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        material, connectivity=8
    )

    # Label 0 es el fondo (area de corte blanca), ignorar
    component_info = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        # Verificar si toca el borde
        touches_border = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)

        component_info.append({
            "label": i,
            "area": int(area),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "centroid": (float(cx), float(cy)),
            "touches_border": touches_border,
        })

    if not component_info:
        return {
            "labels": labels,
            "num_components": 0,
            "main_body_label": -1,
            "island_labels": [],
            "small_labels": [],
            "anchored_labels": [],
            "stats": [],
        }

    # El cuerpo principal es el componente mas grande
    main_body = max(component_info, key=lambda c: c["area"])
    main_body_label = main_body["label"]

    # Clasificar componentes
    anchored = []
    islands = []
    small = []

    for comp in component_info:
        if comp["label"] == main_body_label:
            anchored.append(comp["label"])
        elif comp["touches_border"]:
            anchored.append(comp["label"])
        elif comp["area"] < min_island_size:
            small.append(comp["label"])
        else:
            islands.append(comp["label"])

    return {
        "labels": labels,
        "num_components": len(component_info),
        "main_body_label": main_body_label,
        "island_labels": islands,
        "small_labels": small,
        "anchored_labels": anchored,
        "stats": component_info,
    }


def remove_small_components(
    stencil: np.ndarray,
    labels: np.ndarray,
    small_labels: list,
) -> np.ndarray:
    """
    Elimina componentes muy pequenos convirtiendolos en area de corte (blanco).

    Args:
        stencil: Imagen binaria (H, W).
        labels: Mapa de labels del CCL.
        small_labels: Lista de labels a eliminar.

    Returns:
        Imagen binaria con componentes pequenos removidos.
    """
    result = stencil.copy()
    for lbl in small_labels:
        result[labels == lbl] = 255  # Convertir a area de corte
    return result


def _find_nearest_anchor_point(
    island_label: int,
    labels: np.ndarray,
    anchored_labels: list,
    stencil: np.ndarray,
) -> Optional[tuple]:
    """
    Encuentra el punto mas cercano de una region anclada desde una isla.

    Estrategia: desde el centroide de la isla, lanza rayos en 16 direcciones
    buscando pixeles de regiones ancladas.

    Returns:
        Tuple (island_point, anchor_point) o None si no se encuentra.
    """
    h, w = labels.shape

    # Obtener pixeles del borde de la isla
    island_mask = (labels == island_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Puntos del contorno de la isla
    contour_points = contours[0].reshape(-1, 2)

    # Mascara de regiones ancladas
    anchor_mask = np.zeros_like(labels, dtype=bool)
    for lbl in anchored_labels:
        anchor_mask |= (labels == lbl)

    # Buscar el par de puntos (isla, anclado) mas cercanos
    # Para eficiencia, muestrear puntos del contorno
    step = max(1, len(contour_points) // 50)
    sampled_points = contour_points[::step]

    best_dist = float("inf")
    best_pair = None

    # Direcciones de raycast (16 direcciones)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    directions = [(np.cos(a), np.sin(a)) for a in angles]

    for px, py in sampled_points:
        for dx, dy in directions:
            # Lanzar rayo desde punto del contorno
            for step_size in range(1, max(h, w)):
                nx = int(px + dx * step_size)
                ny = int(py + dy * step_size)

                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    break

                if anchor_mask[ny, nx]:
                    dist = step_size
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = ((int(px), int(py)), (nx, ny))
                    break

    return best_pair


def generate_bridges(
    stencil: np.ndarray,
    labels: np.ndarray,
    island_labels: list,
    anchored_labels: list,
    bridge_width: int = 3,
    bridges_per_island: int = 1,
) -> dict:
    """
    Genera puentes conectando islas al cuerpo principal.

    Para cada isla:
    1. Encuentra el punto mas cercano de una region anclada (raycast)
    2. Dibuja una linea blanca (area de corte se interrumpe) de ancho
       configurable entre isla y region anclada

    Nota: Los puentes son lineas de MATERIAL (negro en el stencil) que
    atraviesan areas de CORTE (blanco). Se dibujan como negro sobre blanco.

    Args:
        stencil: Imagen binaria (H, W), 0=material, 255=corte.
        labels: Mapa de labels del CCL.
        island_labels: Labels de las islas a conectar.
        anchored_labels: Labels de las regiones ancladas.
        bridge_width: Ancho del puente en pixeles.
        bridges_per_island: Numero de puentes por isla.

    Returns:
        dict con:
            - 'stencil': Imagen con puentes aplicados
            - 'bridges': Lista de puentes generados [(start, end), ...]
            - 'num_bridges': Numero total de puentes
    """
    result = stencil.copy()
    bridges = []

    for island_label in island_labels:
        # Encontrar puntos de conexion
        pair = _find_nearest_anchor_point(
            island_label, labels, anchored_labels, stencil
        )

        if pair is None:
            continue

        island_pt, anchor_pt = pair
        bridges.append((island_pt, anchor_pt))

        # Dibujar puente como material (negro=0) atravesando corte (blanco)
        cv2.line(result, island_pt, anchor_pt, 0, thickness=bridge_width)

        # Puentes adicionales si se solicitan
        if bridges_per_island > 1:
            island_mask = (labels == island_label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                island_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if contours:
                contour_pts = contours[0].reshape(-1, 2)
                # Seleccionar puntos en extremos opuestos de la isla
                n_extra = min(bridges_per_island - 1, 3)
                indices = np.linspace(0, len(contour_pts) - 1, n_extra + 2, dtype=int)[1:-1]

                for idx in indices:
                    extra_start = tuple(contour_pts[idx])
                    extra_pair = _find_nearest_anchor_point(
                        island_label, labels, anchored_labels, stencil
                    )
                    if extra_pair:
                        _, extra_anchor = extra_pair
                        cv2.line(
                            result, extra_start, extra_anchor, 0,
                            thickness=bridge_width
                        )
                        bridges.append((extra_start, extra_anchor))

    return {
        "stencil": result,
        "bridges": bridges,
        "num_bridges": len(bridges),
    }


def build_stencil_layer(
    layer_mask: np.ndarray,
    min_island_size: int = 100,
    bridge_width: int = 3,
    bridges_per_island: int = 1,
    morph_clean: bool = True,
    morph_kernel: int = 3,
) -> dict:
    """
    Pipeline completo para construir una capa de stencil.

    Recibe una mascara binaria (de la cuantizacion K-Means) y produce
    un stencil listo para cortar con islas conectadas.

    Args:
        layer_mask: Mascara binaria (H, W), 255=esta capa, 0=no.
        min_island_size: Tamano minimo de isla en pixeles.
        bridge_width: Ancho de puentes.
        bridges_per_island: Puentes por isla.
        morph_clean: Aplicar limpieza morfologica.
        morph_kernel: Tamano del kernel morfologico.

    Returns:
        dict con:
            - 'stencil': Stencil final (H, W), 0=material, 255=corte
            - 'num_islands': Islas detectadas
            - 'num_bridges': Puentes generados
            - 'num_removed': Componentes pequenos eliminados
            - 'island_info': Info detallada de islas
    """
    # Convertir mascara de capa a stencil: donde HAY capa -> corte (255)
    # donde NO hay capa -> material (0)
    # Esto es porque la capa indica "aqui va pintura" = corte
    stencil = layer_mask.copy()

    # Limpieza morfologica
    if morph_clean:
        stencil = morphological_clean(
            stencil, kernel_size=morph_kernel
        )

    # Invertir: para stencil, material=0 (negro), corte=255 (blanco)
    # La mascara ya tiene 255=capa, que sera el area de corte
    # y 0=no-capa, que sera material
    # Pero necesitamos que las islas de MATERIAL esten conectadas
    # Asi que el stencil es: 0 donde hay material, 255 donde se corta

    # Detectar islas
    island_data = detect_islands(stencil, min_island_size=min_island_size)

    # Eliminar componentes muy pequenos
    if island_data["small_labels"]:
        stencil = remove_small_components(
            stencil, island_data["labels"], island_data["small_labels"]
        )
        # Re-detectar despues de eliminar
        island_data = detect_islands(stencil, min_island_size=min_island_size)

    # Generar puentes
    bridge_data = {"stencil": stencil, "bridges": [], "num_bridges": 0}
    if island_data["island_labels"]:
        bridge_data = generate_bridges(
            stencil,
            island_data["labels"],
            island_data["island_labels"],
            island_data["anchored_labels"],
            bridge_width=bridge_width,
            bridges_per_island=bridges_per_island,
        )

    return {
        "stencil": bridge_data["stencil"],
        "num_islands": len(island_data["island_labels"]),
        "num_bridges": bridge_data["num_bridges"],
        "num_removed": len(island_data["small_labels"]),
        "island_info": [
            s for s in island_data["stats"]
            if s["label"] in island_data["island_labels"]
        ],
    }
