"""
API FastAPI para el Generador de Plantillas (Stencils).

Endpoints:
    POST /api/upload          - Sube una imagen y retorna preview
    POST /api/remove-bg       - Remueve fondo con U2-Net
    POST /api/analyze-layers  - Analiza numero optimo de capas (K-Means metrics)
    POST /api/process         - Pipeline completo: cuantizacion + stencil + puentes
    GET  /api/preview/{id}/{layer} - Preview de una capa especifica
    GET  /api/preview/{id}/combined - Preview combinado de todas las capas
    POST /api/export/pdf      - Exporta PDF multi-pagina
"""

import uuid
import io
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from pipeline.background import remove_background
from pipeline.edges import detect_edges, combine_edges_with_tonal
from pipeline.quantizer import find_optimal_layers, quantize_tonal, quantize_color
from pipeline.stencil_builder import build_stencil_layer
from pipeline.page_splitter import calculate_tiles, generate_full_pdf

app = FastAPI(
    title="Generador de Plantillas - Stencil Generator",
    description="Convierte imagenes complejas en stencils multi-capa para pintura",
    version="1.0.0",
)

# CORS para frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Almacen en memoria para sesiones de trabajo
# En produccion usariamos Redis o similar
_sessions: dict = {}

MAX_IMAGE_DIMENSION = 4000


def _resize_if_needed(image: np.ndarray) -> tuple:
    """Redimensiona si excede MAX_IMAGE_DIMENSION. Retorna (imagen, fue_redimensionada)."""
    h, w = image.shape[:2]
    if max(h, w) <= MAX_IMAGE_DIMENSION:
        return image, False

    scale = MAX_IMAGE_DIMENSION / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, True


def _numpy_to_base64(image: np.ndarray) -> str:
    """Convierte array numpy a string base64 PNG."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def _numpy_to_base64_rgb(image: np.ndarray) -> str:
    """Convierte array numpy BGR a base64 PNG."""
    if len(image.shape) == 2:
        # Escala de grises
        _, buffer = cv2.imencode(".png", image)
    else:
        _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Sube una imagen y crea una sesion de trabajo.

    Returns:
        session_id, dimensiones originales, preview de la imagen.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    image, was_resized = _resize_if_needed(image)
    h, w = image.shape[:2]

    session_id = str(uuid.uuid4())[:8]
    _sessions[session_id] = {
        "original": image,
        "no_bg": None,
        "gray": None,
        "layers": None,
        "stencils": None,
    }

    # Preview reducido para el frontend
    preview_scale = min(1.0, 800 / max(h, w))
    preview = cv2.resize(
        image,
        (int(w * preview_scale), int(h * preview_scale)),
        interpolation=cv2.INTER_AREA,
    )

    return {
        "session_id": session_id,
        "width": w,
        "height": h,
        "was_resized": was_resized,
        "preview": _numpy_to_base64(preview),
    }


@app.post("/api/remove-bg")
async def remove_bg(session_id: str, model: str = "u2net"):
    """
    Remueve el fondo de la imagen usando U2-Net.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    session = _sessions[session_id]
    image_bgr = session["original"]

    # Convertir BGR -> RGB para PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    result = remove_background(pil_image, model=model)

    # Convertir resultado a BGR para OpenCV
    white_bg = np.array(result["image_white_bg"])
    white_bg_bgr = cv2.cvtColor(white_bg, cv2.COLOR_RGB2BGR)

    session["no_bg"] = white_bg_bgr
    session["mask"] = result["mask"]

    # Preview
    h, w = white_bg_bgr.shape[:2]
    preview_scale = min(1.0, 800 / max(h, w))
    preview = cv2.resize(
        white_bg_bgr,
        (int(w * preview_scale), int(h * preview_scale)),
        interpolation=cv2.INTER_AREA,
    )

    return {
        "preview": _numpy_to_base64(preview),
        "mask_preview": _numpy_to_base64(
            cv2.resize(result["mask"], (int(w * preview_scale), int(h * preview_scale)))
        ),
    }


@app.post("/api/analyze-layers")
async def analyze_layers(session_id: str, max_k: int = 6):
    """
    Analiza el numero optimo de capas usando Elbow Method y Silhouette Score.

    Retorna metricas para que el usuario elija cuantas capas usar.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    session = _sessions[session_id]
    source = session["no_bg"] if session["no_bg"] is not None else session["original"]

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    session["gray"] = gray

    analysis = find_optimal_layers(gray, max_k=max_k)

    return {
        "k_range": analysis["k_range"],
        "inertias": analysis["inertias"],
        "silhouette_scores": analysis["silhouette_scores"],
        "suggested_k": analysis["suggested_k"],
    }


@app.post("/api/process")
async def process_stencil(
    session_id: str,
    n_layers: int = 3,
    min_island_size: int = 100,
    bridge_width: int = 3,
    bridges_per_island: int = 1,
    edge_detail: str = "medium",
    edge_weight: float = 0.3,
    morph_clean: bool = True,
    morph_kernel: int = 3,
    use_edges: bool = True,
    mode: str = "grayscale",  # "grayscale" | "color"
):
    """
    Pipeline completo de generacion de stencil.

    1. Cuantizacion K-Means en N capas
    2. Deteccion de bordes (opcional)
    3. Combinacion bordes + capas tonales
    4. Generacion de stencil por capa (islas + puentes)
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    session = _sessions[session_id]
    source = session["no_bg"] if session["no_bg"] is not None else session["original"]

    # Cuantizacion segun modo
    if mode == "color":
        quant_result = quantize_color(source, n_layers=n_layers)
    else:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        session["gray"] = gray
        quant_result = quantize_tonal(gray, n_layers=n_layers)

    # Deteccion de bordes (siempre en escala de grises)
    edge_map = None
    if use_edges:
        edge_result = detect_edges(source, detail_level=edge_detail)
        edge_map = edge_result["edges_dilated"]

    # Procesar cada capa
    stencils = []
    layer_previews = []
    layer_stats = []
    layer_colors_hex = quant_result.get("layer_colors_hex", [None] * n_layers)

    for i, layer_mask in enumerate(quant_result["layers"]):
        # Combinar con bordes si disponibles
        if edge_map is not None and edge_weight > 0:
            combined = combine_edges_with_tonal(edge_map, layer_mask, edge_weight)
        else:
            combined = layer_mask

        result = build_stencil_layer(
            combined,
            min_island_size=min_island_size,
            bridge_width=bridge_width,
            bridges_per_island=bridges_per_island,
            morph_clean=morph_clean,
            morph_kernel=morph_kernel,
        )

        stencils.append(result["stencil"])

        # Preview reducido — en modo color, superponer color de capa sobre stencil
        h, w = result["stencil"].shape
        preview_scale = min(1.0, 600 / max(h, w))

        if mode == "color" and layer_colors_hex[i]:
            color_bgr = quant_result["layer_colors_bgr"][i]
            stencil_preview = cv2.cvtColor(result["stencil"], cv2.COLOR_GRAY2BGR)
            # Donde es material (0 en stencil = 255 despues de invertir) pintar con color
            cut_mask = result["stencil"] == 0  # 0 = material en stencil
            stencil_preview[cut_mask] = color_bgr
            stencil_preview[~cut_mask] = (255, 255, 255)
            preview_img = cv2.resize(stencil_preview, (int(w * preview_scale), int(h * preview_scale)))
            layer_previews.append(_numpy_to_base64_rgb(preview_img))
        else:
            preview = cv2.resize(result["stencil"], (int(w * preview_scale), int(h * preview_scale)))
            layer_previews.append(_numpy_to_base64(preview))

        # Nombre de capa con color si disponible
        color_hex = layer_colors_hex[i]
        layer_name = f"Capa {i + 1}"
        if color_hex:
            layer_name = f"Capa {i + 1} ({color_hex})"

        center_val = quant_result["centers"][i]
        tonal_center = round(float(center_val[0]) if hasattr(center_val, "__len__") else float(center_val), 1)

        layer_stats.append({
            "layer": i,
            "name": layer_name,
            "tonal_center": tonal_center,
            "color_hex": color_hex,
            "coverage_pct": quant_result["layer_percentages"][i],
            "islands_detected": result["num_islands"],
            "bridges_added": result["num_bridges"],
            "small_removed": result["num_removed"],
        })

    session["stencils"] = stencils
    session["layer_names"] = [s["name"] for s in layer_stats]
    session["layer_colors"] = layer_colors_hex

    # Preview de imagen cuantizada
    quant_img = quant_result["quantized_image"]
    ref_shape = quant_img.shape
    ref_dim = ref_shape[0] if len(ref_shape) == 2 else ref_shape[0]
    quant_preview_scale = min(1.0, 600 / max(ref_shape[:2]))
    new_w = int(ref_shape[1] * quant_preview_scale)
    new_h = int(ref_shape[0] * quant_preview_scale)
    quant_preview = cv2.resize(quant_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return {
        "n_layers": n_layers,
        "mode": mode,
        "layer_previews": layer_previews,
        "layer_stats": layer_stats,
        "quantized_preview": _numpy_to_base64_rgb(quant_preview) if mode == "color" else _numpy_to_base64(quant_preview),
    }


@app.get("/api/preview/{session_id}/{layer_index}")
async def get_layer_preview(session_id: str, layer_index: int):
    """Retorna el preview de alta resolucion de una capa especifica."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    session = _sessions[session_id]
    if session["stencils"] is None:
        raise HTTPException(status_code=400, detail="Primero procesa la imagen")

    if layer_index < 0 or layer_index >= len(session["stencils"]):
        raise HTTPException(status_code=400, detail="Indice de capa invalido")

    stencil = session["stencils"][layer_index]
    _, buffer = cv2.imencode(".png", stencil)

    return Response(content=buffer.tobytes(), media_type="image/png")


@app.post("/api/export/pdf")
async def export_pdf(
    session_id: str,
    paper_size: str = "letter",
    dpi: int = 150,
    margin: float = 0.5,
    overlap: float = 0.25,
):
    """
    Exporta todas las capas como PDF multi-pagina.

    Cada capa se divide en tiles con marcas de registro para imprimir
    y ensamblar.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    session = _sessions[session_id]
    if session["stencils"] is None:
        raise HTTPException(status_code=400, detail="Primero procesa la imagen")

    pdf_bytes = generate_full_pdf(
        stencil_layers=session["stencils"],
        paper_size=paper_size,
        dpi=dpi,
        margin=margin,
        overlap=overlap,
        layer_names=session.get("layer_names"),
        layer_colors_hex=session.get("layer_colors"),
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=plantilla_stencil.pdf"},
    )


@app.get("/api/tile-info")
async def get_tile_info(
    session_id: str,
    paper_size: str = "letter",
    dpi: int = 150,
    margin: float = 0.5,
    overlap: float = 0.25,
):
    """Retorna informacion sobre como se dividiran las paginas."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    session = _sessions[session_id]
    if session["stencils"] is None:
        raise HTTPException(status_code=400, detail="Primero procesa la imagen")

    stencil = session["stencils"][0]
    h, w = stencil.shape

    tile_config = calculate_tiles(w, h, paper_size, dpi, margin, overlap)

    return {
        "num_cols": tile_config["num_cols"],
        "num_rows": tile_config["num_rows"],
        "total_pages_per_layer": tile_config["total_pages"],
        "total_pages_all_layers": tile_config["total_pages"] * len(session["stencils"]),
        "tile_size_in": (
            round(tile_config["printable_area_in"][0], 2),
            round(tile_config["printable_area_in"][1], 2),
        ),
        "paper_size": paper_size,
        "dpi": dpi,
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Elimina una sesion y libera memoria."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Sesion no encontrada")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
