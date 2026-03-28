"""
Modulo de division en paginas para impresion.

Divide el stencil final en tiles que caben en hojas tamano carta (8.5x11")
u oficio (8.5x14"), con margenes, overlap para alineacion, y marcas de
registro (cruces, numeracion, mini-diagrama de ensamblaje).

Coordenadas:
    El stencil se escala al tamano fisico deseado segun DPI de impresion.
    Cada tile se extrae con overlap en bordes compartidos para facilitar
    el pegado de las hojas.
"""

import cv2
import numpy as np
from reportlab.lib.pagesizes import letter, legal
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import io
import tempfile


# Tamanos de papel en pulgadas
PAPER_SIZES = {
    "letter": (8.5, 11.0),   # Carta
    "legal": (8.5, 14.0),    # Oficio
}


def calculate_tiles(
    image_width: int,
    image_height: int,
    paper_size: str = "letter",
    dpi: int = 150,
    margin: float = 0.5,
    overlap: float = 0.25,
) -> dict:
    """
    Calcula la distribucion de tiles para dividir una imagen en paginas.

    Args:
        image_width: Ancho de la imagen en pixeles.
        image_height: Alto de la imagen en pixeles.
        paper_size: 'letter' o 'legal'.
        dpi: Puntos por pulgada para impresion.
        margin: Margen en pulgadas (para marcas de registro y bordes).
        overlap: Solapamiento en pulgadas entre tiles adyacentes.

    Returns:
        dict con:
            - 'num_cols': Numero de columnas de tiles
            - 'num_rows': Numero de filas de tiles
            - 'total_pages': Total de paginas
            - 'tile_width_px': Ancho de cada tile en pixeles
            - 'tile_height_px': Alto de cada tile en pixeles
            - 'step_x': Paso horizontal entre tiles (sin overlap)
            - 'step_y': Paso vertical entre tiles (sin overlap)
            - 'tiles': Lista de dicts con info de cada tile
            - 'paper_size_in': Tamano del papel en pulgadas
            - 'printable_area_in': Area imprimible en pulgadas
    """
    paper_w, paper_h = PAPER_SIZES.get(paper_size, PAPER_SIZES["letter"])

    # Area imprimible (descontando margenes)
    printable_w = paper_w - 2 * margin
    printable_h = paper_h - 2 * margin

    # Tamano del tile en pixeles
    tile_w = int(printable_w * dpi)
    tile_h = int(printable_h * dpi)

    # Paso efectivo (area unica por tile, sin overlap)
    overlap_px = int(overlap * dpi)
    step_x = tile_w - overlap_px
    step_y = tile_h - overlap_px

    # Numero de tiles necesarios
    num_cols = max(1, int(np.ceil(image_width / step_x)))
    num_rows = max(1, int(np.ceil(image_height / step_y)))

    # Generar info de cada tile
    tiles = []
    page_num = 1
    for row in range(num_rows):
        for col in range(num_cols):
            x_start = col * step_x
            y_start = row * step_y

            # Ajustar si se sale de la imagen
            x_end = min(x_start + tile_w, image_width)
            y_end = min(y_start + tile_h, image_height)

            tiles.append({
                "page": page_num,
                "row": row,
                "col": col,
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_end,
                "y_end": y_end,
                "width": x_end - x_start,
                "height": y_end - y_start,
            })
            page_num += 1

    return {
        "num_cols": num_cols,
        "num_rows": num_rows,
        "total_pages": len(tiles),
        "tile_width_px": tile_w,
        "tile_height_px": tile_h,
        "step_x": step_x,
        "step_y": step_y,
        "overlap_px": overlap_px,
        "tiles": tiles,
        "paper_size_in": (paper_w, paper_h),
        "printable_area_in": (printable_w, printable_h),
        "dpi": dpi,
        "margin": margin,
    }


def extract_tile(
    stencil: np.ndarray,
    tile_info: dict,
    tile_width_px: int,
    tile_height_px: int,
) -> np.ndarray:
    """
    Extrae un tile de la imagen del stencil.

    Si el tile se sale de la imagen, el area excedente se rellena con blanco.

    Args:
        stencil: Imagen del stencil (H, W), uint8.
        tile_info: Dict con x_start, y_start, width, height.
        tile_width_px: Ancho completo del tile.
        tile_height_px: Alto completo del tile.

    Returns:
        Tile como array (tile_height_px, tile_width_px), uint8.
    """
    tile = np.ones((tile_height_px, tile_width_px), dtype=np.uint8) * 255

    src_x = tile_info["x_start"]
    src_y = tile_info["y_start"]
    src_w = tile_info["width"]
    src_h = tile_info["height"]

    tile[0:src_h, 0:src_w] = stencil[src_y:src_y + src_h, src_x:src_x + src_w]

    return tile


def _draw_registration_marks(
    c: canvas.Canvas,
    paper_w: float,
    paper_h: float,
    margin: float,
    mark_length: float = 0.15,
):
    """Dibuja cruces de registro en las esquinas del area imprimible."""
    c.setStrokeColorRGB(0.5, 0.5, 0.5)
    c.setLineWidth(0.5)

    corners = [
        (margin, margin),
        (paper_w - margin, margin),
        (margin, paper_h - margin),
        (paper_w - margin, paper_h - margin),
    ]

    for cx, cy in corners:
        # Cruz horizontal
        c.line(
            (cx - mark_length) * inch, cy * inch,
            (cx + mark_length) * inch, cy * inch,
        )
        # Cruz vertical
        c.line(
            cx * inch, (cy - mark_length) * inch,
            cx * inch, (cy + mark_length) * inch,
        )


def _draw_assembly_diagram(
    c: canvas.Canvas,
    num_rows: int,
    num_cols: int,
    current_row: int,
    current_col: int,
    x_pos: float,
    y_pos: float,
    cell_size: float = 0.2,
):
    """Dibuja un mini-diagrama mostrando la posicion del tile actual en el ensamblaje."""
    for row in range(num_rows):
        for col in range(num_cols):
            x = x_pos + col * cell_size
            y = y_pos - row * cell_size

            if row == current_row and col == current_col:
                c.setFillColorRGB(0, 0, 0)
                c.rect(
                    x * inch, y * inch,
                    cell_size * inch, cell_size * inch,
                    fill=1,
                )
            else:
                c.setFillColorRGB(0.85, 0.85, 0.85)
                c.rect(
                    x * inch, y * inch,
                    cell_size * inch, cell_size * inch,
                    fill=1, stroke=1,
                )

    c.setFillColorRGB(0, 0, 0)


def generate_pdf(
    stencil: np.ndarray,
    tile_config: dict,
    layer_index: int = 0,
    layer_name: str = "Capa 1",
) -> bytes:
    """
    Genera un PDF multi-pagina con los tiles de un stencil.

    Cada pagina contiene:
    - El tile del stencil
    - Marcas de registro en las esquinas
    - Numero de pagina y coordenadas (fila, columna)
    - Mini-diagrama de ensamblaje
    - Nombre de la capa

    Args:
        stencil: Imagen del stencil (H, W), uint8.
        tile_config: Resultado de calculate_tiles().
        layer_index: Indice de la capa (para numeracion).
        layer_name: Nombre descriptivo de la capa.

    Returns:
        PDF como bytes.
    """
    paper_w, paper_h = tile_config["paper_size_in"]
    margin = tile_config["margin"]
    dpi = tile_config["dpi"]

    page_size = letter if paper_w == 8.5 and paper_h == 11.0 else legal

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size)

    for tile_info in tile_config["tiles"]:
        # Extraer tile
        tile = extract_tile(
            stencil, tile_info,
            tile_config["tile_width_px"],
            tile_config["tile_height_px"],
        )

        # Convertir tile a imagen temporal PNG
        _, img_encoded = cv2.imencode(".png", tile)
        img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img_tmp.write(img_encoded.tobytes())
        img_tmp.flush()

        # Dibujar imagen en la pagina
        printable_w, printable_h = tile_config["printable_area_in"]
        c.drawImage(
            img_tmp.name,
            margin * inch,
            margin * inch,
            width=printable_w * inch,
            height=printable_h * inch,
        )

        img_tmp.close()

        # Marcas de registro
        _draw_registration_marks(c, paper_w, paper_h, margin)

        # Info de pagina (en el margen superior)
        c.setFont("Helvetica", 8)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        header_y = (paper_h - margin / 2) * inch
        c.drawString(
            margin * inch, header_y,
            f"{layer_name} | Pagina {tile_info['page']} de {tile_config['total_pages']}"
            f" | Fila {tile_info['row'] + 1}, Columna {tile_info['col'] + 1}",
        )

        # Mini-diagrama de ensamblaje (esquina inferior derecha del margen)
        diagram_x = paper_w - margin - tile_config["num_cols"] * 0.2
        diagram_y = margin * 0.8
        _draw_assembly_diagram(
            c,
            tile_config["num_rows"],
            tile_config["num_cols"],
            tile_info["row"],
            tile_info["col"],
            diagram_x,
            diagram_y,
        )

        c.showPage()

    c.save()
    return buf.getvalue()


def generate_full_pdf(
    stencil_layers: list,
    paper_size: str = "letter",
    dpi: int = 150,
    margin: float = 0.5,
    overlap: float = 0.25,
    layer_names: list = None,
    layer_colors_hex: list = None,
) -> bytes:
    """
    Genera un PDF completo con todas las capas del stencil.

    Cada capa se divide en paginas independientes. El PDF contiene
    primero todas las paginas de la capa 1, luego la capa 2, etc.

    Args:
        stencil_layers: Lista de arrays (H, W) con cada capa.
        paper_size: 'letter' o 'legal'.
        dpi: DPI de impresion.
        margin: Margen en pulgadas.
        overlap: Solapamiento entre tiles.
        layer_names: Nombres de las capas. Si None, se generan automaticamente.

    Returns:
        PDF como bytes.
    """
    if layer_names is None:
        layer_names = [f"Capa {i + 1}" for i in range(len(stencil_layers))]
    if layer_colors_hex is None:
        layer_colors_hex = [None] * len(stencil_layers)

    paper_w, paper_h = PAPER_SIZES.get(paper_size, PAPER_SIZES["letter"])
    page_size = letter if paper_w == 8.5 and paper_h == 11.0 else legal

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size)

    for layer_idx, (stencil, name, color_hex) in enumerate(zip(stencil_layers, layer_names, layer_colors_hex)):
        h, w = stencil.shape
        tile_config = calculate_tiles(w, h, paper_size, dpi, margin, overlap)

        # Pagina de portada por capa
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(
            paper_w / 2 * inch,
            paper_h / 2 * inch,
            name,
        )
        c.setFont("Helvetica", 14)
        c.drawCentredString(
            paper_w / 2 * inch,
            (paper_h / 2 - 0.5) * inch,
            f"{tile_config['total_pages']} paginas | {tile_config['num_rows']}x{tile_config['num_cols']} tiles",
        )
        c.setFont("Helvetica", 10)
        c.drawCentredString(
            paper_w / 2 * inch,
            (paper_h / 2 - 1.0) * inch,
            f"Papel: {paper_size} | DPI: {dpi} | Overlap: {overlap}\"",
        )

        # Swatch de color si esta disponible
        if color_hex and len(color_hex) == 7:
            try:
                r = int(color_hex[1:3], 16) / 255.0
                g = int(color_hex[3:5], 16) / 255.0
                b = int(color_hex[5:7], 16) / 255.0
                swatch_size = 0.8
                swatch_x = paper_w / 2 - swatch_size / 2
                swatch_y = paper_h / 2 - 2.2
                c.setFillColorRGB(r, g, b)
                c.rect(swatch_x * inch, swatch_y * inch,
                       swatch_size * inch, swatch_size * inch, fill=1, stroke=1)
                c.setFillColorRGB(0, 0, 0)
                c.setFont("Helvetica", 9)
                c.drawCentredString(
                    paper_w / 2 * inch,
                    (swatch_y - 0.2) * inch,
                    f"Color: {color_hex.upper()}  —  Pintar con este color",
                )
            except Exception:
                pass

        c.showPage()

        # Paginas de tiles
        for tile_info in tile_config["tiles"]:
            tile = extract_tile(
                stencil, tile_info,
                tile_config["tile_width_px"],
                tile_config["tile_height_px"],
            )

            # Invertir para impresion: negro = area de corte (lo que se recorta),
            # blanco = material que queda. Mas intuitivo para marcar con tijeras/cutter.
            tile = cv2.bitwise_not(tile)

            _, img_encoded = cv2.imencode(".png", tile)
            img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img_tmp.write(img_encoded.tobytes())
            img_tmp.flush()

            printable_w, printable_h = tile_config["printable_area_in"]
            c.drawImage(
                img_tmp.name,
                margin * inch,
                margin * inch,
                width=printable_w * inch,
                height=printable_h * inch,
            )
            img_tmp.close()

            _draw_registration_marks(c, paper_w, paper_h, margin)

            c.setFont("Helvetica", 8)
            c.setFillColorRGB(0.3, 0.3, 0.3)
            header_y = (paper_h - margin / 2) * inch
            c.drawString(
                margin * inch, header_y,
                f"{name} | Pagina {tile_info['page']} de {tile_config['total_pages']}"
                f" | Fila {tile_info['row'] + 1}, Col {tile_info['col'] + 1}",
            )

            diagram_x = paper_w - margin - tile_config["num_cols"] * 0.2
            diagram_y = margin * 0.8
            _draw_assembly_diagram(
                c,
                tile_config["num_rows"],
                tile_config["num_cols"],
                tile_info["row"],
                tile_info["col"],
                diagram_x,
                diagram_y,
            )

            c.showPage()

    c.save()
    return buf.getvalue()
