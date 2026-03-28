"""
Modulo de remocion de fondo usando U2-Net (via rembg).

U2-Net es una red neuronal de segmentacion de objetos salientes (salient object
detection) que identifica el sujeto principal de una imagen y genera una mascara
de alta calidad. Esto permite aislar el sujeto antes de generar el stencil.

Arquitectura U2-Net:
- Encoder-decoder anidado con bloques RSU (Residual U-blocks)
- Entrenado en DUTS-TR dataset (10,553 imagenes)
- Genera mascara de probabilidad por pixel

Pipeline:
    Imagen original -> U2-Net -> Mascara alfa -> Imagen sin fondo
"""

import io
import numpy as np
from PIL import Image
from rembg import remove, new_session


# Sesion singleton para no recargar el modelo en cada request
_session = None


def _get_session():
    """Carga lazy del modelo U2-Net. Se inicializa una sola vez."""
    global _session
    if _session is None:
        # u2net: modelo general, buen balance precision/velocidad
        # u2netp: version ligera, mas rapido pero menos preciso
        # u2net_human_seg: optimizado para personas
        _session = new_session("u2net")
    return _session


def remove_background(
    image: Image.Image,
    model: str = "u2net",
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
) -> dict:
    """
    Remueve el fondo de una imagen usando U2-Net.

    Args:
        image: Imagen PIL en modo RGB o RGBA.
        model: Modelo a usar ('u2net', 'u2netp', 'u2net_human_seg').
        alpha_matting: Si True, refina bordes con alpha matting (mas lento
                       pero bordes mas suaves).
        alpha_matting_foreground_threshold: Umbral para pixeles de primer plano.
        alpha_matting_background_threshold: Umbral para pixeles de fondo.

    Returns:
        dict con:
            - 'image': Imagen PIL sin fondo (RGBA)
            - 'mask': Mascara binaria como array numpy (H, W), 0=fondo, 255=sujeto
            - 'image_white_bg': Imagen con fondo blanco (RGB)
    """
    global _session

    # Recargar sesion si cambia el modelo
    if _session is None or model != "u2net":
        _session = new_session(model)
    else:
        _get_session()

    # rembg espera bytes de imagen
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Ejecutar U2-Net para remocion de fondo
    result_bytes = remove(
        img_bytes.getvalue(),
        session=_session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
    )

    # Resultado es RGBA con fondo transparente
    result_image = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

    # Extraer mascara del canal alfa
    alpha_channel = np.array(result_image)[:, :, 3]
    mask = (alpha_channel > 128).astype(np.uint8) * 255

    # Version con fondo blanco (util para procesamiento posterior)
    white_bg = Image.new("RGB", result_image.size, (255, 255, 255))
    white_bg.paste(result_image, mask=result_image.split()[3])

    return {
        "image": result_image,
        "mask": mask,
        "image_white_bg": white_bg,
    }


def apply_custom_background(image_rgba: Image.Image, color: tuple = (255, 255, 255)) -> Image.Image:
    """
    Aplica un color de fondo solido a una imagen RGBA.

    Args:
        image_rgba: Imagen PIL en modo RGBA (con transparencia).
        color: Tuple RGB del color de fondo.

    Returns:
        Imagen PIL en modo RGB con el fondo aplicado.
    """
    background = Image.new("RGB", image_rgba.size, color)
    background.paste(image_rgba, mask=image_rgba.split()[3])
    return background
