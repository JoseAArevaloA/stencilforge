# Generador de Stencils Multi-Capa

Convierte fotos o ilustraciones en plantillas (stencils) listas para imprimir y pintar, usando Machine Learning y Computer Vision.

---

## Cómo funciona

```
Foto  →  [Remover fondo]  →  [K-Means: N capas tonales]  →  [Canny: bordes]
      →  [Limpiar islas]  →  [Añadir puentes]  →  PDF por capas
```

Cada capa tonal se convierte en una plantilla independiente. Las imprimes, recortas y usas una por una con tu aerosol o pintura.

---

## Inicio rápido

### 1. Instalar dependencias (solo la primera vez)

```bash
cd stencilforge
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-multipart Pillow numpy opencv-python-headless scikit-learn reportlab
pip install --timeout 600 onnxruntime
pip install rembg
```

### 2. Iniciar el backend

```bash
cd backend
source ../venv/bin/activate
uvicorn app:app --port 8000
# o con recarga automática durante desarrollo:
uvicorn app:app --port 8000 --reload
```

### 3. Abrir el frontend

```bash
# Opción A — directo en el navegador
xdg-open frontend/index.html

# Opción B — con servidor HTTP
python3 -m http.server 3000 --directory frontend
# luego abrir http://localhost:3000
```

---

## Flujo de uso

### Paso 1 — Subir imagen

- Arrastra una foto al área central o haz clic en "Seleccionar imagen"
- Formatos soportados: JPG, PNG, WEBP
- Las imágenes mayores a 4000px se redimensionan automáticamente

### Paso 2 — Remover fondo (opcional)

- Clic en **"Remover fondo"** para aislar el sujeto con U2-Net
- El primer uso descarga el modelo (~170 MB), los siguientes son instantáneos
- Recomendado para fotos con fondo complejo; omitir si la imagen ya tiene fondo blanco

### Paso 3 — Analizar capas (opcional)

- Clic en **"Analizar capas óptimas"**
- Muestra el Silhouette Score y el Elbow Method para K-Means
- El sistema sugiere el número de capas ideal según la distribución tonal

### Paso 4 — Generar stencil

Ajusta los parámetros en el panel izquierdo:

| Parámetro | Descripción | Rango recomendado |
|-----------|-------------|-------------------|
| **Número de capas** | Cuántas capas tonales genera K-Means | 2–5 |
| **Detalle de bordes** | Sensibilidad de Canny (low/medium/high) | medium |
| **Peso de bordes** | Cuánto influyen los bordes en la máscara | 0.2–0.5 |
| **Tamaño mínimo isla** | Elimina manchas menores a N píxeles | 50–200 |
| **Ancho de puentes** | Grosor de los puentes que conectan islas | 2–5 px |

Luego clic en **"Generar Stencil"**. Verás una preview de cada capa.

### Paso 5 — Exportar PDF

- Clic en **"Info de páginas"** para ver cuántas hojas ocupará tu stencil
- Ajusta el tamaño de papel (carta / oficio) y DPI si necesitas
- Clic en **"Exportar PDF"** — descarga un PDF con todas las capas, divididas en tiles con marcas de registro para ensamblar

---

## Consejos para mejores resultados

- **Imagen fuente**: fotos con buen contraste, iluminación uniforme y fondo simple
- **Número de capas**: 2–3 para logos/diseños simples, 4–5 para retratos con degradados
- **Bordes**: usa `high` para ilustraciones con líneas finas, `low` para manchas grandes
- **Islas pequeñas**: aumenta el mínimo si el stencil tiene demasiados puntos diminutos que son imposibles de recortar
- **PDF a tamaño real**: imprime al 100% (sin "ajustar a página") y usa las marcas de registro para alinear los tiles

---

## Estructura del proyecto

```
generate-stencil/
├── backend/
│   ├── app.py                  ← API FastAPI (8 endpoints)
│   ├── requirements.txt
│   └── pipeline/
│       ├── background.py       ← Remoción de fondo (U2-Net / rembg)
│       ├── edges.py            ← Detección de bordes Canny adaptativo
│       ├── quantizer.py        ← Cuantización tonal K-Means (sklearn)
│       ├── stencil_builder.py  ← Islas (CCL) + puentes + morfología
│       └── page_splitter.py    ← Tiles + marcas de registro + PDF
└── frontend/
    ├── index.html              ← UI completa
    └── js/
        └── app.js              ← Comunicación con la API
```

## API (para uso programático)

El backend expone una API REST en `http://localhost:8000`. Documentación interactiva en `http://localhost:8000/docs`.

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/upload` | Sube imagen → `session_id` |
| POST | `/api/remove-bg` | Remueve fondo con U2-Net |
| POST | `/api/analyze-layers` | Métricas K-Means (silhouette, elbow) |
| POST | `/api/process` | Pipeline completo → previews por capa |
| GET | `/api/preview/{id}/{layer}` | PNG de alta resolución de una capa |
| POST | `/api/export/pdf` | PDF multi-página con todas las capas |
| GET | `/api/tile-info` | Info de distribución en páginas |
| DELETE | `/api/session/{id}` | Libera memoria de la sesión |

---

## Tecnologías

- **U2-Net** — segmentación semántica para remoción de fondo (Deep Learning)
- **K-Means** — cuantización tonal por clustering de píxeles (ML clásico)
- **Canny adaptativo** — detección de bordes con umbrales automáticos (OpenCV)
- **CCL (Connected Component Labeling)** — detección de islas flotantes
- **ReportLab** — generación de PDF con tiles y marcas de registro
- **FastAPI + Uvicorn** — backend async
