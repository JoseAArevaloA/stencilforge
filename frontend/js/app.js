/**
 * Stencil Generator - Frontend Application
 *
 * Orquesta la comunicacion con el backend FastAPI y gestiona
 * el estado de la UI.
 */

const API_BASE = "http://localhost:8000/api";

// Estado global
const state = {
    sessionId: null,
    hasImage: false,
    bgRemoved: false,
    analyzed: false,
    processed: false,
};

// --- Elementos del DOM ---
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    fileInput: $("#fileInput"),
    uploadZone: $("#uploadZone"),
    loadingOverlay: $("#loadingOverlay"),
    loadingText: $("#loadingText"),
    // Secciones
    welcomeState: $("#welcomeState"),
    previewState: $("#previewState"),
    metricsState: $("#metricsState"),
    layersState: $("#layersState"),
    tileState: $("#tileState"),
    // Controles
    bgSection: $("#bgSection"),
    analysisSection: $("#analysisSection"),
    stencilSection: $("#stencilSection"),
    exportSection: $("#exportSection"),
    // Previews
    previewOriginal: $("#previewOriginal"),
    previewNoBg: $("#previewNoBg"),
    previewBgCard: $("#previewBgCard"),
    previewQuantized: $("#previewQuantized"),
    // Buttons
    btnRemoveBg: $("#btnRemoveBg"),
    btnAnalyze: $("#btnAnalyze"),
    btnProcess: $("#btnProcess"),
    btnExport: $("#btnExport"),
    btnTileInfo: $("#btnTileInfo"),
    // Status
    statusLeft: $("#statusLeft"),
    statusRight: $("#statusRight"),
    sessionInfo: $("#sessionInfo"),
    bgStatus: $("#bgStatus"),
    suggestedK: $("#suggestedK"),
    layersGrid: $("#layersGrid"),
    tileInfoText: $("#tileInfoText"),
    tileGridVisual: $("#tileGridVisual"),
};

// --- Utilidades ---

function showLoading(text = "Procesando...") {
    els.loadingText.textContent = text;
    els.loadingOverlay.classList.add("active");
}

function hideLoading() {
    els.loadingOverlay.classList.remove("active");
}

function setStatus(text) {
    els.statusLeft.textContent = text;
}

function show(el) {
    el.classList.remove("hidden");
}

function hide(el) {
    el.classList.add("hidden");
}

async function apiCall(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, options);
    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Error desconocido" }));
        const msg = err.detail || `Error ${response.status}`;
        // Sesion expirada: guiar al usuario a re-subir la imagen
        if (response.status === 404 && msg.toLowerCase().includes("sesion")) {
            state.sessionId = null;
            state.hasImage = false;
            state.processed = false;
            hide(els.previewState);
            hide(els.metricsState);
            hide(els.layersState);
            hide(els.tileState);
            hide(els.exportSection);
            show(els.welcomeState);
            throw new Error("La sesión expiró (el servidor se reinició). Vuelve a subir la imagen.");
        }
        throw new Error(msg);
    }
    return response;
}

// --- Upload ---

function initUpload() {
    els.uploadZone.addEventListener("click", () => els.fileInput.click());

    // Drop zone en el panel lateral
    els.uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        els.uploadZone.classList.add("dragover");
    });
    els.uploadZone.addEventListener("dragleave", () => {
        els.uploadZone.classList.remove("dragover");
    });
    els.uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        els.uploadZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    // Drop zone en el area central (toda la pantalla)
    document.addEventListener("dragover", (e) => {
        e.preventDefault();
        if (els.welcomeState && !els.welcomeState.classList.contains("hidden")) {
            els.welcomeState.style.outline = "3px dashed var(--accent)";
            els.welcomeState.style.borderRadius = "12px";
        }
    });
    document.addEventListener("dragleave", (e) => {
        if (e.relatedTarget === null) {
            if (els.welcomeState) els.welcomeState.style.outline = "";
        }
    });
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        if (els.welcomeState) els.welcomeState.style.outline = "";
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    els.fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });
}

async function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Por favor selecciona un archivo de imagen.");
        return;
    }

    showLoading("Subiendo imagen...");
    setStatus("Subiendo imagen...");

    try {
        const formData = new FormData();
        formData.append("file", file);

        const resp = await apiCall("/upload", {
            method: "POST",
            body: formData,
        });

        const data = await resp.json();
        state.sessionId = data.session_id;
        state.hasImage = true;

        // Mostrar preview
        els.previewOriginal.src = `data:image/png;base64,${data.preview}`;

        hide(els.welcomeState);
        show(els.previewState);
        show(els.bgSection);
        show(els.analysisSection);
        show(els.stencilSection);

        els.sessionInfo.textContent = `Sesion: ${data.session_id} | ${data.width}x${data.height}px`;
        show(els.sessionInfo);

        if (data.was_resized) {
            setStatus(`Imagen redimensionada a ${data.width}x${data.height}px`);
        } else {
            setStatus(`Imagen cargada: ${data.width}x${data.height}px`);
        }

        els.statusRight.textContent = `${data.width}x${data.height}px`;
    } catch (err) {
        alert(`Error al subir: ${err.message}`);
        setStatus("Error al subir imagen");
    } finally {
        hideLoading();
    }
}

// --- Remove Background ---

async function handleRemoveBg() {
    if (!state.sessionId) return;

    const model = $("#bgModel").value;
    showLoading("Removiendo fondo con U2-Net... (puede tomar un momento)");
    setStatus("Ejecutando U2-Net...");

    try {
        const resp = await apiCall(
            `/remove-bg?session_id=${state.sessionId}&model=${model}`,
            { method: "POST" }
        );

        const data = await resp.json();
        state.bgRemoved = true;

        els.previewNoBg.src = `data:image/png;base64,${data.preview}`;
        els.previewNoBg.style.display = "block";
        show(els.previewBgCard);
        els.bgStatus.textContent = "Listo";
        els.bgStatus.style.color = "var(--success)";

        setStatus("Fondo removido exitosamente");
    } catch (err) {
        alert(`Error al remover fondo: ${err.message}`);
        setStatus("Error en U2-Net");
    } finally {
        hideLoading();
    }
}

// --- Analyze Layers ---

async function handleAnalyze() {
    if (!state.sessionId) return;

    showLoading("Analizando capas optimas con K-Means...");
    setStatus("Calculando metricas de clustering...");

    try {
        const resp = await apiCall(
            `/analyze-layers?session_id=${state.sessionId}&max_k=6`,
            { method: "POST" }
        );

        const data = await resp.json();
        state.analyzed = true;

        renderMetrics(data);
        show(els.metricsState);

        // Actualizar slider al valor sugerido
        $("#nLayers").value = data.suggested_k;
        $("#layersValue").textContent = data.suggested_k;

        els.suggestedK.textContent =
            `K-Means sugiere ${data.suggested_k} capas (mejor Silhouette Score: ${data.silhouette_scores[data.suggested_k - 2].toFixed(3)})`;

        setStatus("Analisis completado");
    } catch (err) {
        alert(`Error en analisis: ${err.message}`);
        setStatus("Error en analisis");
    } finally {
        hideLoading();
    }
}

function renderMetrics(data) {
    const { k_range, silhouette_scores, inertias } = data;

    // Silhouette bars
    const maxSil = Math.max(...silhouette_scores);
    let silHTML = "";
    for (let i = 0; i < k_range.length; i++) {
        const pct = (silhouette_scores[i] / maxSil) * 100;
        const isBest = silhouette_scores[i] === maxSil;
        silHTML += `
            <div class="metric-bar-row">
                <span class="label">K=${k_range[i]}</span>
                <div class="bar-container">
                    <div class="bar bar-silhouette" style="width:${pct}%; opacity:${isBest ? 1 : 0.6}"></div>
                </div>
                <span class="value" style="${isBest ? 'color:var(--accent)' : ''}">${silhouette_scores[i].toFixed(3)}</span>
            </div>`;
    }
    $("#silhouetteChart").innerHTML = silHTML;

    // Inertia bars
    const maxInertia = Math.max(...inertias);
    let inerHTML = "";
    for (let i = 0; i < k_range.length; i++) {
        const pct = (inertias[i] / maxInertia) * 100;
        inerHTML += `
            <div class="metric-bar-row">
                <span class="label">K=${k_range[i]}</span>
                <div class="bar-container">
                    <div class="bar bar-inertia" style="width:${pct}%"></div>
                </div>
                <span class="value">${(inertias[i] / 1000).toFixed(1)}k</span>
            </div>`;
    }
    $("#inertiaChart").innerHTML = inerHTML;
}

// --- Process Stencil ---

async function handleProcess() {
    if (!state.sessionId) return;

    const mode = document.querySelector('input[name="mode"]:checked')?.value || "grayscale";
    const params = new URLSearchParams({
        session_id: state.sessionId,
        n_layers: $("#nLayers").value,
        min_island_size: $("#minIslandSize").value,
        bridge_width: $("#bridgeWidth").value,
        bridges_per_island: $("#bridgesPerIsland").value,
        edge_detail: $("#edgeDetail").value,
        edge_weight: (parseInt($("#edgeWeight").value) / 100).toFixed(2),
        morph_clean: $("#morphClean").checked,
        use_edges: $("#useEdges").checked,
        mode,
    });

    // Limpiar resultados anteriores antes de procesar
    els.layersGrid.innerHTML = "";
    hide(els.layersState);
    hide(els.exportSection);
    hide(els.tileState);
    state.processed = false;

    showLoading("Generando stencil... (K-Means + bordes + islas + puentes)");
    setStatus("Procesando pipeline completo...");

    try {
        const resp = await apiCall(`/process?${params}`, { method: "POST" });
        const data = await resp.json();
        state.processed = true;

        // Preview cuantizada
        els.previewQuantized.src = `data:image/png;base64,${data.quantized_preview}`;

        // Capas
        renderLayers(data);
        show(els.layersState);
        show(els.exportSection);

        setStatus(`Stencil generado: ${data.n_layers} capas`);
    } catch (err) {
        alert(`Error al procesar: ${err.message}`);
        setStatus("Error al procesar");
    } finally {
        hideLoading();
    }
}

function renderLayers(data) {
    let html = "";
    for (const stat of data.layer_stats) {
        const colorSwatch = stat.color_hex
            ? `<span style="display:inline-block; width:14px; height:14px; border-radius:3px;
                            background:${stat.color_hex}; border:1px solid rgba(255,255,255,0.3);
                            vertical-align:middle; margin-right:4px;"></span>
               <span style="font-family:monospace; font-size:0.75rem;">${stat.color_hex.toUpperCase()}</span>`
            : `<span style="color:var(--text-secondary); font-size:0.8rem;">tono: ${stat.tonal_center}</span>`;

        html += `
            <div class="layer-card">
                <div class="card-header">
                    <span>${stat.name.split(" (")[0]}</span>
                    <span>${colorSwatch}</span>
                </div>
                <img src="data:image/png;base64,${data.layer_previews[stat.layer]}" alt="${stat.name}">
                <div class="card-stats">
                    <span>Cobertura: ${stat.coverage_pct}%</span>
                    <span>Islas: ${stat.islands_detected}</span>
                    <span>Puentes: ${stat.bridges_added}</span>
                </div>
            </div>`;
    }
    els.layersGrid.innerHTML = html;
}

// --- Tile Info ---

async function handleTileInfo() {
    if (!state.sessionId || !state.processed) return;

    const params = new URLSearchParams({
        session_id: state.sessionId,
        paper_size: $("#paperSize").value,
        dpi: $("#dpi").value,
    });

    try {
        const resp = await apiCall(`/tile-info?${params}`);
        const data = await resp.json();

        els.tileInfoText.textContent =
            `${data.total_pages_per_layer} paginas por capa x ${data.layer_stats?.length || parseInt($("#nLayers").value)} capas = ` +
            `${data.total_pages_all_layers} paginas total | ` +
            `Grid: ${data.num_rows} filas x ${data.num_cols} columnas | ` +
            `Area imprimible: ${data.tile_size_in[0]}" x ${data.tile_size_in[1]}" | ` +
            `DPI: ${data.dpi}`;

        // Visual grid
        let gridHTML = "";
        els.tileGridVisual.style.gridTemplateColumns = `repeat(${data.num_cols}, 24px)`;
        let page = 1;
        for (let r = 0; r < data.num_rows; r++) {
            for (let c = 0; c < data.num_cols; c++) {
                gridHTML += `<div class="tile-cell">${page}</div>`;
                page++;
            }
        }
        els.tileGridVisual.innerHTML = gridHTML;

        show(els.tileState);
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
}

// --- Export PDF ---

async function handleExport() {
    if (!state.sessionId || !state.processed) return;

    const params = new URLSearchParams({
        session_id: state.sessionId,
        paper_size: $("#paperSize").value,
        dpi: $("#dpi").value,
    });

    showLoading("Generando PDF multi-pagina...");
    setStatus("Exportando PDF...");

    try {
        const resp = await fetch(`${API_BASE}/export/pdf?${params}`, {
            method: "POST",
        });

        if (!resp.ok) throw new Error("Error al generar PDF");

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "plantilla_stencil.pdf";
        a.click();
        URL.revokeObjectURL(url);

        setStatus("PDF descargado exitosamente");
    } catch (err) {
        alert(`Error al exportar: ${err.message}`);
        setStatus("Error al exportar PDF");
    } finally {
        hideLoading();
    }
}

// --- Slider value displays ---

function initSliders() {
    const sliders = [
        { id: "nLayers", display: "layersValue" },
        { id: "edgeWeight", display: "edgeWeightValue", transform: (v) => (v / 100).toFixed(2) },
        { id: "minIslandSize", display: "islandSizeValue" },
        { id: "bridgeWidth", display: "bridgeWidthValue" },
        { id: "bridgesPerIsland", display: "bridgesPerIslandValue" },
        { id: "dpi", display: "dpiValue" },
    ];

    for (const { id, display, transform } of sliders) {
        const input = $(`#${id}`);
        const disp = $(`#${display}`);
        input.addEventListener("input", () => {
            disp.textContent = transform ? transform(input.value) : input.value;
        });
    }
}

// --- Init ---

function init() {
    initUpload();
    initSliders();

    els.btnRemoveBg.addEventListener("click", handleRemoveBg);
    els.btnAnalyze.addEventListener("click", handleAnalyze);
    els.btnProcess.addEventListener("click", handleProcess);
    els.btnExport.addEventListener("click", handleExport);
    els.btnTileInfo.addEventListener("click", handleTileInfo);
}

init();
