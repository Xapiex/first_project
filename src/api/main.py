from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.models.predict_model import (
    MODEL_CONFIG,
    get_models_status,
    predict,
)


# ============================================================
# MODELOS DE RESPUESTA
# ============================================================

class PredictionResult(BaseModel):
    model: str
    architecture: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    validation_accuracy: float
    validation_error_rate: float


class PredictionResponse(PredictionResult):
    filename: str


class CompareResponse(BaseModel):
    filename: str
    results: Dict[str, PredictionResult]
    highest_confidence_model: str
    best_validation_model: str


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Cats vs Dogs - MLOps API",
    description=(
        "Sistema de clasificación de gatos y perros "
        "utilizando ResNet34 y EfficientNet-B0 con "
        "FastAI, PyTorch, FastAPI y Docker."
    ),
    version="3.0.0",
)


ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp",
}


# ============================================================
# UTILIDADES
# ============================================================

def validate_file(file: UploadFile):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ningún archivo."
        )

    extension = (
        Path(file.filename)
        .suffix
        .lower()
        .lstrip(".")
    )

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Formato no soportado. "
                "Usa JPG, JPEG, PNG, BMP o WEBP."
            ),
        )

    return extension


async def save_temp_image(
    file: UploadFile,
    extension: str
):

    with NamedTemporaryFile(
        delete=False,
        suffix=f".{extension}"
    ) as temp_file:

        temp_file.write(await file.read())

        return temp_file.name


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "Cats vs Dogs - MLOps",
        "status": "API funcionando correctamente",
        "version": "3.0.0",
        "demo": "/demo",
        "documentation": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "models": get_models_status(),
    }


# ============================================================
# MODELOS
# ============================================================

@app.get("/models")
def models():

    return {
        "available_models": get_models_status()
    }


# ============================================================
# PREDICCIÓN INDIVIDUAL
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Clasificar una imagen",
)
async def predict_image(

    file: UploadFile = File(...),

    model: str = Query(
        default="resnet34",
        description=(
            "Modelo: resnet34 o efficientnet_b0"
        ),
    ),
):

    if model not in MODEL_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modelo inválido. Disponibles: "
                f"{list(MODEL_CONFIG.keys())}"
            ),
        )

    extension = validate_file(file)

    temp_path = None

    try:

        temp_path = await save_temp_image(
            file,
            extension
        )

        result = predict(
            temp_path,
            model_name=model
        )

        return {
            "filename": file.filename,
            **result,
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error al realizar "
                f"la predicción: {str(exc)}"
            )
        )

    finally:

        if (
            temp_path
            and Path(temp_path).exists()
        ):
            Path(temp_path).unlink(
                missing_ok=True
            )


# ============================================================
# COMPARACIÓN
# ============================================================

@app.post(
    "/predict/compare",
    response_model=CompareResponse,
    summary="Comparar ResNet34 y EfficientNet-B0",
)
async def compare_models(
    file: UploadFile = File(...)
):

    extension = validate_file(file)

    temp_path = None

    try:

        temp_path = await save_temp_image(
            file,
            extension
        )

        results = {}

        for model_name in MODEL_CONFIG:

            results[model_name] = predict(
                temp_path,
                model_name=model_name
            )

        highest_confidence_model = max(
            results,
            key=lambda name: (
                results[name]["confidence"]
            )
        )

        best_validation_model = max(
            MODEL_CONFIG,
            key=lambda name: (
                MODEL_CONFIG[name][
                    "validation_accuracy"
                ]
            )
        )

        return {
            "filename": file.filename,
            "results": results,
            "highest_confidence_model":
                highest_confidence_model,
            "best_validation_model":
                best_validation_model,
        }

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error al comparar "
                f"modelos: {str(exc)}"
            )
        )

    finally:

        if (
            temp_path
            and Path(temp_path).exists()
        ):
            Path(temp_path).unlink(
                missing_ok=True
            )


# ============================================================
# DEMO WEB
# ============================================================

DEMO_HTML = """
<!DOCTYPE html>
<html lang="es">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Cats vs Dogs - MLOps
</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Inter,
        Segoe UI,
        Arial,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #0f172a,
            #172554
        );

    min-height: 100vh;
    color: #e2e8f0;
}

.container {
    max-width: 1150px;
    margin: auto;
    padding: 45px 25px;
}

header {
    text-align: center;
    margin-bottom: 35px;
}

h1 {
    margin: 0;
    font-size: 42px;
    color: white;
}

.subtitle {
    color: #94a3b8;
    margin-top: 10px;
}

.tech {
    margin-top: 18px;
}

.tech span {
    background: rgba(
        255,
        255,
        255,
        0.08
    );

    padding: 7px 13px;
    border-radius: 20px;
    margin: 4px;
    display: inline-block;
    font-size: 13px;
}

.panel {
    background: rgba(
        255,
        255,
        255,
        0.07
    );

    border: 1px solid rgba(
        255,
        255,
        255,
        0.12
    );

    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(10px);
}

.upload-area {
    text-align: center;
}

#preview {
    display: none;

    max-width: 300px;
    max-height: 250px;

    object-fit: contain;

    margin: 20px auto;

    border-radius: 14px;
}

input[type=file] {
    margin: 20px;
}

button {
    border: none;

    padding: 13px 28px;

    border-radius: 10px;

    background:
        linear-gradient(
            135deg,
            #3b82f6,
            #6366f1
        );

    color: white;

    font-size: 16px;

    font-weight: 600;

    cursor: pointer;
}

button:hover {
    transform: translateY(-1px);
}

button:disabled {
    opacity: 0.5;
    cursor: wait;
}

#status {
    text-align: center;
    margin: 20px;
    min-height: 24px;
}

.results {
    margin-top: 30px;

    display: grid;

    grid-template-columns:
        repeat(
            2,
            minmax(0, 1fr)
        );

    gap: 20px;
}

.card {
    background: rgba(
        255,
        255,
        255,
        0.08
    );

    border: 1px solid rgba(
        255,
        255,
        255,
        0.12
    );

    border-radius: 18px;

    padding: 25px;
}

.card h2 {
    margin-top: 0;
    color: white;
}

.prediction {
    font-size: 30px;
    font-weight: bold;

    margin: 18px 0;
}

.row {
    display: flex;

    justify-content: space-between;

    border-bottom: 1px solid rgba(
        255,
        255,
        255,
        0.08
    );

    padding: 10px 0;
}

.value {
    font-weight: 600;
    color: white;
}

.bar-container {
    margin-top: 15px;
}

.bar-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
}

.bar-bg {
    height: 9px;

    background: rgba(
        255,
        255,
        255,
        0.12
    );

    border-radius: 10px;

    overflow: hidden;
}

.bar {
    height: 100%;

    background:
        linear-gradient(
            90deg,
            #38bdf8,
            #6366f1
        );
}

.summary {
    margin-top: 25px;

    display: grid;

    grid-template-columns:
        repeat(
            2,
            minmax(0, 1fr)
        );

    gap: 15px;
}

.summary-box {
    background: rgba(
        15,
        23,
        42,
        0.7
    );

    padding: 18px;

    border-radius: 15px;

    text-align: center;
}

.summary-value {
    font-size: 20px;

    font-weight: bold;

    color: #38bdf8;

    margin-top: 7px;
}

.links {
    margin-top: 30px;
    text-align: center;
}

.links a {
    color: #93c5fd;
    margin: 0 12px;
    text-decoration: none;
}

@media (
    max-width: 800px
) {

    .results,
    .summary {
        grid-template-columns: 1fr;
    }

    h1 {
        font-size: 32px;
    }
}

</style>

</head>


<body>

<div class="container">

<header>

<h1>
🐱 Cats vs Dogs 🐶
</h1>

<div class="subtitle">
Sistema de clasificación de imágenes
con comparación de modelos
</div>

<div class="tech">

<span>FastAI</span>
<span>PyTorch</span>
<span>FastAPI</span>
<span>Docker</span>
<span>MLflow</span>
<span>API v3.0.0</span>

</div>

</header>


<div class="panel upload-area">

<h2>
Analizar imagen
</h2>

<p>
Selecciona una fotografía de un gato
o un perro.
</p>

<input
    id="fileInput"
    type="file"
    accept=".jpg,.jpeg,.png,.bmp,.webp"
>

<br>

<img
    id="preview"
    alt="Vista previa"
>

<br>

<button
    id="analyzeButton"
    onclick="analyzeImage()"
>
Comparar modelos
</button>

<div id="status"></div>

</div>


<div
    id="results"
    class="results"
></div>


<div
    id="summary"
    class="summary"
    style="display:none;"
>

<div class="summary-box">

<div>
Mayor confianza en esta imagen
</div>

<div
    class="summary-value"
    id="confidenceWinner"
></div>

</div>


<div class="summary-box">

<div>
Mejor accuracy de validación
</div>

<div
    class="summary-value"
    id="validationWinner"
></div>

</div>

</div>


<div class="links">

<a href="/docs">
Documentación Swagger
</a>

<a href="/models">
Información de modelos
</a>

<a href="/health">
Estado del servicio
</a>

</div>

</div>


<script>

const fileInput =
    document.getElementById(
        "fileInput"
    );

const preview =
    document.getElementById(
        "preview"
    );


fileInput.addEventListener(
    "change",
    function () {

        const file = this.files[0];

        if (!file) {
            return;
        }

        preview.src =
            URL.createObjectURL(file);

        preview.style.display =
            "block";
    }
);


function modelCard(data) {

    const dog =
        data.probabilities.PERRO;

    const cat =
        data.probabilities.GATO;

    return `
        <div class="card">

            <h2>
                ${data.architecture}
            </h2>

            <div class="prediction">
                ${
                    data.prediction === "GATO"
                    ? "🐱 GATO"
                    : "🐶 PERRO"
                }
            </div>

            <div class="row">
                <span>Confianza</span>
                <span class="value">
                    ${data.confidence} %
                </span>
            </div>

            <div class="row">
                <span>
                    Accuracy validación
                </span>

                <span class="value">
                    ${data.validation_accuracy} %
                </span>
            </div>

            <div class="row">
                <span>
                    Error rate
                </span>

                <span class="value">
                    ${data.validation_error_rate} %
                </span>
            </div>

            <div class="row">
                <span>
                    Tiempo inferencia
                </span>

                <span class="value">
                    ${data.inference_time_ms} ms
                </span>
            </div>


            <div class="bar-container">

                <div class="bar-label">
                    <span>🐶 Perro</span>
                    <span>${dog} %</span>
                </div>

                <div class="bar-bg">
                    <div
                        class="bar"
                        style="width:${dog}%"
                    ></div>
                </div>

            </div>


            <div class="bar-container">

                <div class="bar-label">
                    <span>🐱 Gato</span>
                    <span>${cat} %</span>
                </div>

                <div class="bar-bg">
                    <div
                        class="bar"
                        style="width:${cat}%"
                    ></div>
                </div>

            </div>

        </div>
    `;
}


async function analyzeImage() {

    const input =
        document.getElementById(
            "fileInput"
        );

    const button =
        document.getElementById(
            "analyzeButton"
        );

    const status =
        document.getElementById(
            "status"
        );

    const results =
        document.getElementById(
            "results"
        );

    if (!input.files.length) {

        status.innerHTML =
            "Selecciona una imagen primero.";

        return;
    }


    const formData =
        new FormData();

    formData.append(
        "file",
        input.files[0]
    );


    button.disabled = true;

    status.innerHTML =
        "Analizando imagen con ambos modelos...";

    results.innerHTML = "";


    try {

        const response =
            await fetch(
                "/predict/compare",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            const error =
                await response.json();

            throw new Error(
                error.detail
                || "Error en la API"
            );
        }


        const data =
            await response.json();


        results.innerHTML =
            modelCard(
                data.results.resnet34
            )
            +
            modelCard(
                data.results.efficientnet_b0
            );


        const confidence =
            data.results[
                data.highest_confidence_model
            ];


        const validation =
            data.results[
                data.best_validation_model
            ];


        document.getElementById(
            "confidenceWinner"
        ).innerHTML =
            confidence.architecture
            + " — "
            + confidence.confidence
            + " %";


        document.getElementById(
            "validationWinner"
        ).innerHTML =
            validation.architecture
            + " — "
            + validation.validation_accuracy
            + " %";


        document.getElementById(
            "summary"
        ).style.display =
            "grid";


        status.innerHTML =
            "✅ Análisis completado";

    }

    catch (error) {

        status.innerHTML =
            "❌ "
            + error.message;
    }

    finally {

        button.disabled = false;
    }
}

</script>

</body>

</html>
"""


@app.get(
    "/demo",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def demo():

    return HTMLResponse(
        content=DEMO_HTML
    )