from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.models.predict_model import predict

app = FastAPI(
    title="First Project API",
    description="API para el proyecto de clasificación de imágenes",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "API funcionando correctamente"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Recibe una imagen, la evalúa con el modelo entrenado y retorna la predicción."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se recibió ningún archivo.")

    allowed = {"jpg", "jpeg", "png", "bmp", "webp"}
    extension = Path(file.filename).suffix.lower().lstrip(".")
    if extension not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Tipo de archivo no soportado. Usa JPG, JPEG, PNG, BMP o WEBP."
        )

    try:
        with NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        result, confidence = predict(temp_path)

        return {
            "filename": file.filename,
            "prediction": result,
            "confidence": round(float(confidence), 2),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la imagen: {str(exc)}"
        )
    finally:
        temp_path = locals().get("temp_path")
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)
