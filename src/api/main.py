from fastapi import FastAPI

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
