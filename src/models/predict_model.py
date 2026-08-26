import os
import pathlib
import time

from fastai.vision.all import load_learner

# FastAI necesita esta función para reconstruir
# correctamente los modelos exportados.
from src.features.build_features import is_cat


# ============================================================
# COMPATIBILIDAD WINDOWS -> DOCKER / LINUX
# ============================================================

if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================

MODEL_CONFIG = {
    "resnet34": {
        "file": "clasificador.pkl",
        "architecture": "ResNet34",
        "validation_accuracy": 99.59,
        "validation_error_rate": 0.41,
        "correct_predictions": 1472,
        "validation_samples": 1478,
    },
    "efficientnet_b0": {
        "file": "clasificador_efficientnet_b0_e1_img224_bs64.pkl",
        "architecture": "EfficientNet-B0",
        "validation_accuracy": 97.77,
        "validation_error_rate": 2.23,
        "correct_predictions": 1445,
        "validation_samples": 1478,
    },
}


# Modelos cargados en memoria
_LOADED_MODELS = {}


# ============================================================
# ESTADO DE LOS MODELOS
# ============================================================

def get_models_status():
    """
    Devuelve información técnica de los modelos disponibles.
    """

    status = {}

    for model_name, config in MODEL_CONFIG.items():

        model_path = MODELS_DIR / config["file"]

        status[model_name] = {
            "architecture": config["architecture"],
            "file": config["file"],
            "exists": model_path.exists(),
            "loaded": model_name in _LOADED_MODELS,
            "validation_accuracy": config["validation_accuracy"],
            "validation_error_rate": config["validation_error_rate"],
            "correct_predictions": config["correct_predictions"],
            "validation_samples": config["validation_samples"],
        }

    return status


# ============================================================
# CARGAR MODELO
# ============================================================

def get_model(model_name="resnet34"):
    """
    Carga un modelo una sola vez.
    Después permanece almacenado en memoria.
    """

    if model_name not in MODEL_CONFIG:
        raise ValueError(
            f"Modelo no válido: {model_name}. "
            f"Disponibles: {list(MODEL_CONFIG.keys())}"
        )

    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    config = MODEL_CONFIG[model_name]

    model_path = MODELS_DIR / config["file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}"
        )

    print(
        f"Cargando {config['architecture']} "
        f"desde {model_path}..."
    )

    learner = load_learner(
        model_path,
        cpu=True
    )

    _LOADED_MODELS[model_name] = learner

    print(
        f"{config['architecture']} "
        f"cargado correctamente."
    )

    return learner


# ============================================================
# PREDICCIÓN
# ============================================================

def predict(image_path, model_name="resnet34"):
    """
    Ejecuta una predicción sobre una imagen.
    """

    learner = get_model(model_name)

    config = MODEL_CONFIG[model_name]

    start_time = time.perf_counter()

    pred_class, pred_idx, probabilities = learner.predict(
        image_path
    )

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    dog_probability = float(
        probabilities[0].item()
    ) * 100

    cat_probability = float(
        probabilities[1].item()
    ) * 100

    pred_value = pred_class

    if isinstance(pred_value, str):
        pred_value = (
            pred_value.strip().lower() == "true"
        )

    prediction = (
        "GATO"
        if bool(pred_value)
        else "PERRO"
    )

    confidence = float(
        probabilities[int(pred_idx)].item()
    ) * 100

    return {
        "model": model_name,
        "architecture": config["architecture"],
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probabilities": {
            "PERRO": round(dog_probability, 2),
            "GATO": round(cat_probability, 2),
        },
        "inference_time_ms": round(
            inference_time_ms,
            2
        ),
        "validation_accuracy": config[
            "validation_accuracy"
        ],
        "validation_error_rate": config[
            "validation_error_rate"
        ],
    }