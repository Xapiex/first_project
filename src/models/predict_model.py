from src.models.model_manager import ModelManager


# ============================================================
# ADMINISTRADOR POO
# ============================================================

# Se crea una sola instancia para toda la aplicación.
# Esta instancia administra ResNet34 y EfficientNet-B0.
_MODEL_MANAGER = ModelManager()


# ============================================================
# CONFIGURACIÓN COMPATIBLE CON FASTAPI
# ============================================================

# Conservamos MODEL_CONFIG porque FastAPI ya lo utiliza.
# Los datos ahora se obtienen de los objetos ImageClassifier.
MODEL_CONFIG = {
    model_name: {
        "file": classifier.model_file,
        "architecture": classifier.architecture,
        "validation_accuracy": classifier.validation_accuracy,
        "validation_error_rate": classifier.validation_error_rate,
        "correct_predictions": classifier.correct_predictions,
        "validation_samples": classifier.validation_samples,
    }
    for model_name, classifier in _MODEL_MANAGER.models.items()
}


# ============================================================
# ESTADO DE LOS MODELOS
# ============================================================

def get_models_status():
    """
    Devuelve información técnica de los modelos disponibles.

    Se conserva esta función para mantener compatibilidad
    con FastAPI, pero internamente utiliza ModelManager.
    """

    return _MODEL_MANAGER.get_models_status()


# ============================================================
# CARGAR MODELO
# ============================================================

def get_model(model_name="resnet34"):
    """
    Devuelve el Learner de FastAI correspondiente.

    Se conserva esta función para compatibilidad con
    cualquier código existente que espere el modelo cargado.
    """

    classifier = _MODEL_MANAGER.get_model(model_name)

    return classifier.load()


# ============================================================
# PREDICCIÓN
# ============================================================

def predict(image_path, model_name="resnet34"):
    """
    Ejecuta una predicción utilizando la estructura POO.
    """

    return _MODEL_MANAGER.predict(
        image_path,
        model_name=model_name
    )


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================

def compare_models(image_path):
    """
    Ejecuta la misma imagen con todos los modelos disponibles.
    """

    return _MODEL_MANAGER.compare_models(image_path)